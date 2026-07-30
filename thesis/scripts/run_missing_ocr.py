"""
Run Chandra OCR only for pages that are missing output.

This uses thesis/config.json paths:
  raw_data_dir      -> split_papers
  existing_ocr_dir  -> output
  chandra_method    -> hf / vllm

Run dry-run first:
  python thesis/scripts/run_missing_ocr.py --dry-run

Run OCR:
  python thesis/scripts/run_missing_ocr.py
"""

from __future__ import annotations

import argparse
import os
import signal
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import common


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--per-page-timeout",
        type=int,
        default=None,
        help="Maximum seconds to allow one Chandra subprocess to run.",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=int,
        default=60,
        help="Print a still-running message every N seconds while Chandra is quiet.",
    )
    parser.add_argument(
        "--notify-every",
        type=int,
        default=0,
        help="Print/send a progress notification every N attempted pages.",
    )
    parser.add_argument(
        "--notify-email",
        default=None,
        help="Email address for progress notifications. Requires mail/mailx on the node.",
    )
    return parser.parse_args()


def notify(message: str, email: str = None, subject: str = "Chandra OCR progress") -> None:
    timestamped = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(timestamped, flush=True)
    if not email:
        return

    mailer = shutil.which("mail") or shutil.which("mailx")
    if not mailer:
        print("Notification email skipped: mail/mailx was not found on this node.", flush=True)
        return

    subprocess.run(
        [mailer, "-s", subject, email],
        input=timestamped + "\n",
        text=True,
        check=False,
    )


def find_chandra(cfg: dict) -> str:
    """Find the Chandra executable from PATH or the configured conda env."""
    found = shutil.which("chandra")
    if found:
        return found

    conda_env = cfg.get("conda_env_chandra")
    if conda_env:
        candidate = Path(conda_env) / "bin" / "chandra"
        if candidate.exists():
            return str(candidate)

    raise SystemExit(
        "Could not find `chandra`. Activate the Chandra environment or check "
        "conda_env_chandra in thesis/config.json."
    )


def run_streaming(command, env, timeout=None, heartbeat_seconds=60) -> int:
    """
    Run Chandra with inherited stdout/stderr, plus heartbeat and robust timeout.

    subprocess.run(timeout=...) can leave child processes behind for tools that
    launch workers. Starting a process group lets us terminate the whole Chandra
    tree if a page hangs.
    """
    start = time.monotonic()
    next_heartbeat = start + max(1, heartbeat_seconds)
    process = subprocess.Popen(
        command,
        env=env,
        start_new_session=True,
    )

    while True:
        return_code = process.poll()
        if return_code is not None:
            return return_code

        now = time.monotonic()
        elapsed = int(now - start)
        if timeout is not None and elapsed >= timeout:
            print(
                f"Timeout reached after {elapsed}s. Stopping Chandra for this page...",
                flush=True,
            )
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                print("Chandra did not stop after SIGTERM; killing it now.", flush=True)
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            raise subprocess.TimeoutExpired(command, timeout)

        if heartbeat_seconds and now >= next_heartbeat:
            print(f"Still processing this page... elapsed={elapsed}s", flush=True)
            next_heartbeat = now + heartbeat_seconds

        time.sleep(2)


def main() -> None:
    args = parse_args()
    cfg = common.load_config()
    metadata_dir = common.resolve(cfg["metadata_dir"])
    raw_root = common.resolve(cfg["raw_data_dir"])
    ocr_root = common.resolve(cfg["existing_ocr_dir"])
    method = cfg.get("chandra_method", "hf")
    chandra = find_chandra(cfg)

    pages_path = metadata_dir / "pages.jsonl"
    if not pages_path.exists():
        raise SystemExit("Missing pages.jsonl. Run build_metadata.py first.")

    records = list(common.read_jsonl(pages_path))
    missing = [r for r in records if common.ocr_markdown_path(ocr_root, r) is None]
    if args.limit is not None:
        missing = missing[: args.limit]

    notify(f"Missing OCR pages selected: {len(missing)}", args.notify_email)
    if args.dry_run:
        for record in missing[:20]:
            print(record["image_path"])
        if len(missing) > 20:
            print(f"... {len(missing) - 20} more")
        return

    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    chandra_bin_dir = str(Path(chandra).parent)
    env["PATH"] = chandra_bin_dir + os.pathsep + env.get("PATH", "")
    succeeded = 0
    failed = 0

    for index, record in enumerate(missing, start=1):
        source = common.resolve(record["image_path"])
        rel = source.relative_to(raw_root)
        place, grade_folder, filename = rel.parts[-3:]
        stem = source.stem
        output_dir = ocr_root / place / grade_folder / stem
        output_dir.mkdir(parents=True, exist_ok=True)
        command = [chandra, str(source), str(output_dir), "--method", method]

        print(f"\n[{index}/{len(missing)}] {source}", flush=True)
        print(f"Output folder: {output_dir}", flush=True)
        print(f"Timeout: {args.per_page_timeout or 'none'} seconds", flush=True)
        try:
            return_code = run_streaming(
                command,
                env=env,
                timeout=args.per_page_timeout,
                heartbeat_seconds=args.heartbeat_seconds,
            )
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, command)
            succeeded += 1
        except subprocess.CalledProcessError as exc:
            failed += 1
            print(f"Failed with exit code {exc.returncode}: {' '.join(command)}")
            if not args.continue_on_error:
                raise
        except subprocess.TimeoutExpired:
            failed += 1
            print(f"Timed out after {args.per_page_timeout}s: {' '.join(command)}")
            if not args.continue_on_error:
                raise

        if args.notify_every and index % args.notify_every == 0:
            notify(
                f"Progress: attempted {index}/{len(missing)} pages; "
                f"succeeded={succeeded}; failed={failed}",
                args.notify_email,
            )

    notify(
        f"Finished missing OCR run. Attempted={len(missing)}; "
        f"succeeded={succeeded}; failed={failed}",
        args.notify_email,
        subject="Chandra OCR finished",
    )

    print(f"\nSucceeded: {succeeded}")
    print(f"Failed   : {failed}")
    print("\nNext: python thesis/scripts/collect_ocr.py")


if __name__ == "__main__":
    main()
