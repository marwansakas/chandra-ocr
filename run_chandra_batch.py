from pathlib import Path
import subprocess
import sys
import os


# ==========================
# Configuration
# ==========================

SOURCE_PATH = Path("Exams")
DESTINATION_PATH = Path("ExamOutput")

# Choose OCR method: "hf" or "vllm"
# "hf" can use GPU if the installed PyTorch supports CUDA.
# "vllm" usually expects a GPU and is often faster, but requires vLLM to be installed correctly.
CHANDRA_METHOD = "hf"

# GPU settings
USE_GPU_IF_AVAILABLE = True

# Use "0" for the first GPU.
# If you are on a cluster and Slurm already assigns a GPU,
# keep this as None so you do not override Slurm.
CUDA_VISIBLE_DEVICES = None
# Example for local machine:
# CUDA_VISIBLE_DEVICES = "0"

# Supported files
SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".tif",
    ".tiff",
    ".bmp",
}

PAGE_RANGE = None

CONTINUE_ON_ERROR = True

SKIP_EXISTING_OUTPUT = True


def is_supported_file(file_path: Path) -> bool:
    return file_path.suffix.lower() in SUPPORTED_EXTENSIONS


def check_gpu_available() -> bool:
    """
    Checks whether Python/PyTorch can see a CUDA GPU.
    This does not guarantee Chandra will use it, but it confirms
    the environment is GPU-ready.
    """

    if not USE_GPU_IF_AVAILABLE:
        print("GPU usage disabled by configuration.")
        return False

    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"CUDA GPU is available: {gpu_name}")
            return True

        print("CUDA GPU is NOT available to PyTorch. Chandra may run on CPU.")
        return False

    except ImportError:
        print("PyTorch is not installed in this environment.")
        print("Chandra may still run, but GPU availability could not be checked.")
        return False


def build_chandra_environment() -> dict:
    """
    Builds environment variables for the Chandra subprocess.
    """

    env = os.environ.copy()

    if USE_GPU_IF_AVAILABLE:
        # If CUDA_VISIBLE_DEVICES is set manually, use it.
        # If it is None, preserve the current environment.
        if CUDA_VISIBLE_DEVICES is not None:
            env["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

        # Helpful for CUDA memory fragmentation in PyTorch-based workloads.
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    else:
        # Force CPU only.
        env["CUDA_VISIBLE_DEVICES"] = ""

    return env


def output_already_exists(output_dir: Path, input_file: Path) -> bool:
    """
    Checks whether this file appears to have already been processed.
    """

    if not output_dir.exists():
        return False

    stem = input_file.stem
    candidate_dirs = (output_dir, output_dir / stem)
    for candidate_dir in candidate_dirs:
        markdown = candidate_dir / f"{stem}.md"
        metadata = candidate_dir / f"{stem}_metadata.json"
        if markdown.is_file() and metadata.is_file():
            return True
    return False


def run_chandra_ocr(input_file: Path, output_dir: Path, env: dict) -> bool:
    """
    Runs Chandra OCR on one PDF/image file.

    Returns True if successful, False otherwise.
    """

    if SKIP_EXISTING_OUTPUT and output_already_exists(output_dir, input_file):
        print(f"Skipping already processed file: {input_file}")
        print(f"Existing output folder: {output_dir}")
        return True

    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        "chandra",
        str(input_file),
        str(output_dir),
        "--method",
        CHANDRA_METHOD,
    ]

    if PAGE_RANGE is not None:
        command.extend(["--page-range", PAGE_RANGE])

    print(f"\nRunning OCR on: {input_file}")
    print(f"Output folder: {output_dir}")
    print(f"Method: {CHANDRA_METHOD}")

    if USE_GPU_IF_AVAILABLE:
        visible_devices = env.get("CUDA_VISIBLE_DEVICES", "not manually set")
        print(f"CUDA_VISIBLE_DEVICES: {visible_devices}")
    else:
        print("Running with GPU disabled.")

    try:
        result = subprocess.run(
            command,
            check=True,
            text=True,
            capture_output=True,
            env=env,
        )

        print("Success.")
        if result.stdout.strip():
            print(result.stdout)

        return True

    except subprocess.CalledProcessError as error:
        print("Failed.")
        print(f"Command: {' '.join(command)}")

        if error.stdout:
            print("\nSTDOUT:")
            print(error.stdout)

        if error.stderr:
            print("\nSTDERR:")
            print(error.stderr)

        return False


def process_all_files(source_path: Path, destination_path: Path) -> None:
    if not source_path.exists():
        raise FileNotFoundError(f"Source path does not exist: {source_path}")

    if not source_path.is_dir():
        raise NotADirectoryError(f"Source path must be a folder: {source_path}")

    gpu_available = check_gpu_available()

    if USE_GPU_IF_AVAILABLE and not gpu_available:
        print("\nWarning: GPU was requested, but CUDA is not available.")
        print("The program will probably run on CPU unless Chandra handles GPU differently.\n")

    env = build_chandra_environment()

    files = [
        file_path
        for file_path in source_path.rglob("*")
        if file_path.is_file() and is_supported_file(file_path)
    ]

    if not files:
        print("No supported PDF or image files found.")
        return

    total_files = len(files)
    successful_files = 0
    failed_files = 0
    skipped_files = 0

    print(f"Found {total_files} supported files.")

    for index, file_path in enumerate(files, start=1):
        print(f"\n==============================")
        print(f"File {index}/{total_files}")
        print(f"==============================")

        relative_parent = file_path.parent.relative_to(source_path)
        mirrored_output_dir = destination_path / relative_parent / file_path.stem

        was_already_processed = SKIP_EXISTING_OUTPUT and output_already_exists(
            mirrored_output_dir,
            file_path,
        )

        success = run_chandra_ocr(file_path, mirrored_output_dir, env)

        if was_already_processed:
            skipped_files += 1
        elif success:
            successful_files += 1
        else:
            failed_files += 1

            if not CONTINUE_ON_ERROR:
                print("Stopping because CONTINUE_ON_ERROR is False.")
                break

    print("\n========== Final Summary ==========")
    print(f"Total files found: {total_files}")
    print(f"Successful OCR files this run: {successful_files}")
    print(f"Skipped already processed files: {skipped_files}")
    print(f"Failed OCR files: {failed_files}")
    print(f"Output saved to: {destination_path}")


if __name__ == "__main__":
    process_all_files(SOURCE_PATH, DESTINATION_PATH)
