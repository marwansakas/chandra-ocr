"""Public runner for the trained thesis TrOCR model."""

from line_runner_common import ROOT
from trocr_runner import run


if __name__ == "__main__":
    run(
        default_model=ROOT / "thesis" / "models" / "ground_truth_line_trocr",
        default_output=ROOT / "thesis" / "predictions" / "my_TrOCR",
        model_name="my_TrOCR",
    )
