"""Published checkpoints verified on 2026-09-07; no tier claims are assumed."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    module: str
    checkpoint: str
    revision: str = ""
    subfolder: str = ""
    prompt: str = ""


MODELS = {
    "paddle-arabic": ModelSpec(
        "paddle-arabic", "PaddleOCR Arabic PP-OCRv5", "paddle_arabic",
        "arabic_PP-OCRv5_mobile_rec",
    ),
    "got-ocr2": ModelSpec(
        "got-ocr2", "GOT-OCR2.0", "got_ocr2", "stepfun-ai/GOT-OCR-2.0-hf",
        "d3017ef2c2c1395888c8d635c5e0508bcb0ac78d",
    ),
    "glm-ocr": ModelSpec(
        "glm-ocr", "GLM-OCR", "glm_ocr", "zai-org/GLM-OCR",
        "ca5d8b3e287e52589e37c28385d9655ee4372f9d", prompt="Text Recognition:",
    ),
    "hunyuan-ocr": ModelSpec(
        "hunyuan-ocr", "HunyuanOCR 1.0", "hunyuan_ocr", "tencent/HunyuanOCR",
        "47644ecc4fc854efa4f505155158831f36773ee4", subfolder="v1.0",
        prompt="Extract all the text in the image. Output only the original text, without translation or explanation.",
    ),
    "lighton-ocr2": ModelSpec(
        "lighton-ocr2", "LightOnOCR-2", "lighton_ocr2", "lightonai/LightOnOCR-2-1B",
        "c97bd377f04481830395218fa8951df9deaba756",
    ),
}
