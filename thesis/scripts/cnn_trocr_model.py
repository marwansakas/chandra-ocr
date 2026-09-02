"""CNN-fronted TrOCR model utilities.

The CNN block is intentionally small and residual: it learns local handwriting
features while preserving the input shape expected by the existing TrOCR image
encoder.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


CNN_CONFIG_NAME = "cnn_trocr_config.json"
CNN_WEIGHTS_NAME = "cnn_block.pt"


@dataclass(frozen=True)
class CNNTrOCRConfig:
    input_channels: int = 3
    hidden_channels: int = 32
    layers: int = 2
    kernel_size: int = 3
    residual_scale: float = 0.1


def build_cnn_block(config: CNNTrOCRConfig):
    import torch

    if config.layers < 1:
        raise ValueError("CNN block must contain at least one layer")
    if config.kernel_size % 2 != 1:
        raise ValueError("CNN kernel size must be odd so image size is preserved")

    padding = config.kernel_size // 2
    modules: list[torch.nn.Module] = []
    in_channels = config.input_channels
    for _ in range(max(0, config.layers - 1)):
        modules.extend(
            [
                torch.nn.Conv2d(in_channels, config.hidden_channels, kernel_size=config.kernel_size, padding=padding),
                torch.nn.BatchNorm2d(config.hidden_channels),
                torch.nn.GELU(),
            ]
        )
        in_channels = config.hidden_channels
    modules.append(torch.nn.Conv2d(in_channels, config.input_channels, kernel_size=config.kernel_size, padding=padding))
    return torch.nn.Sequential(*modules)


class CNNTrOCRModel:
    def __init__(self, trocr_model, config: CNNTrOCRConfig | None = None):
        import torch

        self.config = config or CNNTrOCRConfig()
        self.trocr = trocr_model
        self.cnn = build_cnn_block(self.config)
        self.module = _CNNTrOCRModule(self.trocr, self.cnn, self.config.residual_scale)

    def to(self, device):
        self.module.to(device)
        return self

    def train(self):
        self.module.train()

    def eval(self):
        self.module.eval()

    def parameters(self):
        return self.module.parameters()

    @property
    def generation_config(self):
        return self.trocr.generation_config

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def generate(self, pixel_values, **kwargs):
        return self.module.generate(pixel_values, **kwargs)


class _CNNTrOCRModule:
    def __init__(self, trocr_model, cnn, residual_scale: float):
        import torch

        class Module(torch.nn.Module):
            def __init__(self, trocr_model, cnn, residual_scale: float):
                super().__init__()
                self.trocr = trocr_model
                self.cnn = cnn
                self.residual_scale = residual_scale

            def enhance_pixels(self, pixel_values):
                return pixel_values + self.residual_scale * self.cnn(pixel_values)

            def forward(self, pixel_values, **kwargs):
                return self.trocr(pixel_values=self.enhance_pixels(pixel_values), **kwargs)

            def generate(self, pixel_values, **kwargs):
                return self.trocr.generate(self.enhance_pixels(pixel_values), **kwargs)

        self._module = Module(trocr_model, cnn, residual_scale)

    def __getattr__(self, name):
        return getattr(self._module, name)

    def __call__(self, *args, **kwargs):
        return self._module(*args, **kwargs)


def save_cnn_trocr_model(model: CNNTrOCRModel, output_dir: Path) -> None:
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    model.trocr.save_pretrained(output_dir)
    torch.save(model.cnn.state_dict(), output_dir / CNN_WEIGHTS_NAME)
    (output_dir / CNN_CONFIG_NAME).write_text(
        json.dumps(asdict(model.config), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_cnn_trocr_model(model_dir: Path):
    import torch
    from transformers import VisionEncoderDecoderModel

    config_path = model_dir / CNN_CONFIG_NAME
    weights_path = model_dir / CNN_WEIGHTS_NAME
    if not config_path.is_file() or not weights_path.is_file():
        raise SystemExit(
            f"CNN-TrOCR checkpoint is missing {CNN_CONFIG_NAME} or {CNN_WEIGHTS_NAME}: {model_dir}"
        )
    config = CNNTrOCRConfig(**json.loads(config_path.read_text(encoding="utf-8")))
    trocr = VisionEncoderDecoderModel.from_pretrained(model_dir)
    model = CNNTrOCRModel(trocr, config)
    model.cnn.load_state_dict(torch.load(weights_path, map_location="cpu"))
    return model
