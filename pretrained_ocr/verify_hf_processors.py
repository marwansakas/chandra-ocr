"""Optional online API smoke check; downloads processors/configs, never model weights.

Run: python -m pretrained_ocr.verify_hf_processors
"""

from importlib import import_module

from PIL import Image
import torch
import transformers

from .registry import MODELS


def main():
    print(f"Transformers {transformers.__version__}", flush=True)
    for key, spec in MODELS.items():
        if key == "paddle-arabic":
            continue
        kind = import_module(f"pretrained_ocr.models.{spec.module}").Backend
        model_class = getattr(transformers, kind.model_class)
        source = dict(revision=spec.revision, subfolder=spec.subfolder, trust_remote_code=False)
        # Match the concrete class used at inference, including legacy checkpoint configs.
        config = model_class.config_class.from_pretrained(spec.checkpoint, **source)
        with torch.device("meta"):
            model = model_class(config)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        del model
        backend = kind.__new__(kind)
        backend.spec = spec
        backend.processor = getattr(transformers, kind.processor_class).from_pretrained(
            spec.checkpoint, **source, **kind.processor_options,
        )
        inputs = backend.make_inputs(Image.new("RGB", (512, 64), "white"))
        print(f"{key}: {model_class.__name__}; {config.model_type}; {parameters:,} parameters; " +
              str({key: tuple(value.shape) for key, value in inputs.items()}), flush=True)


if __name__ == "__main__":
    main()
