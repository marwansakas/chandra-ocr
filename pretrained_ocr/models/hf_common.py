"""Shared inference mechanics. Only image pixels and fixed prompts reach models."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Prediction:
    text: str
    generated_tokens: int | None = None
    hit_token_limit: bool = False
    confidence: float | None = None


class HFBackend:
    model_class = ""
    processor_class = "AutoProcessor"
    processor_options = {}

    def __init__(self, spec, options):
        import torch
        import transformers

        self.spec, self.options, self.torch = spec, options, torch
        requested = options.device
        self.device = "cuda:0" if requested == "auto" and torch.cuda.is_available() else (
            "cpu" if requested == "auto" else requested
        )
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable in this environment. Use --device cpu or install a CUDA PyTorch wheel.")
        dtype = options.dtype
        if dtype == "auto":
            if self.device.startswith("cuda"):
                with torch.cuda.device(self.device):
                    dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
            else:
                dtype = "float32"
        if self.device == "cpu" and dtype != "float32":
            raise ValueError("Use --dtype float32 (or auto) for the CPU baseline")
        self.dtype = getattr(torch, dtype)
        source = dict(revision=spec.revision, subfolder=spec.subfolder, trust_remote_code=False)
        self.processor = getattr(transformers, self.processor_class).from_pretrained(
            spec.checkpoint, **source, **self.processor_options,
        )
        self.model = getattr(transformers, self.model_class).from_pretrained(
            spec.checkpoint, **source, dtype=self.dtype,
            attn_implementation=options.attention,
        ).to(self.device).eval()
        self.info = {"device": self.device, "dtype": dtype, "attention": options.attention,
                     "resolved_revision": getattr(self.model.config, "_commit_hash", spec.revision)}

    def make_inputs(self, image):
        content = [{"type": "image", "image": image}]
        if self.spec.prompt:
            content.append({"type": "text", "text": self.spec.prompt})
        return self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
        )

    def generation_options(self):
        return {}

    def predict(self, path: Path):
        from PIL import Image, ImageOps

        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        inputs = self.make_inputs(image)
        inputs.pop("token_type_ids", None)
        inputs = {k: v.to(device=self.device, dtype=self.dtype) if v.is_floating_point()
                  else v.to(self.device) for k, v in inputs.items()}
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs, do_sample=False, max_new_tokens=self.options.max_new_tokens,
                **self.generation_options(),
            )
        generated = output[0, inputs["input_ids"].shape[1]:]
        return Prediction(
            self.processor.decode(generated, skip_special_tokens=True), len(generated),
            len(generated) >= self.options.max_new_tokens,
        )
