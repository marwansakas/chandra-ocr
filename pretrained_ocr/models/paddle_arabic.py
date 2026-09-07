"""Recognition only: ground-truth images are already cropped to text lines."""

from .hf_common import Prediction


class Backend:
    def __init__(self, spec, options):
        from paddleocr import TextRecognition

        device = options.device
        if device == "auto":
            device = "cpu"
        elif device == "cuda":
            device = "gpu:0"
        elif device.startswith("cuda:"):
            device = device.replace("cuda:", "gpu:", 1)
        self.model = TextRecognition(model_name=spec.checkpoint, device=device, enable_mkldnn=False)
        self.info = {"device": device, "checkpoint": spec.checkpoint, "recognition_only": True}

    def predict(self, path):
        results = list(self.model.predict(input=str(path), batch_size=1))
        if len(results) != 1:
            raise RuntimeError(f"Expected one line result; received {len(results)}")
        result = results[0]
        # PaddleX Result exposes a JSON property with a 'res' object.
        payload = result.json if hasattr(result, "json") else result
        if isinstance(payload, str):
            import json
            payload = json.loads(payload)
        payload = payload.get("res", payload)
        if "rec_text" not in payload:
            raise RuntimeError("PaddleOCR returned no rec_text field")
        return Prediction(str(payload["rec_text"]), confidence=float(payload["rec_score"]))
