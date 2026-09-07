from .hf_common import HFBackend


class Backend(HFBackend):
    model_class = "HunYuanVLForConditionalGeneration"
    processor_options = {"backend": "pil"}

    def generation_options(self):
        # The archived config has pad_token_id=-1; use its actual tokenizer pad ID.
        return {"repetition_penalty": 1.08, "pad_token_id": self.processor.tokenizer.pad_token_id}
