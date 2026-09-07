from .hf_common import HFBackend


class Backend(HFBackend):
    model_class = "GotOcr2ForConditionalGeneration"

    def make_inputs(self, image):
        return self.processor(images=image, return_tensors="pt", format=False)

    def generation_options(self):
        return {"tokenizer": self.processor.tokenizer, "stop_strings": "<|im_end|>"}
