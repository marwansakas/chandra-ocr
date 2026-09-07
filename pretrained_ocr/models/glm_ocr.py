from .hf_common import HFBackend


class Backend(HFBackend):
    model_class = "GlmOcrForConditionalGeneration"
