from .hf_common import HFBackend


class Backend(HFBackend):
    model_class = "LightOnOcrForConditionalGeneration"
    processor_class = "LightOnOcrProcessor"
