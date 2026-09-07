import argparse
import unittest
from unittest.mock import patch

from pretrained_ocr.models import paddle_arabic
from pretrained_ocr.models.got_ocr2 import Backend as GotBackend
from pretrained_ocr.models.glm_ocr import Backend as GlmBackend
from pretrained_ocr.models.hunyuan_ocr import Backend as HunyuanBackend
from pretrained_ocr.models.lighton_ocr2 import Backend as LightOnBackend
from pretrained_ocr.registry import MODELS


class AdapterContractTests(unittest.TestCase):
    def test_got_plain_text_and_stop_token(self):
        backend = GotBackend.__new__(GotBackend)
        seen = {}
        class Processor:
            tokenizer = object()
            def __call__(self, **kwargs):
                seen.update(kwargs)
                return kwargs
        backend.processor = Processor()
        backend.make_inputs("pixels")
        self.assertEqual(seen, {"images": "pixels", "return_tensors": "pt", "format": False})
        self.assertEqual(backend.generation_options()['stop_strings'], '<|im_end|>')

    def test_glm_task_prompt_and_lighton_image_only(self):
        for kind, key, expected in [(GlmBackend, 'glm-ocr', 2), (LightOnBackend, 'lighton-ocr2', 1)]:
            backend = kind.__new__(kind)
            backend.spec = MODELS[key]
            backend.processor = argparse.Namespace(apply_chat_template=lambda messages, **kwargs: messages)
            messages = backend.make_inputs("pixels")
            self.assertEqual(len(messages[0]['content']), expected)
            self.assertEqual(messages[0]['content'][0], {"type": "image", "image": "pixels"})
        self.assertEqual(MODELS['glm-ocr'].prompt, 'Text Recognition:')

    def test_paddle_result_json(self):
        backend = paddle_arabic.Backend.__new__(paddle_arabic.Backend)
        result = argparse.Namespace(json={"res": {"rec_text": "كتاب", "rec_score": 0.8}})
        backend.model = argparse.Namespace(predict=lambda **kwargs: iter([result]))
        prediction = backend.predict("image.png")
        self.assertEqual(prediction.text, 'كتاب')
        self.assertEqual(prediction.confidence, 0.8)

    def test_hunyuan_uses_valid_tokenizer_padding(self):
        backend = HunyuanBackend.__new__(HunyuanBackend)
        backend.processor = argparse.Namespace(tokenizer=argparse.Namespace(pad_token_id=120002))
        self.assertEqual(backend.generation_options()['pad_token_id'], 120002)


if __name__ == '__main__':
    unittest.main()
