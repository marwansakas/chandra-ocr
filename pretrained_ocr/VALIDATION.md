# Validation performed September 7, 2026

Environment: Windows, Python 3.12; CPU verification.

- Offline dataset, Arabic scoring, output protection, resume, comparison and adapter
  tests pass (12 tests). Inference in these tests uses a fake backend.
- PyTorch 2.10.0+cpu, Torchvision 0.25.0+cpu and Transformers 5.16.1 install together;
  `pip check` passes.
- All four real Hugging Face processors/configurations download at the pinned
  revisions and produce tensors from a synthetic 512 x 64 image. Their concrete
  model classes initialize on the meta device (architecture check without weights).
- PaddlePaddle 3.3.1 and PaddleOCR 3.7.0 install in a separate environment. The real
  `arabic_PP-OCRv5_mobile_rec` weights download and complete a CPU prediction and
  report-writing run on one synthetic printed Latin/digit line. This is a plumbing
  smoke test, not a handwriting or Arabic accuracy measurement.
- Source dataset fixtures remain byte-for-byte unchanged in the runner tests.

The processor checks exposed a Torchvision requirement and a model-specific
processor option incompatibility; both are handled by the setup instructions and
adapters. The archived Hunyuan config contains `pad_token_id=-1`; generation uses
the valid pad ID from its tokenizer. The LightOn checkpoint retains a legacy
`mistral3` configuration tag; its dedicated LightOn model class is used as documented
by the publisher. Those legacy configurations may emit upstream warnings.

Not performed: downloading/inferencing the four full VLM checkpoints, CUDA
execution, or scoring the user's Linux dataset. The BGU host did not accept SSH
authentication from this session. Run the documented smoke/full commands there to
verify checkpoint loading, resource availability and actual CER/WER.
