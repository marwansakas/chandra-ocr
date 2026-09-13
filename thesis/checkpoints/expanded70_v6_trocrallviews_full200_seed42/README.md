# expanded70 v6 best-checkpoint snapshot

This directory is a byte-for-byte snapshot of the best validation-CER checkpoint available while `expanded70_v6_trocrallviews_full200_seed42` was actively training on 13 September 2026.

- Selected epoch: 25
- Strict validation CER: 82.462517%
- Frozen dataset fingerprint: `544e5ae50ed41c3e4d53ec4d1dc6a4f266e154c7e1a4c05b5dfebfbd419db3f7`
- Frozen split SHA-256: `f82b01d6cdc3b564534a777939bcd2962be0b07fb009f60bce40c4178948eb73`
- Model SHA-256: `41fa4fb979a28a529ecf2c4b24abb1b6379af022376bea90e08ee9bb828cb22b`
- Training-state SHA-256: `f846c920edf507b44e1bdf5eea837dacab526204d259356badf8c22e03b3a177`

The dataset and image files are intentionally excluded. `run.json`, `best.json`, and the checkpoint manifest preserve the training and source provenance. `training_state.pt` is included so continuation is possible only with the same verified frozen data, configuration, and source snapshot.
