# Models

Large model checkpoints are stored locally under this directory and are intentionally not suitable for normal Git tracking.

Best local scratch checkpoint:

```text
models/scratch_casia_arcface/epoch_021.pth
models/scratch_casia_arcface/best_lfw.pth -> epoch_021.pth
```

The checkpoint is about 221 MB. Use the training commands in `README.md` to reproduce it if it is not present.

Best local course-advanced checkpoint:

```text
models/advanced_ir18_arcface/epoch_020.pth
models/advanced_ir18_arcface/best_lfw.pth -> epoch_020.pth
```

This IR-ResNet18 checkpoint is about 225 MB. The `.pth` files remain ignored by Git; metadata files such as `history.json`, `training_config.json`, and `best_lfw.json` are tracked.

Best local self-advanced checkpoint:

```text
models/self_hsd_ir18_arcface/epoch_024.pth
models/self_hsd_ir18_arcface/best_lfw.pth -> epoch_024.pth
```

This hard-example self-distilled IR-ResNet18 checkpoint is about 225 MB. It starts from the locally trained course-advanced checkpoint and does not use external face-recognition pretrained weights.
