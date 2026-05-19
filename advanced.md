# Course Advanced: Scratch IR-ResNet18 ArcFace

本文档记录 `baseline.md` 之后完成的课程要求进阶实验。进阶点放在模型构建上：把 baseline 的 `InceptionResnetV1(pretrained=None)` 替换为从头实现并从头训练的 InsightFace-style `IR-ResNet18`，训练目标仍为 ArcFace，数据、预处理和 LFW 评测协议保持一致，便于直接比较。

本实验不使用他人的人脸识别预训练权重。可以复用公开代码思想和 PyTorch/facenet-pytorch 工具，但最终 backbone 和 ArcFace head 都由本地训练得到。

## 进阶内容

Baseline：

```text
InceptionResnetV1(pretrained=None)
ArcFace head
CASIA-WebFace parquet mirror
LFW 6000-pair 10-fold evaluation
```

课程进阶：

```text
IR-ResNet18 backbone, pretrained weights: none
ArcFace head, 10,572 classes
same CASIA-WebFace training data
same LFW 6000-pair protocol
```

主要代码：

- `src/face_backbones.py`：新增 backbone builder、`IRBlock`、`IRResNet`。
- `src/train_casia_parquet_arcface.py`：新增 `--backbone`，支持 `inception_resnet_v1`、`ir_resnet18`、`ir_resnet34`。
- `src/evaluate_lfw.py`：评测时自动从 checkpoint 读取 backbone 类型，兼容旧 baseline checkpoint。

## 模型架构

`IR-ResNet18` 输入为 112x112 RGB 人脸图，输出 512 维 embedding。

```text
input: 3 x 112 x 112
stem: Conv3x3(3->64) + BN + PReLU
stage1: IRBlock x2, 64 channels, downsample to 56x56
stage2: IRBlock x2, 128 channels, downsample to 28x28
stage3: IRBlock x2, 256 channels, downsample to 14x14
stage4: IRBlock x2, 512 channels, downsample to 7x7
head: BN + Dropout(0.4) + Linear(512*7*7 -> 512) + BN
embedding: 512 dim
```

每个 `IRBlock` 使用 BN-Conv-BN-PReLU-Conv-BN 的残差分支，并按通道数和步长决定 shortcut。相比 baseline 的 InceptionResnetV1，这个结构更接近 ArcFace/InsightFace 常用的人脸识别残差骨干，参数量更大，训练期间也能更充分利用 RTX PRO 6000。

分类头仍为 ArcFace：

```text
ArcMarginProduct
in_features: 512
out_features: 10572
scale: 64.0
margin: 0.35
loss: cross entropy over ArcFace logits
```

## 数据预处理

训练阶段沿用 baseline，并符合 PPT “数据预处理”章节要求：

```text
parquet PNG bytes -> PIL RGB
RandomResizedCrop(112, scale=(0.86, 1.0))
RandomHorizontalFlip()
RandomRotation(10)
ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08)
PILToTensor()
fixed_image_standardization()
```

测试阶段使用 MTCNN 检测对齐到 112x112。为公平比较，进阶模型分别评测了 `mtcnn_margin=0` 和 `mtcnn_margin=16`。

## 训练方式

训练命令：

```bash
TORCH_HOME=/home/data1/data_mining/models/torch \
python src/train_casia_parquet_arcface.py \
  --backbone ir_resnet18 \
  --epochs 20 \
  --batch-size 512 \
  --num-workers 12 \
  --output-dir models/advanced_ir18_arcface \
  --arc-margin 0.35 \
  --lr 0.05 \
  --device cuda
```

训练配置：

| 项目 | 数值 |
| --- | --- |
| checkpoint | `models/advanced_ir18_arcface/epoch_020.pth` |
| backbone | `ir_resnet18` |
| pretrained backbone | `false` |
| CASIA 镜像样本数 | 490,592 |
| 身份类别 | 10,572 |
| epochs | 20 |
| batch size | 512 |
| workers | 12 |
| steps per epoch | 960 |
| optimizer | SGD + momentum + Nesterov |
| scheduler | CosineAnnealingLR |
| AMP | enabled |
| device | CUDA |

第 20 轮训练日志：

```text
train_loss: 4.654311
train_accuracy: 0.568587
samples_seen: 490592
```

`samples_seen=490592` 说明该轮覆盖了当前可下载 CASIA-WebFace 镜像的全部图像。训练期间 RTX PRO 6000 利用率可到约 99%，显存约 20GB，功耗约 474W。

## 评测方式

同 baseline：

- 使用作业给定 `design/lfw_test_pair.txt`。
- 共 6,000 对，其中 3,000 同人、3,000 异人。
- 对 7,701 张唯一图片提取 embedding。
- cosine similarity 作为 pair score。
- 10 折验证：每折 300 正 + 300 负，9 折选阈值，1 折测试。

评测命令，MTCNN margin 0：

```bash
TORCH_HOME=/home/data1/data_mining/models/torch \
python src/evaluate_lfw.py \
  --lfw-root data/raw/lfw-deepfunneled \
  --pairs-file design/lfw_test_pair.txt \
  --checkpoint models/advanced_ir18_arcface/epoch_020.pth \
  --preprocess mtcnn \
  --mtcnn-margin 0 \
  --image-size 112 \
  --batch-size 512 \
  --num-workers 0 \
  --device cuda \
  --output-dir results/advanced_ir18_arcface_lfw_epoch20
```

评测命令，MTCNN margin 16：

```bash
TORCH_HOME=/home/data1/data_mining/models/torch \
python src/evaluate_lfw.py \
  --lfw-root data/raw/lfw-deepfunneled \
  --pairs-file design/lfw_test_pair.txt \
  --checkpoint models/advanced_ir18_arcface/epoch_020.pth \
  --preprocess mtcnn \
  --mtcnn-margin 16 \
  --image-size 112 \
  --batch-size 512 \
  --num-workers 0 \
  --device cuda \
  --output-dir results/advanced_ir18_arcface_lfw_epoch20_margin16
```

## 结果

| 实验 | Backbone | MTCNN margin | LFW 10 折准确率 | ROC AUC | 混淆矩阵 `[[TN, FP], [FN, TP]]` |
| --- | --- | ---: | ---: | ---: | --- |
| baseline | InceptionResnetV1 | 0 | 84.8500% ± 1.1959% | 0.917717 | `[[2502, 498], [411, 2589]]` |
| baseline + 对齐边距 | InceptionResnetV1 | 16 | 86.4833% ± 1.8355% | 0.930206 | `[[2632, 368], [443, 2557]]` |
| 课程进阶 | IR-ResNet18 | 0 | 92.4333% ± 1.1624% | 0.965645 | `[[2865, 135], [319, 2681]]` |
| 课程进阶 + 对齐边距 | IR-ResNet18 | 16 | 94.1167% ± 0.9430% | 0.971984 | `[[2912, 88], [265, 2735]]` |

相对 `baseline.md` 中的严格 baseline，课程进阶在相同 `margin=0` 条件下提升：

```text
92.4333% - 84.8500% = +7.5833 percentage points
```

在当前最终评测配置 `margin=16` 下，相对 baseline 的 `margin=16` 结果提升：

```text
94.1167% - 86.4833% = +7.6334 percentage points
```

## 输出文件

模型元数据：

- `models/advanced_ir18_arcface/training_config.json`
- `models/advanced_ir18_arcface/history.json`
- `models/advanced_ir18_arcface/best_lfw.json`

本地权重：

- `models/advanced_ir18_arcface/epoch_020.pth`
- `models/advanced_ir18_arcface/best_lfw.pth -> epoch_020.pth`

`.pth` 权重约 225MB，不纳入普通 Git 跟踪。

评测结果：

- `results/advanced_ir18_arcface_lfw_epoch20/metrics.json`
- `results/advanced_ir18_arcface_lfw_epoch20/fold_metrics.csv`
- `results/advanced_ir18_arcface_lfw_epoch20/pair_scores.csv`
- `results/advanced_ir18_arcface_lfw_epoch20/roc_curve.png`
- `results/advanced_ir18_arcface_lfw_epoch20/confusion_matrix.png`
- `results/advanced_ir18_arcface_lfw_epoch20/score_histogram.png`
- `results/advanced_ir18_arcface_lfw_epoch20_margin16/metrics.json`
- `results/advanced_ir18_arcface_lfw_epoch20_margin16/fold_metrics.csv`
- `results/advanced_ir18_arcface_lfw_epoch20_margin16/pair_scores.csv`
- `results/advanced_ir18_arcface_lfw_epoch20_margin16/roc_curve.png`
- `results/advanced_ir18_arcface_lfw_epoch20_margin16/confusion_matrix.png`
- `results/advanced_ir18_arcface_lfw_epoch20_margin16/score_histogram.png`

`lfw_embeddings.npz` 是中间缓存，体积较大，不纳入 Git。

## 结论

课程要求进阶已完成：在不使用外部人脸识别预训练权重的前提下，IR-ResNet18 + ArcFace 从头训练显著超过 baseline。最终 LFW 6000 对 10 折准确率为 94.1167%，并输出 ROC、混淆矩阵、逐折结果和 pair score，可直接用于课程报告与展示。
