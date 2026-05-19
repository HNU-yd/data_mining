# Baseline: Scratch CASIA-WebFace Face Verification

本文档记录进阶实验之前的 baseline。它的目标是提供一条完整、可复现、可比较的人脸验证流水线：从 CASIA-WebFace 镜像训练一个本地权重，再在作业给定的 LFW 6000 对上做 10 折验证。

该 baseline 不使用他人的人脸识别预训练权重。模型骨干从 `pretrained=None` 初始化，外部代码库只作为模型结构和工具函数来源。

## Baseline 定位

这个 baseline 只包含完成作业基本流程所需的组件：

- CASIA-WebFace 训练数据读取和增强。
- 一个固定的人脸特征模型。
- ArcFace 分类训练。
- LFW 6000 对验证。
- ROC、混淆矩阵、pair score 等基础分析输出。

不属于 baseline 的内容：

- 不使用外部人脸识别预训练 checkpoint 作为主结果。
- 不使用更强的 IR-ResNet、SE-ResNet、Partial FC 等进阶结构。
- 不使用测试时增强作为默认结果。
- 不把 MTCNN margin 调参后的结果算进 baseline；`margin=16` 是后续进阶改进。

## 数据

训练集使用 CASIA-WebFace parquet 镜像：

```text
data/raw/casia_webface_parquet
```

当前镜像统计：

```text
images: 490,592
classes: 10,572
label range: 0..10571
image format: RGB PNG bytes, 112x112 aligned face
```

PPT 标称 CASIA-WebFace 为 494,414 张、10,575 人。当前公开镜像少 3,822 张和 3 个身份标签，已在 `README.md` 和 `STATUS.md` 中说明。

测试集使用 LFW deepfunneled：

```text
data/raw/lfw-deepfunneled
design/lfw_test_pair.txt
```

测试对统计：

```text
pairs: 6,000
positive pairs: 3,000
negative pairs: 3,000
unique images used by pairs: 7,701
```

## 数据预处理

训练阶段：

```text
parquet PNG bytes -> PIL RGB
RandomResizedCrop(112, scale=(0.86, 1.0))
RandomHorizontalFlip()
RandomRotation(10)
ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08)
PILToTensor()
fixed_image_standardization()
```

数据加载：

- 20 个 parquet shard。
- shard 顺序随机打乱。
- row group 顺序随机打乱。
- row group 内样本随机打乱。
- shuffle buffer：16,384。
- batch size：512。
- workers：12。

LFW baseline 评测阶段：

```text
MTCNN face detection
image_size = 112
mtcnn_margin = 0
fixed_image_standardization by MTCNN post_process
```

`mtcnn_margin=0` 是 baseline 默认设置。后续进阶实验将 margin 调整为 16，使准确率从 84.85% 提升到 86.48%。

## 模型架构

Baseline 模型由两部分组成。

Backbone：

```text
InceptionResnetV1(pretrained=None, classify=False)
input: 3 x 112 x 112
output: 512-d embedding
```

Classification head：

```text
ArcMarginProduct
in_features: 512
out_features: 10,572
scale: 64.0
margin: 0.35 for the final baseline checkpoint stage
```

训练时使用 ArcFace logits 做 cross entropy；评测时丢弃分类头，只使用 backbone 输出 embedding。LFW 图像对分数为两个 L2 normalized embedding 的 cosine similarity。

整体流程：

```text
CASIA image
  -> augmentation
  -> InceptionResnetV1 backbone
  -> 512-d embedding
  -> ArcFace head
  -> cross entropy

LFW image pair
  -> MTCNN align
  -> backbone
  -> normalized embeddings
  -> cosine similarity
  -> threshold by 10-fold protocol
```

## 运行方式

### 1. 激活环境

```bash
cd /home/data1/data_mining
conda activate data_mining
```

如果环境不存在：

```bash
bash setup_env.sh
conda activate data_mining
```

### 2. 下载数据

```bash
python src/download_lfw.py
python src/download_casia_webface.py
```

### 3. 训练 baseline 权重

第一阶段：从随机初始化训练 10 轮。

```bash
TORCH_HOME=/home/data1/data_mining/models/torch \
python src/train_casia_parquet_arcface.py \
  --epochs 10 \
  --batch-size 512 \
  --num-workers 8 \
  --output-dir models/scratch_casia_arcface \
  --device cuda
```

第二阶段：从第 10 轮继续训练 10 轮，降低 ArcFace margin。

```bash
TORCH_HOME=/home/data1/data_mining/models/torch \
python src/train_casia_parquet_arcface.py \
  --epochs 10 \
  --resume models/scratch_casia_arcface/latest.pth \
  --batch-size 512 \
  --num-workers 12 \
  --output-dir models/scratch_casia_arcface \
  --arc-margin 0.35 \
  --lr 0.02 \
  --device cuda
```

第三阶段：补跑 1 轮，修正多 worker 尾批计数，确保当前镜像全部 490,592 张都在该轮被读取。

```bash
TORCH_HOME=/home/data1/data_mining/models/torch \
python src/train_casia_parquet_arcface.py \
  --epochs 1 \
  --resume models/scratch_casia_arcface/latest.pth \
  --batch-size 512 \
  --num-workers 12 \
  --output-dir models/scratch_casia_arcface \
  --arc-margin 0.35 \
  --lr 0.002 \
  --device cuda
```

Baseline checkpoint：

```text
models/scratch_casia_arcface/epoch_021.pth
```

第 21 轮训练日志：

```text
samples_seen: 490592
```

### 4. 评测 baseline

```bash
TORCH_HOME=/home/data1/data_mining/models/torch \
python src/evaluate_lfw.py \
  --lfw-root data/raw/lfw-deepfunneled \
  --pairs-file design/lfw_test_pair.txt \
  --checkpoint models/scratch_casia_arcface/epoch_021.pth \
  --preprocess mtcnn \
  --mtcnn-margin 0 \
  --image-size 112 \
  --batch-size 512 \
  --num-workers 0 \
  --device cuda \
  --output-dir results/scratch_casia_arcface_lfw_epoch21
```

输出目录：

```text
results/scratch_casia_arcface_lfw_epoch21
```

主要输出文件：

- `metrics.json`
- `fold_metrics.csv`
- `pair_scores.csv`
- `roc_curve.png`
- `confusion_matrix.png`
- `score_histogram.png`

`lfw_embeddings.npz` 也会生成，但体积较大，不纳入 Git。

## Baseline 指标

评测配置：

```text
checkpoint: models/scratch_casia_arcface/epoch_021.pth
preprocess: mtcnn
mtcnn_margin: 0
tta_flip: false
image_size: 112
```

LFW 6000 对 10 折结果：

| 指标 | 数值 |
| --- | ---: |
| 10 折准确率 | 84.8500% ± 1.1959% |
| ROC AUC | 0.917717 |
| 全局最优准确率 | 85.1333% |
| 全局最优阈值 | 0.983750 |
| MTCNN 检测成功率 | 99.9870% |

10 折混淆矩阵 `[[TN, FP], [FN, TP]]`：

```text
[[2502,  498],
 [ 411, 2589]]
```

逐折准确率：

```text
85.83%, 84.00%, 84.00%, 85.33%, 82.33%,
85.67%, 84.33%, 86.83%, 84.67%, 85.50%
```

## 与进阶实验的分界

Baseline 的 10 折均值为 84.85%，接近但未稳定超过 85%。这说明训练和验证链路已经有效，但还需要进阶改进。

后续进阶实验可以从这里开始比较：

- LFW MTCNN margin 从 0 调整到 16。
- 更强 backbone，例如 IR-ResNet。
- 更稳定的 ArcFace/CosFace 超参数。
- 更长训练或更好的学习率计划。
- WebDataset/LMDB 缓存，减少 parquet/PNG 解码瓶颈。
- 测试时增强和多 crop，但需要单独记录为进阶设置。

已完成的第一个进阶对照：

```text
checkpoint: models/scratch_casia_arcface/epoch_021.pth
preprocess: mtcnn
mtcnn_margin: 16
LFW 10-fold accuracy: 86.4833% ± 1.8355%
output: results/scratch_casia_arcface_lfw_epoch21_margin16
```

因此，`results/scratch_casia_arcface_lfw_epoch21` 是 baseline 结果，`results/scratch_casia_arcface_lfw_epoch21_margin16` 是在 baseline 之上的进阶改进结果。
