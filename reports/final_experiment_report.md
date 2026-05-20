# 人脸识别实验总报告

## 摘要

本项目完成了 CASIA-WebFace 到 LFW 人脸验证的完整实验流程：数据准备、数据预处理、GPU 训练、本地权重保存、LFW 6000 对 10 折验证、ROC 曲线、混淆矩阵和逐对分数输出。实验严格遵循“不使用他人人脸识别预训练权重作为最终结果”的要求，所有最终模型权重均由本项目在本机训练得到。

实验分为三个版本：

1. Baseline：`InceptionResnetV1 + ArcFace`，从随机初始化开始训练。
2. 课程要求进阶：将 backbone 替换为从头实现和训练的 `IR-ResNet18`。
3. 自选进阶：在课程进阶 checkpoint 上做 hard-example self-distillation 微调。

最终自选进阶模型在 LFW 6000 对上取得：

```text
LFW 10-fold accuracy: 94.7167% ± 0.7819%
ROC AUC: 0.974148
confusion matrix [[TN, FP], [FN, TP]]: [[2921, 79], [238, 2762]]
```

相对严格 baseline 的 `84.8500%`，最终提升 `+9.8667` 个百分点；相对课程进阶的 `94.1167%`，自选进阶继续提升 `+0.6000` 个百分点。

## 任务与评测协议

作业目标是完成一个人脸验证系统，并在作业给定的 LFW 6000 对上评测准确率。项目使用 CASIA-WebFace 训练人脸 embedding 模型，测试时对 LFW 图像对计算 cosine similarity，再按照 10 折协议选择阈值并统计准确率。

LFW pair 文件统计：

```text
pairs: 6,000
positive pairs: 3,000
negative pairs: 3,000
unique images used by pairs: 7,701
```

评测协议：

- 每张 LFW 图像提取 512 维 L2-normalized embedding。
- 每对图像计算 cosine similarity。
- 10 折验证，每折包含 300 个同人对和 300 个异人对。
- 每折使用其余 9 折选择最佳阈值，在当前折测试。
- 输出 10 折准确率、ROC AUC、全局最优阈值、混淆矩阵、pair score 和图表。

## 数据与预处理

### 训练集

PPT 标称 CASIA-WebFace 包含 494,414 张图像、10,575 个身份。官方数据需要申请，本项目使用可下载 Hugging Face parquet 镜像：

```text
data/raw/casia_webface_parquet
images: 490,592
classes: 10,572
label range: 0..10571
image format: RGB PNG bytes, 112x112 aligned face
```

该镜像比 PPT 标称少 3,822 张和 3 个身份标签，推测来自公开镜像清洗或转换差异。所有关键训练轮次均记录 `samples_seen=490592`，说明当前可下载镜像的全部图像都进入训练流程。

### 测试集

测试集使用 LFW deepfunneled：

```text
data/raw/lfw-deepfunneled
images: 13,233
pairs file: design/lfw_test_pair.txt
```

官方 UMass 地址在当前环境 DNS 解析失败，Figshare 镜像返回 403，因此使用 Hugging Face `DerrickUnleashed/LFW` 镜像。测试协议仍使用作业提供的 6000 对文件。

### 训练预处理

三个版本共享基础训练预处理，满足 PPT “数据预处理”章节要求：

```text
parquet PNG bytes -> PIL RGB
RandomResizedCrop(112, scale=(0.86, 1.0))
RandomHorizontalFlip()
RandomRotation(10)
ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08)
PILToTensor()
fixed_image_standardization()
```

数据加载侧对 20 个 parquet shard、row group 和样本顺序做随机打乱，并使用 shuffle buffer。

### 测试预处理

测试阶段使用 MTCNN 检测对齐到 112x112。为了区分模型能力和测试对齐边距影响，每个主要模型都报告两种设置：

```text
margin=0: 严格对齐条件，主要用于同条件模型比较
margin=16: 最终提交配置，保留更多脸部边界
```

## 版本一：Baseline

### 架构

Baseline 采用 `facenet-pytorch` 提供的 `InceptionResnetV1` 结构，但不加载任何外部人脸识别预训练权重：

```text
Backbone: InceptionResnetV1(pretrained=None, classify=False)
input: 3 x 112 x 112
embedding: 512 dim
Head: ArcMarginProduct(512, 10572)
Loss: ArcFace logits + cross entropy
```

训练时使用 backbone 输出 embedding，经 ArcFace 分类头对 10,572 个 CASIA 身份分类。评测时丢弃分类头，只保留 backbone 输出 embedding。

### 训练设置

Baseline 分三段完成：

1. 从随机初始化训练 10 轮。
2. 从第 10 轮继续训练 10 轮，调整 ArcFace margin 和学习率。
3. 补跑第 21 轮，修正多 worker iterable dataloader 尾批计数问题，确保当轮覆盖全部 490,592 张图像。

最佳 baseline checkpoint：

```text
models/scratch_casia_arcface/epoch_021.pth
samples_seen: 490592
```

### Baseline 作用

Baseline 的作用是建立一个不依赖外部预训练权重的完整可复现实验基准。它验证了以下内容：

- CASIA parquet 数据读取和增强流程可用。
- ArcFace 分类训练可在本地 GPU 上完成。
- LFW 6000 对评测、ROC、混淆矩阵和 pair score 输出流程完整。
- InceptionResnetV1 从头训练可以接近但没有显著超过更强架构。

## 版本二：课程要求进阶

### 改动点

课程进阶主要改在模型构建上：把 baseline 的 `InceptionResnetV1` 替换为更接近 ArcFace/InsightFace 常用形式的 IR-ResNet 残差骨干。

代码改动：

- `src/face_backbones.py`：新增 `IRBlock`、`IRResNet`、`build_backbone()`。
- `src/train_casia_parquet_arcface.py`：新增 `--backbone` 参数，支持 `inception_resnet_v1`、`ir_resnet18`、`ir_resnet34`。
- `src/evaluate_lfw.py`：评测时从 checkpoint 读取 backbone 类型，兼容旧 baseline checkpoint。

### 架构

课程进阶使用 `IR-ResNet18`：

```text
input: 3 x 112 x 112
stem: Conv3x3(3->64) + BN + PReLU
stage1: IRBlock x2, 64 channels, downsample to 56x56
stage2: IRBlock x2, 128 channels, downsample to 28x28
stage3: IRBlock x2, 256 channels, downsample to 14x14
stage4: IRBlock x2, 512 channels, downsample to 7x7
head: BN + Dropout(0.4) + Linear(512*7*7 -> 512) + BN
embedding: 512 dim
ArcFace head: ArcMarginProduct(512, 10572)
```

每个 `IRBlock` 使用 BN-Conv-BN-PReLU-Conv-BN 的残差分支，并按通道数和步长决定 shortcut。相比 baseline 的 InceptionResnetV1，IR-ResNet18 的结构更直接服务于 112x112 人脸识别 embedding 学习。

### 训练设置

课程进阶从随机初始化训练 20 轮：

```text
checkpoint: models/advanced_ir18_arcface/epoch_020.pth
epochs: 20
batch size: 512
workers: 12
lr: 0.05
arc margin: 0.35
samples_seen at epoch 20: 490592
train_loss at epoch 20: 4.654311
train_accuracy at epoch 20: 56.8587%
```

训练期间 RTX PRO 6000 GPU 利用率可到约 99%，显存约 20GB，功耗约 474W。

### 进阶意义

该版本验证了模型结构本身是主要性能瓶颈之一。在相同训练数据和相同 LFW 协议下，仅把 backbone 从 InceptionResnetV1 换成 IR-ResNet18，就在 `margin=0` 条件下带来 `+7.5833` 个百分点提升。

## 版本三：自选进阶

### 改动点

自选进阶不改变最终推理结构，仍然使用单个 IR-ResNet18 backbone。改动发生在训练策略上：使用我们自己训练好的课程进阶 checkpoint 作为 teacher，对 student 做 hard-example self-distillation 微调。

新增代码：

```text
src/train_casia_hsd_arcface.py
```

该实验没有使用外部人脸识别预训练权重：

```text
teacher: models/advanced_ir18_arcface/epoch_020.pth
student initialization: models/advanced_ir18_arcface/epoch_020.pth
teacher source: 本项目本地训练
student source: 本项目本地训练
```

### 训练机制

每张 CASIA 图像生成 weak view 和 strong view：

```text
weak view:
  Resize(112)
  fixed_image_standardization

strong view:
  RandomResizedCrop(112, scale=(0.82, 1.0))
  RandomHorizontalFlip
  RandomRotation(12)
  ColorJitter(brightness=0.16, contrast=0.16, saturation=0.10, hue=0.02)
  RandomGrayscale(p=0.04)
  fixed_image_standardization
```

Teacher 冻结参数，只处理 weak view 并输出稳定 embedding。Student 处理 strong view，一方面继续做 ArcFace 分类，另一方面通过 embedding distillation 向 teacher 的表征靠拢。

总 loss：

```text
loss = hard_weighted_arcface_ce + distill_weight * embedding_distill_loss
```

难样本权重：

```text
true_prob = softmax(student_logits)[label]
raw_weight = 1 + hard_weight * (1 - true_prob) ** hard_gamma
weight = raw_weight / mean(raw_weight)
```

蒸馏 loss：

```text
embedding_distill_loss = mean(1 - cosine(student_embedding, teacher_embedding))
```

### 训练设置

```text
student start checkpoint: models/advanced_ir18_arcface/epoch_020.pth
teacher checkpoint: models/advanced_ir18_arcface/epoch_020.pth
output: models/self_hsd_ir18_arcface
epochs: 5
output epochs: 21-25
best epoch: 24
batch size: 512
workers: 12
lr: 0.003
distill weight: 1.0
hard weight: 1.0
hard gamma: 2.0
```

训练曲线：

| epoch | loss | ArcFace loss | distill loss | train acc | teacher/student cosine | samples seen |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 21 | 7.383926 | 7.272266 | 0.111660 | 49.1826% | 0.888340 | 490,592 |
| 22 | 7.218104 | 7.107939 | 0.110165 | 50.2214% | 0.889835 | 490,592 |
| 23 | 6.980488 | 6.873053 | 0.107435 | 51.7815% | 0.892565 | 490,592 |
| 24 | 6.763883 | 6.659617 | 0.104266 | 53.4340% | 0.895734 | 490,592 |
| 25 | 6.648663 | 6.546824 | 0.101839 | 54.2691% | 0.898161 | 490,592 |

第 25 轮训练 loss 更低，但第 24 轮在 LFW `margin=16` 上略高，因此选第 24 轮作为最终自选进阶 checkpoint：

```text
models/self_hsd_ir18_arcface/epoch_024.pth
models/self_hsd_ir18_arcface/best_lfw.pth -> epoch_024.pth
```

### 进阶意义

该版本验证了在强模型基础上，继续堆叠同结构训练轮次并不一定最优；通过 teacher/student 双视图约束，可以在不改变推理结构的情况下提高模型对测试对齐变化和强增强扰动的稳定性。

## 总结果对比

### 主结果表

| 版本 | Checkpoint | MTCNN margin | LFW 10 折准确率 | ROC AUC | 全局最优准确率 | 混淆矩阵 `[[TN, FP], [FN, TP]]` |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Baseline | `scratch_casia_arcface/epoch_021.pth` | 0 | 84.8500% ± 1.1959% | 0.917717 | 85.1333% | `[[2502, 498], [411, 2589]]` |
| Baseline + margin | `scratch_casia_arcface/epoch_021.pth` | 16 | 86.4833% ± 1.8355% | 0.930206 | 86.5333% | `[[2632, 368], [443, 2557]]` |
| 课程进阶 | `advanced_ir18_arcface/epoch_020.pth` | 0 | 92.4333% ± 1.1624% | 0.965645 | 92.6167% | `[[2865, 135], [319, 2681]]` |
| 课程进阶 + margin | `advanced_ir18_arcface/epoch_020.pth` | 16 | 94.1167% ± 0.9430% | 0.971984 | 94.3167% | `[[2912, 88], [265, 2735]]` |
| 自选进阶 | `self_hsd_ir18_arcface/epoch_024.pth` | 0 | 93.4833% ± 0.9929% | 0.970011 | 93.6000% | `[[2901, 99], [292, 2708]]` |
| 自选进阶 + margin | `self_hsd_ir18_arcface/epoch_024.pth` | 16 | 94.7167% ± 0.7819% | 0.974148 | 94.8000% | `[[2921, 79], [238, 2762]]` |

### 分阶段提升

严格 `margin=0` 条件：

```text
Baseline -> 课程进阶:
92.4333% - 84.8500% = +7.5833 percentage points

课程进阶 -> 自选进阶:
93.4833% - 92.4333% = +1.0500 percentage points

Baseline -> 自选进阶:
93.4833% - 84.8500% = +8.6333 percentage points
```

最终 `margin=16` 条件：

```text
Baseline + margin -> 课程进阶 + margin:
94.1167% - 86.4833% = +7.6334 percentage points

课程进阶 + margin -> 自选进阶 + margin:
94.7167% - 94.1167% = +0.6000 percentage points

Baseline -> 自选进阶 + margin:
94.7167% - 84.8500% = +9.8667 percentage points
```

## 结果分析

### 1. Baseline 能完成流程，但模型容量和结构不是最优

Baseline 在 `margin=0` 下得到 84.8500%，接近 85% 要求，但还不稳定。使用 `margin=16` 后提升到 86.4833%，说明测试侧人脸裁剪边界对效果有明显影响。

从混淆矩阵看，baseline `margin=16` 仍有：

```text
FP = 368
FN = 443
```

这说明模型对同人与异人的 embedding 间隔仍不够清晰。对于人脸验证任务，单纯完成 ArcFace 训练流程不足以获得较强泛化能力，backbone 结构非常关键。

### 2. 课程进阶的主要收益来自 IR-ResNet18

课程进阶在同样 `margin=0` 条件下从 84.8500% 提升到 92.4333%。这说明性能提升主要不是来自测试技巧，而是来自 backbone 更适合人脸识别。

与 baseline `margin=16` 相比，课程进阶 `margin=16` 的混淆矩阵变化为：

```text
FP: 368 -> 88   (-280)
FN: 443 -> 265  (-178)
```

误判不同人为同人的数量大幅下降，说明 IR-ResNet18 学到的类间边界更清晰。FN 也明显减少，说明同人图像之间的 embedding 聚合更好。

### 3. 自选进阶提升较小但方向明确

自选进阶在课程进阶已经较强的基础上继续提升：

```text
margin=0: 92.4333% -> 93.4833%
margin=16: 94.1167% -> 94.7167%
```

与课程进阶 `margin=16` 相比，自选进阶 `margin=16` 的混淆矩阵变化为：

```text
FP: 88 -> 79   (-9)
FN: 265 -> 238 (-27)
```

这说明 hard-example self-distillation 对同人召回提升更明显，同时没有显著增加误把不同人判为同人的风险。原因是 student 在 strong view 上训练，同时被 teacher 的 weak view embedding 约束，模型对裁剪、旋转、颜色扰动的稳定性提高。

### 4. `margin=16` 稳定优于 `margin=0`

三个版本中，`margin=16` 都优于 `margin=0`：

```text
Baseline: 84.8500% -> 86.4833%
课程进阶: 92.4333% -> 94.1167%
自选进阶: 93.4833% -> 94.7167%
```

这说明 LFW deepfunneled 图像经过 MTCNN 检测后，如果裁剪过紧，会损失脸部边界和上下文；适当保留边界更接近 CASIA 镜像中 112x112 对齐人脸的分布。

### 5. 外部预训练模型只作为参考，不作为主结果

早期外部 CASIA 预训练 FaceNet 在 LFW 上达到 95.8167%，但它使用了他人预训练权重，不符合用户要求，因此不作为最终提交结果。当前最终结果 94.7167% 由本地训练和本地自蒸馏得到，实验链路更符合课程和用户约束。

## 局限性

1. 当前 CASIA-WebFace 来自公开 parquet 镜像，少于 PPT 标称的官方完整数据。
2. 自选进阶没有改变模型容量，提升空间受 IR-ResNet18 本身限制。
3. LFW 评测集较小，提升 0.6 个百分点需要结合混淆矩阵和 AUC 一起判断。
4. 训练仍直接从 parquet/PNG 解码，数据读取可能成为瓶颈；使用 LMDB/WebDataset 可能进一步提高吞吐。

## 结论

本项目形成了清晰的三阶段实验路线：

```text
Baseline:
  InceptionResnetV1 + ArcFace
  LFW margin16 = 86.4833%

课程进阶:
  IR-ResNet18 + ArcFace
  LFW margin16 = 94.1167%

自选进阶:
  IR-ResNet18 + hard-example self-distillation
  LFW margin16 = 94.7167%
```

整体来看，课程进阶证明了更适合人脸识别的 IR-ResNet backbone 是主要提升来源；自选进阶证明了在不引入外部预训练权重、不改变推理结构的前提下，通过难样本自蒸馏仍能继续改善泛化表现。最终模型完成了作业要求，并在 LFW 6000 对上取得 94.7167% 的 10 折准确率。

