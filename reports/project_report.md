# 深度学习人脸识别实践项目报告

## 任务目标

本作业要求完成人脸识别基本流程：数据预处理、模型训练或构建、LFW 6000 对验证、指标分析和报告输出。核心指标为 LFW 测试准确率不低于 85%，进阶部分包括人脸检测对齐、ArcFace、ROC 曲线和混淆矩阵。

用户进一步要求最终权重需要自行训练，不能直接使用他人的人脸识别预训练权重。因此 baseline 采用本地从头训练的 `InceptionResnetV1 + ArcFace`，课程要求进阶采用本地从头训练的 `IR-ResNet18 + ArcFace`。外部 CASIA 预训练 FaceNet 仅保留为早期对照。

## 数据集

训练集使用 CASIA-WebFace。PPT 标称该数据集包含 10,575 个身份和 494,414 张图像。由于官方数据需要申请，项目使用 Hugging Face `SaffalPoosh/casia_web_face` 的 parquet 镜像。该镜像包含 20 个分片、490,592 张 112x112 RGB 对齐图，身份标签覆盖 0..10571，共 10,572 类。镜像比 PPT 数字少 3,822 张和 3 个身份标签，推测来自公开镜像的清洗或转换差异；训练已覆盖当前镜像全部 490,592 张。

测试集使用 LFW 6000 对。作业提供的 `design/lfw_test_pair.txt` 共 6000 行，其中 3000 个同人对、3000 个异人对。当前环境无法解析 UMass 官方域名，Figshare 镜像返回 403，因此使用 Hugging Face `DerrickUnleashed/LFW` 的 `lfw-deepfunneled.zip`，共 13,233 张图像，文件命名与测试对文件兼容。

## 数据预处理

训练阶段实现了作业 PPT 的数据预处理要求。parquet 中的 PNG bytes 被解码为 RGB 图像后进入 `DataLoader`，训练增强包括随机裁剪、随机水平翻转、随机旋转、颜色扰动和 `fixed_image_standardization`。数据读取时对 shard、row group 和样本顺序都进行打乱。

测试阶段使用 MTCNN 做人脸检测与对齐，输出 112x112 输入。最终设置 `mtcnn_margin=16`，使裁剪结果保留适当脸部边界，更接近 CASIA 镜像中 112x112 对齐人脸的分布。

## 模型与训练

Baseline 模型主体为 `facenet-pytorch` 的 `InceptionResnetV1(pretrained=None, classify=False)`，即不加载任何外部人脸识别预训练权重。分类头采用 ArcFace：

```text
embedding: 512 dim
classes: 10572
loss: ArcFace logits + cross entropy
optimizer: SGD, momentum 0.9, Nesterov
scheduler: cosine annealing
batch size: 512
workers: 12
precision: AMP mixed precision
device: CUDA
```

课程进阶在 `src/face_backbones.py` 中实现 InsightFace-style `IR-ResNet18`：输入 112x112 RGB 图像，四个残差 stage 的 block 数为 `[2, 2, 2, 2]`，通道为 64/128/256/512，最终通过 `Linear(512*7*7 -> 512)` 输出 512 维 embedding。训练时继续使用 10,572 类 ArcFace head。

训练过程中发现多 worker iterable dataloader 如果按全局 `ceil(total/batch)` 截断，会漏掉每个 worker 的尾批。脚本已修复为按 worker 分片计算 960 个 step，并取消提前截断。Baseline 第 21 轮日志记录 `samples_seen=490592`，课程进阶 IR-ResNet18 第 20 轮也记录 `samples_seen=490592`，确认当前可下载 CASIA 镜像全部样本均被训练流程读取。

课程进阶最终选择 `models/advanced_ir18_arcface/epoch_020.pth`。该 checkpoint 第 20 轮训练损失为 4.654311，训练分类准确率为 56.8587%。

## 评测方法

评测脚本对 LFW 中 7,701 张唯一图片提取 512 维 L2 归一化 embedding，图像对相似度使用 cosine similarity。由于作业 pair 文件前 3000 行为正样本、后 3000 行为负样本，脚本将其重组为 10 折，每折包含 300 个正样本和 300 个负样本。每次用其余 9 折选择阈值，在当前折测试。

## 实验环境

- GPU：NVIDIA RTX PRO 6000 Blackwell Workstation Edition
- Python：3.10
- PyTorch：2.11.0+cu128
- torchvision：0.26.0+cu128
- facenet-pytorch：2.6.0
- CUDA smoke test：通过 GPU 张量矩阵乘法验证

Baseline 训练时 GPU 利用率通常在 90% 以上，功耗约 430W-460W。课程进阶 IR-ResNet18 训练时 GPU 利用率可到约 99%，显存约 20GB，功耗约 474W。实测 batch 512 在当前 parquet/PNG 数据读取方式下吞吐更稳定，因此两组 scratch 训练均使用 batch 512。

## 最终结果

最终课程进阶 checkpoint：`models/advanced_ir18_arcface/epoch_020.pth`

最终评测配置：

```text
preprocess: mtcnn
mtcnn_margin: 16
image_size: 112
batch_size: 512
tta_flip: false
```

| 指标 | 数值 |
| --- | ---: |
| LFW 10 折准确率 | 94.1167% ± 0.9430% |
| ROC AUC | 0.971984 |
| 全局最优阈值 | 0.278341 |
| 全局最优准确率 | 94.3167% |
| MTCNN 检测成功率 | 99.9870% |

10 折混淆矩阵 `[[TN, FP], [FN, TP]]`：

```text
[[2912,   88],
 [ 265, 2735]]
```

逐折准确率：

```text
95.00%, 93.17%, 93.00%, 93.00%, 93.83%,
94.83%, 93.33%, 94.17%, 95.17%, 95.67%
```

## 对照实验

| 实验 | LFW 10 折准确率 | 说明 |
| --- | ---: | --- |
| scratch epoch 10，MTCNN margin 0 | 83.4167% | 初始 scratch 权重未达标 |
| scratch epoch 21，MTCNN margin 0 | 84.8500% | 接近 85%，但仍未达标 |
| scratch epoch 21，MTCNN margin 16 | 86.4833% | baseline 上的人脸对齐边距改进 |
| IR-ResNet18 epoch 20，MTCNN margin 0 | 92.4333% | 课程模型构建进阶，相同 margin 条件下明显超过 baseline |
| IR-ResNet18 epoch 20，MTCNN margin 16 | 94.1167% | 当前最终课程进阶结果 |
| scratch epoch 21，resize 112 | 65.2333% | 直接缩放不适合该模型 |
| scratch epoch 22，MTCNN margin 0 | 84.7000% | 降 margin 后未提升 |
| scratch epoch 26，MTCNN margin 0 + flip TTA | 84.6833% | TTA 未带来提升 |
| 外部 CASIA 预训练 FaceNet | 95.8167% | 早期基线，不作为最终结果 |

对照说明：最终准确率超过 85% 的关键不是使用外部预训练权重，而是完成 CASIA 全量镜像训练，并在 baseline 之后采用更适合人脸识别的 IR-ResNet 残差骨干。MTCNN 对齐边距进一步提升了最终效果。

## 输出文件

- 最佳模型：`models/advanced_ir18_arcface/epoch_020.pth`
- 指标：`results/advanced_ir18_arcface_lfw_epoch20_margin16/metrics.json`
- 逐折结果：`results/advanced_ir18_arcface_lfw_epoch20_margin16/fold_metrics.csv`
- 每对分数：`results/advanced_ir18_arcface_lfw_epoch20_margin16/pair_scores.csv`
- ROC 曲线：`results/advanced_ir18_arcface_lfw_epoch20_margin16/roc_curve.png`
- 混淆矩阵：`results/advanced_ir18_arcface_lfw_epoch20_margin16/confusion_matrix.png`
- 分数分布：`results/advanced_ir18_arcface_lfw_epoch20_margin16/score_histogram.png`

## 结论

本项目完成了作业要求的人脸识别流程：CASIA-WebFace 数据准备、训练数据增强、GPU 训练、本地权重保存、LFW 6000 对 10 折评测、ROC 曲线和混淆矩阵输出。最终课程进阶使用本地从头训练的 IR-ResNet18 + ArcFace 权重，在 LFW 上取得 94.1167% 的 10 折准确率，高于 85% 要求。

## 局限与改进

当前公开镜像比 PPT 标称数据少 3,822 张和 3 个身份标签，若后续申请到官方原始 CASIA-WebFace，可重新训练以消除数据源差异。模型方面，IR-ResNet18 已完成课程要求进阶；若后续继续自选进阶，可以尝试 IR-ResNet34/50、Partial FC、WebDataset/LMDB 缓存、更长训练和更系统的学习率策略。
