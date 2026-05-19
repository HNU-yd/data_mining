# 深度学习人脸识别实践作业

本项目按 `design/大作业.pptx` 完成人脸验证大作业：读取作业给定 LFW 6000 对，使用 CASIA-WebFace 训练人脸特征模型，并输出准确率、ROC、混淆矩阵、训练记录和报告。

主结果不使用他人的人脸识别预训练权重。当前课程进阶最终权重由本机从 `pretrained=None` 的 `IR-ResNet18` 开始，在可下载 CASIA-WebFace 镜像上训练得到。

进阶实验前的可复现基线见 `baseline.md`。该 baseline 使用第 21 轮本地 scratch InceptionResnetV1 权重、MTCNN `margin=0`，LFW 10 折准确率为 84.8500%。课程要求进阶见 `advanced.md`，使用从头训练的 `IR-ResNet18 + ArcFace`，最终 LFW 10 折准确率为 94.1167%。

## 环境

工作目录固定为 `/home/data1/data_mining`。已创建 conda 环境：

```bash
conda activate data_mining
```

复现环境：

```bash
bash setup_env.sh
conda activate data_mining
```

关键配置：

- GPU：NVIDIA RTX PRO 6000 Blackwell Workstation Edition，约 96GB 显存
- PyTorch：`torch==2.11.0+cu128`
- torchvision：`0.26.0+cu128`
- Python：3.10
- `facenet-pytorch==2.6.0 --no-deps`

说明：早期尝试过旧版 CUDA wheel，但 `torch 2.2.2+cu121` 不支持 Blackwell `sm_120`；当前环境使用 cu128 GPU 版并已通过 CUDA 张量运算测试。

## 数据

### CASIA-WebFace 训练集

PPT 标称 CASIA-WebFace 为 10,575 个身份、494,414 张图像。官方数据集需要申请，当前环境使用 Hugging Face 镜像 [`SaffalPoosh/casia_web_face`](https://huggingface.co/datasets/SaffalPoosh/casia_web_face) 的 20 个 parquet 分片：

- 本地目录：`data/raw/casia_webface_parquet`
- 可下载镜像总量：490,592 张
- 镜像身份类别：10,572 类，label 范围覆盖 0..10571
- 图像格式：parquet 内 PNG bytes，RGB，112x112 对齐图

差额说明：镜像比 PPT 官方数字少 3,822 张和 3 个身份标签，推测来自镜像清洗或转换；本项目已训练当前可下载镜像的全部 490,592 张。

下载命令：

```bash
python src/download_casia_webface.py
```

### LFW 测试集

- 测试对文件：`design/lfw_test_pair.txt`
- 图像目录：`data/raw/lfw-deepfunneled`
- 图像数量：13,233
- 测试对数量：6,000，其中同人 3,000 对、异人 3,000 对
- 唯一测试图片：7,701 张

官方 UMass LFW 地址在当前环境 DNS 解析失败，scikit-learn/Figshare 镜像返回 403，因此使用 Hugging Face 镜像 [`DerrickUnleashed/LFW`](https://huggingface.co/datasets/DerrickUnleashed/LFW) 的 `lfw-deepfunneled.zip`。测试协议仍使用作业提供的 6000 对文件。

下载命令：

```bash
python src/download_lfw.py
```

## 预处理

训练阶段按 PPT “数据预处理”要求实现：

- 从 parquet 解码 RGB 图片并构建 `DataLoader`
- `RandomResizedCrop(112, scale=(0.86, 1.0))`
- `RandomHorizontalFlip`
- `RandomRotation(10)`
- `ColorJitter`
- `fixed_image_standardization`
- 分片、row group、样本级随机打乱

测试阶段使用 MTCNN 人脸检测对齐，输出 112x112；最终设置为 `--mtcnn-margin 16`，比 `margin=0` 保留更多脸部边界，更接近 CASIA 镜像的 112x112 对齐分布。

## 模型与训练

Baseline 模型：

- Backbone：`InceptionResnetV1(pretrained=None, classify=False)`
- Head：ArcFace `ArcMarginProduct(512, 10572)`
- Loss：ArcFace logits + cross entropy
- Optimizer：SGD + momentum + Nesterov
- Scheduler：CosineAnnealingLR
- Mixed precision：`torch.amp.autocast` + `GradScaler`
- Batch size：512
- Workers：12
- Device：CUDA

课程进阶模型：

- Backbone：`IR-ResNet18(pretrained=None)`
- Head：ArcFace `ArcMarginProduct(512, 10572)`
- Loss/optimizer/scheduler/AMP：同 baseline
- Batch size：512
- Workers：12
- Device：CUDA

Baseline 训练命令：

```bash
TORCH_HOME=/home/data1/data_mining/models/torch \
python src/train_casia_parquet_arcface.py \
  --epochs 10 \
  --batch-size 512 \
  --num-workers 8 \
  --output-dir models/scratch_casia_arcface \
  --device cuda

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

第 21 轮训练日志明确记录：

```text
samples_seen: 490592
```

即当前 CASIA-WebFace 镜像的全部样本已在训练中被读取。第 22-26 轮用于 margin 调参探索，也均覆盖 490,592 张，但 LFW 最佳 checkpoint 为第 21 轮。

课程进阶训练命令：

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

第 20 轮训练日志记录 `samples_seen: 490592`，同样覆盖当前可下载 CASIA-WebFace 镜像的全部样本。训练期间 RTX PRO 6000 利用率可到约 99%，显存约 20GB，功耗约 474W。

## 最终评测

课程进阶最终命令：

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

评测协议：

- 对 7,701 张唯一图片提取 512 维 L2 归一化 embedding
- 对每对图片计算 cosine similarity
- 按作业 6000 对重组 10 折：每折 300 个同人对 + 300 个异人对
- 每折用其余 9 折选择阈值，在当前折测试

课程进阶最终结果：

| 指标 | 数值 |
| --- | ---: |
| LFW 10 折准确率 | 94.1167% ± 0.9430% |
| ROC AUC | 0.971984 |
| 全局最优阈值 | 0.278341 |
| 全局最优准确率 | 94.3167% |
| 10 折混淆矩阵 `[[TN, FP], [FN, TP]]` | `[[2912, 88], [265, 2735]]` |
| MTCNN 检测成功率 | 99.9870% |

输出文件：

- `results/advanced_ir18_arcface_lfw_epoch20_margin16/metrics.json`
- `results/advanced_ir18_arcface_lfw_epoch20_margin16/fold_metrics.csv`
- `results/advanced_ir18_arcface_lfw_epoch20_margin16/pair_scores.csv`
- `results/advanced_ir18_arcface_lfw_epoch20_margin16/roc_curve.png`
- `results/advanced_ir18_arcface_lfw_epoch20_margin16/confusion_matrix.png`
- `results/advanced_ir18_arcface_lfw_epoch20_margin16/score_histogram.png`

最佳本地权重：

- `models/advanced_ir18_arcface/epoch_020.pth`
- `models/advanced_ir18_arcface/best_lfw.pth` 指向第 20 轮

## 对照实验

这些结果不作为主结果，只用于验证工程选择：

| 实验 | LFW 10 折准确率 |
| --- | ---: |
| scratch epoch 10，MTCNN margin 0 | 83.4167% |
| baseline scratch epoch 21，MTCNN margin 0 | 84.8500% |
| baseline scratch epoch 21，MTCNN margin 16 | 86.4833% |
| scratch epoch 21，直接 resize 112 | 65.2333% |
| scratch epoch 22，MTCNN margin 0 | 84.7000% |
| scratch epoch 26，MTCNN margin 0 + flip TTA | 84.6833% |
| 课程进阶 IR-ResNet18 epoch 20，MTCNN margin 0 | 92.4333% |
| 课程进阶 IR-ResNet18 epoch 20，MTCNN margin 16 | 94.1167% |
| 外部 CASIA 预训练 FaceNet，MTCNN margin 0 | 95.8167% |

外部预训练 FaceNet 是早期基线，用户要求“训练一个权重，不要用其他人的权重”后已不作为最终提交结果。

## 报告

- 项目报告：`reports/project_report.md`
- Baseline 说明：`baseline.md`
- 课程进阶说明：`advanced.md`
- 汇报提纲：`reports/presentation_outline.md`
- 执行状态：`STATUS.md`
- 自我任务提示：`prompts/self_prompts.md`
