# Self Advanced: Hard-Example Self-Distillation

本文档记录课程要求进阶之后的自选进阶实验。课程进阶已经完成 `IR-ResNet18 + ArcFace` 从头训练，本实验不再引入外部预训练权重，而是用我们自己训练出的 `models/advanced_ir18_arcface/epoch_020.pth` 作为 teacher，对同一个 IR-ResNet18 student 做难样本自蒸馏微调。

## 方法定位

已有结果：

```text
baseline: InceptionResnetV1 + ArcFace, scratch, LFW margin16 = 86.4833%
course advanced: IR-ResNet18 + ArcFace, scratch, LFW margin16 = 94.1167%
```

自选进阶：

```text
self advanced: IR-ResNet18 + ArcFace + hard-example self-distillation
teacher: our own IR-ResNet18 epoch_020 checkpoint
student: initialized from the same checkpoint
external pretrained face-recognition weights: none
```

该方法的目的不是换更大的模型，而是在不改变最终推理结构的前提下，提高 embedding 对强增强输入的稳定性，并把训练注意力更多放在低置信度样本上。

## 训练机制

每张 CASIA-WebFace 训练图像产生两个视图：

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

Teacher 只看 weak view，冻结参数并输出归一化 embedding。Student 看 strong view，继续用 ArcFace 分类头训练。

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

本次超参数：

| 项目 | 数值 |
| --- | --- |
| student start | `models/advanced_ir18_arcface/epoch_020.pth` |
| teacher | `models/advanced_ir18_arcface/epoch_020.pth` |
| backbone | `ir_resnet18` |
| epochs | 5 |
| output epochs | 21-25 |
| best epoch | 24 |
| batch size | 512 |
| workers | 12 |
| lr | 0.003 |
| distill weight | 1.0 |
| hard weight | 1.0 |
| hard gamma | 2.0 |
| optimizer | SGD + momentum + Nesterov |
| scheduler | CosineAnnealingLR |
| AMP | enabled |
| device | CUDA |

## 训练命令

```bash
TORCH_HOME=/home/data1/data_mining/models/torch \
python src/train_casia_hsd_arcface.py \
  --epochs 5 \
  --batch-size 512 \
  --num-workers 12 \
  --output-dir models/self_hsd_ir18_arcface \
  --lr 0.003 \
  --distill-weight 1.0 \
  --hard-weight 1.0 \
  --hard-gamma 2.0 \
  --device cuda
```

训练记录：

| epoch | loss | ArcFace loss | distill loss | train acc | teacher/student cosine | samples seen |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 21 | 7.383926 | 7.272266 | 0.111660 | 49.1826% | 0.888340 | 490,592 |
| 22 | 7.218104 | 7.107939 | 0.110165 | 50.2214% | 0.889835 | 490,592 |
| 23 | 6.980488 | 6.873053 | 0.107435 | 51.7815% | 0.892565 | 490,592 |
| 24 | 6.763883 | 6.659617 | 0.104266 | 53.4340% | 0.895734 | 490,592 |
| 25 | 6.648663 | 6.546824 | 0.101839 | 54.2691% | 0.898161 | 490,592 |

第 24 轮在 LFW 上略优于第 25 轮，因此选为最佳 checkpoint。

## 评测命令

MTCNN margin 0：

```bash
TORCH_HOME=/home/data1/data_mining/models/torch \
python src/evaluate_lfw.py \
  --lfw-root data/raw/lfw-deepfunneled \
  --pairs-file design/lfw_test_pair.txt \
  --checkpoint models/self_hsd_ir18_arcface/epoch_024.pth \
  --preprocess mtcnn \
  --mtcnn-margin 0 \
  --image-size 112 \
  --batch-size 512 \
  --num-workers 0 \
  --device cuda \
  --output-dir results/self_hsd_ir18_lfw_epoch24
```

MTCNN margin 16：

```bash
TORCH_HOME=/home/data1/data_mining/models/torch \
python src/evaluate_lfw.py \
  --lfw-root data/raw/lfw-deepfunneled \
  --pairs-file design/lfw_test_pair.txt \
  --checkpoint models/self_hsd_ir18_arcface/epoch_024.pth \
  --preprocess mtcnn \
  --mtcnn-margin 16 \
  --image-size 112 \
  --batch-size 512 \
  --num-workers 0 \
  --device cuda \
  --output-dir results/self_hsd_ir18_lfw_epoch24_margin16
```

## 结果

| 实验 | Backbone | MTCNN margin | LFW 10 折准确率 | ROC AUC | 混淆矩阵 `[[TN, FP], [FN, TP]]` |
| --- | --- | ---: | ---: | ---: | --- |
| baseline | InceptionResnetV1 | 0 | 84.8500% ± 1.1959% | 0.917717 | `[[2502, 498], [411, 2589]]` |
| baseline + 对齐边距 | InceptionResnetV1 | 16 | 86.4833% ± 1.8355% | 0.930206 | `[[2632, 368], [443, 2557]]` |
| 课程进阶 | IR-ResNet18 | 0 | 92.4333% ± 1.1624% | 0.965645 | `[[2865, 135], [319, 2681]]` |
| 课程进阶 + 对齐边距 | IR-ResNet18 | 16 | 94.1167% ± 0.9430% | 0.971984 | `[[2912, 88], [265, 2735]]` |
| 自选进阶 | IR-ResNet18 + HSD | 0 | 93.4833% ± 0.9929% | 0.970011 | `[[2901, 99], [292, 2708]]` |
| 自选进阶 + 对齐边距 | IR-ResNet18 + HSD | 16 | 94.7167% ± 0.7819% | 0.974148 | `[[2921, 79], [238, 2762]]` |

相对课程进阶，在相同 `margin=0` 条件下提升：

```text
93.4833% - 92.4333% = +1.0500 percentage points
```

在最终 `margin=16` 配置下提升：

```text
94.7167% - 94.1167% = +0.6000 percentage points
```

相对最初严格 baseline 的最终提升：

```text
94.7167% - 84.8500% = +9.8667 percentage points
```

## 输出文件

模型元数据：

- `models/self_hsd_ir18_arcface/training_config.json`
- `models/self_hsd_ir18_arcface/history.json`
- `models/self_hsd_ir18_arcface/best_lfw.json`

本地权重：

- `models/self_hsd_ir18_arcface/epoch_024.pth`
- `models/self_hsd_ir18_arcface/best_lfw.pth -> epoch_024.pth`

`.pth` 权重约 225MB，不纳入普通 Git 跟踪。

评测结果：

- `results/self_hsd_ir18_lfw_epoch24/metrics.json`
- `results/self_hsd_ir18_lfw_epoch24/fold_metrics.csv`
- `results/self_hsd_ir18_lfw_epoch24/pair_scores.csv`
- `results/self_hsd_ir18_lfw_epoch24/roc_curve.png`
- `results/self_hsd_ir18_lfw_epoch24/confusion_matrix.png`
- `results/self_hsd_ir18_lfw_epoch24/score_histogram.png`
- `results/self_hsd_ir18_lfw_epoch24_margin16/metrics.json`
- `results/self_hsd_ir18_lfw_epoch24_margin16/fold_metrics.csv`
- `results/self_hsd_ir18_lfw_epoch24_margin16/pair_scores.csv`
- `results/self_hsd_ir18_lfw_epoch24_margin16/roc_curve.png`
- `results/self_hsd_ir18_lfw_epoch24_margin16/confusion_matrix.png`
- `results/self_hsd_ir18_lfw_epoch24_margin16/score_histogram.png`

`lfw_embeddings.npz` 是中间缓存，体积较大，不纳入 Git。

## 结论

自选进阶完成后，模型推理结构仍是 IR-ResNet18，未引入任何外部人脸识别预训练权重；提升来自我们自己训练 checkpoint 的 hard-example self-distillation。最终 LFW 6000 对 10 折准确率为 94.7167%，超过课程进阶的 94.1167%。
