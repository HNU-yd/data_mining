# Status

## 2026-05-19

- 工作目录限制：所有项目文件、数据、模型缓存和结果均放在 `/home/data1/data_mining` 下。
- 已读取 `design/大作业.pptx` 和 `design/lfw_test_pair.txt`。
- PPT 关键要求：
  - 训练集：CASIA-WebFace，10,575 个身份，约 49.4 万张图像。
  - 测试集：LFW 6000 对。
  - 数据预处理：统一处理、随机裁剪、旋转、翻转、构建 dataloader；进阶可做人脸检测对齐。
  - 模型：CNN/ResNet/ArcFace/FaceNet 等。
  - 指标：LFW 准确率，进阶 ROC 和混淆矩阵。
- 已检查 `design/lfw_test_pair.txt`：
  - 共 6000 行。
  - 3000 个同人对，3000 个异人对。
  - 文件排列为前 3000 正样本、后 3000 负样本，因此评测脚本按每折 300 正 + 300 负重组 10 折。

## 环境记录

- 已创建 conda 环境 `data_mining`。
- 最初安装过 CPU/旧 CUDA 版本 PyTorch，按用户要求改为 GPU 版。
- `torch 2.2.2+cu121` 不支持 RTX PRO 6000 Blackwell `sm_120`，已升级为：
  - `torch==2.11.0+cu128`
  - `torchvision==0.26.0+cu128`
- CUDA smoke test 通过，GPU 矩阵乘法可用。
- Baseline 训练期间 GPU 利用率常见 90% 以上，功耗约 430W-460W；显存占用约 4.7GB。显存未吃满的原因是 112x112 小图、InceptionResnetV1 模型较小，以及 parquet/PNG 解码成为瓶颈。实测 batch 768/1024/2048 比 batch 512 更慢，因此 baseline 使用 batch 512。
- 课程进阶 IR-ResNet18 训练期间 GPU 利用率可到约 99%，显存约 20GB，功耗约 474W，同样使用 batch 512 和 12 workers。
- 自选进阶 hard-example self-distillation 训练期间 GPU 利用率约 94%-99%，显存约 20.5GB，功耗约 484W-502W；teacher/student 双前向比普通 IR-ResNet18 训练更充分利用 GPU。

## 数据记录

- LFW：
  - 官方 UMass 地址当前 DNS 解析失败。
  - Figshare 镜像返回 403。
  - 已使用 Hugging Face `DerrickUnleashed/LFW` 的 `lfw-deepfunneled.zip`。
  - 解压后有效图片数为 13,233。
- CASIA-WebFace：
  - 官方数据需要申请。
  - 已下载 Hugging Face `SaffalPoosh/casia_web_face` 的 20 个 parquet 分片。
  - 本地可下载镜像总数为 490,592 张，10,572 个身份标签。
  - PPT 官方数字为 494,414 张、10,575 个身份；当前镜像少 3,822 张和 3 个身份标签，已在 README 和报告中说明。

## 训练记录

- 已实现 `src/face_backbones.py` 和 `src/train_casia_parquet_arcface.py`：
  - 支持 `InceptionResnetV1(pretrained=None, classify=False)` baseline 从头训练。
  - 支持课程进阶 `IR-ResNet18` / `IR-ResNet34` 从头训练。
  - ArcFace 分类头。
  - 数据增强：随机裁剪、翻转、旋转、颜色扰动、标准化。
  - 分片/row group/样本打乱。
  - 混合精度和 CUDA 训练。
- 旧版训练循环每轮计数约 490,173 张，原因是多 worker iterable dataloader 按全局 step 提前截断。
- 已修复训练脚本：
  - 每轮 seed 随 epoch 改变。
  - 根据 worker 分片尾批计算 960 step。
  - 不再提前 `break`。
- 第 21 轮训练日志：
  - `samples_seen: 490592`
  - 覆盖当前可下载 CASIA 镜像全部样本。
- 第 22-26 轮也均记录 `samples_seen: 490592`，用于 margin 调参探索。
- 已完成课程要求进阶训练：
  - checkpoint：`models/advanced_ir18_arcface/epoch_020.pth`
  - backbone：`ir_resnet18`
  - epochs：20
  - 第 20 轮 `samples_seen: 490592`
  - 第 20 轮 `train_loss: 4.654311`
  - 第 20 轮 `train_accuracy: 0.568587`
- 已完成自选进阶 hard-example self-distillation：
  - script：`src/train_casia_hsd_arcface.py`
  - teacher：`models/advanced_ir18_arcface/epoch_020.pth`
  - student start：`models/advanced_ir18_arcface/epoch_020.pth`
  - best checkpoint：`models/self_hsd_ir18_arcface/epoch_024.pth`
  - epochs：21-25
  - 第 24 轮 `samples_seen: 490592`
  - 第 24 轮 `train_loss: 6.763883`
  - 第 24 轮 `train_accuracy: 0.534340`
  - 第 24 轮 `teacher_student_cosine: 0.895734`

## 最终实验结果

最终自选进阶采用本地训练 checkpoint：

- `models/self_hsd_ir18_arcface/epoch_024.pth`
- `models/self_hsd_ir18_arcface/best_lfw.pth -> epoch_024.pth`

最终评测命令：

```bash
TORCH_HOME=/home/data1/data_mining/models/torch \
/home/yudi/miniconda3/envs/data_mining/bin/python src/evaluate_lfw.py \
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

- LFW 10 折准确率：94.7167% ± 0.7819%。
- ROC AUC：0.974148。
- 全局最优准确率：94.8000%，阈值 0.270818。
- 10 折混淆矩阵 `[[TN, FP], [FN, TP]]`：`[[2921, 79], [238, 2762]]`。
- MTCNN 检测成功率：99.9870%。

## 对照记录

- 外部 CASIA 预训练 FaceNet + MTCNN：95.8167%，只作为早期基线，不作为最终结果。
- scratch epoch 10 + MTCNN margin 0：83.4167%。
- scratch epoch 21 + MTCNN margin 0：84.8500%，已整理为 `baseline.md`。
- scratch epoch 21 + MTCNN margin 16：86.4833%，作为 baseline 上的人脸对齐边距改进。
- 课程进阶 IR-ResNet18 epoch 20 + MTCNN margin 0：92.4333%。
- 课程进阶 IR-ResNet18 epoch 20 + MTCNN margin 16：94.1167%，已整理为 `advanced.md`。
- 自选进阶 HSD IR-ResNet18 epoch 24 + MTCNN margin 0：93.4833%。
- 自选进阶 HSD IR-ResNet18 epoch 24 + MTCNN margin 16：94.7167%，已整理为 `self_advanced.md`。
- scratch epoch 21 + resize 112：65.2333%。
- scratch epoch 22 + MTCNN margin 0：84.7000%。
- scratch epoch 26 + MTCNN margin 0 + flip TTA：84.6833%。

## 交付文件

- 代码：
  - `src/download_lfw.py`
  - `src/download_casia_webface.py`
  - `src/face_backbones.py`
  - `src/train_casia_parquet_arcface.py`
  - `src/train_casia_hsd_arcface.py`
  - `src/evaluate_lfw.py`
  - `src/train_casia_classifier.py`
- 环境：
  - `setup_env.sh`
  - `requirements.txt`
  - `environment.yml`
- 最佳结果：
  - `results/self_hsd_ir18_lfw_epoch24_margin16/metrics.json`
  - `results/self_hsd_ir18_lfw_epoch24_margin16/fold_metrics.csv`
  - `results/self_hsd_ir18_lfw_epoch24_margin16/pair_scores.csv`
  - `results/self_hsd_ir18_lfw_epoch24_margin16/roc_curve.png`
  - `results/self_hsd_ir18_lfw_epoch24_margin16/confusion_matrix.png`
  - `results/self_hsd_ir18_lfw_epoch24_margin16/score_histogram.png`
- 文档：
  - `README.md`
  - `STATUS.md`
  - `baseline.md`
  - `advanced.md`
  - `self_advanced.md`
  - `reports/final_experiment_report.md`
  - `reports/project_report.md`
  - `reports/presentation_outline.md`
  - `prompts/self_prompts.md`
