# 汇报提纲

## 1. 作业目标

- 完成人脸识别流程：数据预处理、模型训练、LFW 验证、指标分析。
- 使用 CASIA-WebFace 训练权重，LFW 6000 对测试。
- 输出 ROC 曲线、混淆矩阵、baseline 文档、课程进阶文档、模型文件和项目报告。

## 2. 数据集

- CASIA-WebFace：PPT 标称 10,575 人、494,414 张。
- 当前可下载镜像：490,592 张、10,572 类，已全部训练过。
- LFW：13,233 张图像，作业给定 6000 对，3000 正样本、3000 负样本。

## 3. 预处理

- 训练：随机裁剪、水平翻转、旋转、颜色扰动、标准化。
- 数据加载：parquet 分片、row group、样本级打乱，GPU dataloader。
- 测试：MTCNN 检测对齐到 112x112，baseline 和课程进阶都评测 margin 0 / margin 16。

## 4. 模型与训练

- Baseline backbone：`InceptionResnetV1(pretrained=None)`。
- 课程进阶 backbone：`IR-ResNet18(pretrained=None)`。
- Head：ArcFace，10,572 类。
- Optimizer：SGD + cosine scheduler。
- Mixed precision：CUDA AMP。
- Baseline checkpoint：第 21 轮 `models/scratch_casia_arcface/epoch_021.pth`。
- 课程进阶 checkpoint：第 20 轮 `models/advanced_ir18_arcface/epoch_020.pth`。
- 两个最终 checkpoint 日志均记录 `samples_seen=490592`。

## 5. 评测协议

- 对 7,701 张唯一图片提取 512 维 embedding。
- cosine similarity 计算图像对分数。
- 10 折验证：每折 300 正 + 300 负，9 折选阈值，1 折测试。

## 6. 最终结果

- LFW 10 折准确率：94.1167% ± 0.9430%。
- ROC AUC：0.971984。
- 全局最优阈值：0.278341。
- 混淆矩阵：`[[2912, 88], [265, 2735]]`。
- MTCNN 检测成功率：99.9870%。

## 7. 对照分析

- scratch epoch 21 + MTCNN margin 0：84.8500%。
- scratch epoch 21 + MTCNN margin 16：86.4833%。
- IR-ResNet18 epoch 20 + MTCNN margin 0：92.4333%。
- IR-ResNet18 epoch 20 + MTCNN margin 16：94.1167%。
- scratch epoch 21 + resize：65.2333%。
- 外部预训练 FaceNet：95.8167%，只作为早期基线，不作为最终提交。
- 结论：本地训练权重能达标，IR-ResNet18 模型构建进阶带来主要提升，测试侧对齐边距进一步提高最终效果。

## 8. 总结

- 已按要求训练本地权重，没有把他人预训练权重作为主结果。
- 已完成数据预处理、baseline、课程进阶、全量镜像训练、LFW 10 折评测、ROC 和混淆矩阵。
- 当前公开镜像少于 PPT 官方数量，已在报告中说明。
