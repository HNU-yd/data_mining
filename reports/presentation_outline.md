# 汇报提纲

## 1. 作业目标

- 完成人脸识别流程：数据预处理、模型训练、LFW 验证、指标分析。
- 使用 CASIA-WebFace 训练权重，LFW 6000 对测试。
- 输出 ROC 曲线、混淆矩阵、模型文件和项目报告。

## 2. 数据集

- CASIA-WebFace：PPT 标称 10,575 人、494,414 张。
- 当前可下载镜像：490,592 张、10,572 类，已全部训练过。
- LFW：13,233 张图像，作业给定 6000 对，3000 正样本、3000 负样本。

## 3. 预处理

- 训练：随机裁剪、水平翻转、旋转、颜色扰动、标准化。
- 数据加载：parquet 分片、row group、样本级打乱，GPU dataloader。
- 测试：MTCNN 检测对齐到 112x112，最终 margin 16。

## 4. 模型与训练

- Backbone：`InceptionResnetV1(pretrained=None)`。
- Head：ArcFace，10,572 类。
- Optimizer：SGD + cosine scheduler。
- Mixed precision：CUDA AMP。
- 最佳 checkpoint：第 21 轮 `epoch_021.pth`。
- 第 21 轮日志：`samples_seen=490592`。

## 5. 评测协议

- 对 7,701 张唯一图片提取 512 维 embedding。
- cosine similarity 计算图像对分数。
- 10 折验证：每折 300 正 + 300 负，9 折选阈值，1 折测试。

## 6. 最终结果

- LFW 10 折准确率：86.4833% ± 1.8355%。
- ROC AUC：0.930206。
- 全局最优阈值：0.984878。
- 混淆矩阵：`[[2632, 368], [443, 2557]]`。
- MTCNN 检测成功率：99.9870%。

## 7. 对照分析

- scratch epoch 21 + MTCNN margin 0：84.8500%。
- scratch epoch 21 + resize：65.2333%。
- 外部预训练 FaceNet：95.8167%，只作为早期基线，不作为最终提交。
- 结论：本地训练权重能达标，测试侧对齐边距对最终效果很关键。

## 8. 总结

- 已按要求训练本地权重，没有把他人预训练权重作为主结果。
- 已完成数据预处理、全量镜像训练、LFW 10 折评测、ROC 和混淆矩阵。
- 当前公开镜像少于 PPT 官方数量，已在报告中说明。
