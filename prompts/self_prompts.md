# Self Prompts

1. Read `design/大作业.pptx` and `design/lfw_test_pair.txt`; extract every hard requirement before writing code.
2. Build a GPU-enabled, reproducible environment named `data_mining`; verify CUDA with an actual tensor operation on the installed GPU.
3. Download or prepare LFW under `/home/data1/data_mining/data`; if the official UMass host is unavailable, use a documented mirror and keep the original 6000-pair protocol.
4. Download a CASIA-WebFace source under `/home/data1/data_mining/data`; if the official dataset requires application, use a documented public mirror and record its exact row count.
5. Do not use other people's face-recognition pretrained weights as the final result. Train a local checkpoint from `pretrained=None`; external pretrained models may only be used as baselines.
6. Implement PPT preprocessing requirements: dataloader, random crop, rotation, flip, standardization, and MTCNN face alignment for evaluation.
7. Train with GPU, mixed precision, and measured throughput. Prefer the fastest measured configuration over simply maximizing VRAM allocation.
8. Evaluate all 6000 LFW pairs with 10-fold threshold selection; save metrics, ROC, confusion matrix, scores, embeddings, and a `.pth` model artifact.
9. Write `README.md` and `STATUS.md` with exact commands, environment details, data source, model source, training coverage, results, limitations, and next steps.
10. After freezing the baseline, complete the course-required advanced experiment with a scratch IR-ResNet18 backbone, compare it against the baseline under the same LFW protocol, and document the architecture, commands, metrics, and artifacts in `advanced.md`.
11. After the course advanced result is frozen, run a self-designed advanced experiment using only locally trained weights; document the method, commands, training curves, LFW comparison, and artifacts in `self_advanced.md`.
