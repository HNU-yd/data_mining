#!/usr/bin/env python3
"""Train an InceptionResnetV1 classifier on a CASIA-WebFace ImageFolder tree.

The executed experiment uses the released FaceNet weights trained on CASIA-WebFace.
This script is included so the training workflow is reproducible when the CASIA
images are available locally.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = self.ce(logits, targets)
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


def make_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(160, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.PILToTensor(),
            transforms.Lambda(lambda x: fixed_image_standardization(x.float())),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize((160, 160), antialias=True),
            transforms.PILToTensor(),
            transforms.Lambda(lambda x: fixed_image_standardization(x.float())),
        ]
    )
    return train_tf, val_tf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="CASIA-WebFace ImageFolder root")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "models" / "casia_training")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--loss", choices=["ce", "focal"], default="ce")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_tf, val_tf = make_transforms()

    full = datasets.ImageFolder(args.data_dir, transform=train_tf)
    num_classes = len(full.classes)
    if num_classes < 2:
        raise ValueError(f"Need at least two identities under {args.data_dir}")
    val_size = max(1, int(len(full) * args.val_ratio))
    train_size = len(full) - val_size
    train_set, val_set = random_split(full, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    val_set.dataset.transform = val_tf

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    device = torch.device(args.device)
    model = InceptionResnetV1(classify=True, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss() if args.loss == "ce" else FocalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    history = []
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} train"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item()) * images.size(0)

        model.eval()
        correct = 0
        total = 0
        with torch.inference_mode():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} val"):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(images)
                pred = logits.argmax(dim=1)
                correct += int((pred == labels).sum().item())
                total += int(labels.numel())
        train_loss = running_loss / train_size
        val_acc = correct / max(total, 1)
        record = {"epoch": epoch, "train_loss": train_loss, "val_accuracy": val_acc}
        history.append(record)
        print(record)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "classes": full.classes,
            "history": history,
        }
        torch.save(checkpoint, args.output_dir / f"epoch_{epoch:03d}.pth")
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(checkpoint, args.output_dir / "best.pth")

    with (args.output_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
