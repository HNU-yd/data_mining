#!/usr/bin/env python3
"""Train a scratch face-recognition backbone on CASIA-WebFace parquet shards."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import deque
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from facenet_pytorch import fixed_image_standardization
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, IterableDataset
from torchvision import transforms
from tqdm import tqdm

try:
    from face_backbones import build_backbone, canonical_backbone_name
except ModuleNotFoundError:
    from .face_backbones import build_backbone, canonical_backbone_name

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "casia_webface_parquet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "scratch_casia_arcface"


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_accuracy: float
    samples_seen: int
    learning_rate: float


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features: int, out_features: int, scale: float = 64.0, margin: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        sine = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return logits * self.scale


class CASIAParquetIterable(IterableDataset):
    def __init__(
        self,
        shards: list[Path],
        train: bool,
        image_size: int,
        shuffle_buffer: int,
        seed: int,
        max_samples: int | None = None,
    ):
        self.shards = shards
        self.train = train
        self.image_size = image_size
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.max_samples = max_samples
        self.epoch = 0
        self.transform = self._build_transform()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _build_transform(self) -> transforms.Compose:
        if self.train:
            return transforms.Compose(
                [
                    transforms.RandomResizedCrop(self.image_size, scale=(0.86, 1.0), antialias=True),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(10),
                    transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
                    transforms.PILToTensor(),
                    transforms.Lambda(lambda x: fixed_image_standardization(x.float())),
                ]
            )
        return transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size), antialias=True),
                transforms.PILToTensor(),
                transforms.Lambda(lambda x: fixed_image_standardization(x.float())),
            ]
        )

    def _rows(self, rng: random.Random) -> Iterator[tuple[bytes, int]]:
        worker = torch.utils.data.get_worker_info()
        shards = self.shards
        if worker is not None:
            shards = shards[worker.id :: worker.num_workers]
        shards = list(shards)
        rng.shuffle(shards)
        for shard in shards:
            pf = pq.ParquetFile(shard)
            groups = list(range(pf.num_row_groups))
            rng.shuffle(groups)
            for group in groups:
                table = pf.read_row_group(group, columns=["image", "label"])
                records = table.to_pylist()
                rng.shuffle(records)
                for row in records:
                    yield row["image"]["bytes"], int(row["label"])

    def __iter__(self) -> Iterator[tuple[torch.Tensor, int]]:
        worker = torch.utils.data.get_worker_info()
        worker_id = 0 if worker is None else worker.id
        rng = random.Random(self.seed + self.epoch * 1009 + worker_id)
        max_samples = self.max_samples
        if worker is not None and max_samples is not None:
            base = max_samples // worker.num_workers
            extra = 1 if worker_id < (max_samples % worker.num_workers) else 0
            max_samples = base + extra
        emitted = 0
        buffer: deque[tuple[bytes, int]] = deque()
        for item in self._rows(rng):
            if self.shuffle_buffer <= 1:
                yield self._decode(item)
                emitted += 1
            else:
                buffer.append(item)
                if len(buffer) >= self.shuffle_buffer:
                    idx = rng.randrange(len(buffer))
                    buffer.rotate(-idx)
                    chosen = buffer.popleft()
                    buffer.rotate(idx)
                    yield self._decode(chosen)
                    emitted += 1
            if max_samples is not None and emitted >= max_samples:
                return
        while buffer:
            idx = rng.randrange(len(buffer))
            buffer.rotate(-idx)
            chosen = buffer.popleft()
            buffer.rotate(idx)
            yield self._decode(chosen)
            emitted += 1
            if max_samples is not None and emitted >= max_samples:
                return

    def _decode(self, item: tuple[bytes, int]) -> tuple[torch.Tensor, int]:
        image_bytes, label = item
        with Image.open(BytesIO(image_bytes)) as img:
            img = img.convert("RGB")
            return self.transform(img), label


def count_rows_and_classes(shards: list[Path]) -> tuple[int, int, list[int]]:
    total_rows = 0
    max_label = -1
    row_counts: list[int] = []
    for shard in shards:
        pf = pq.ParquetFile(shard)
        shard_rows = pf.metadata.num_rows
        row_counts.append(shard_rows)
        total_rows += shard_rows
        for rg in range(pf.num_row_groups):
            labels = pf.read_row_group(rg, columns=["label"]).column("label").to_numpy()
            max_label = max(max_label, int(labels.max()))
    return total_rows, max_label + 1, row_counts


def estimate_loader_batches(row_counts: list[int], batch_size: int, num_workers: int) -> int:
    if num_workers <= 0:
        return math.ceil(sum(row_counts) / batch_size)
    rows_by_worker = [0 for _ in range(num_workers)]
    for shard_idx, rows in enumerate(row_counts):
        rows_by_worker[shard_idx % num_workers] += rows
    return sum(math.ceil(rows / batch_size) for rows in rows_by_worker if rows > 0)


def estimate_max_sample_batches(max_samples: int, batch_size: int, num_workers: int) -> int:
    if num_workers <= 0:
        return math.ceil(max_samples / batch_size)
    base = max_samples // num_workers
    remainder = max_samples % num_workers
    quotas = [base + (1 if worker_id < remainder else 0) for worker_id in range(num_workers)]
    return sum(math.ceil(quota / batch_size) for quota in quotas if quota > 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--resume", type=Path, default=None, help="resume backbone/head weights from a previous checkpoint")
    parser.add_argument("--resume-optimizer", action="store_true", help="also restore optimizer and scheduler states")
    parser.add_argument(
        "--backbone",
        choices=["inception_resnet_v1", "ir_resnet18", "ir_resnet34"],
        default="inception_resnet_v1",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--arc-scale", type=float, default=64.0)
    parser.add_argument("--arc-margin", type=float, default=0.5)
    parser.add_argument("--shuffle-buffer", type=int, default=16384)
    parser.add_argument("--max-samples", type=int, default=None, help="debug only; default trains every row")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shards = sorted(args.data_dir.glob("train-*-of-00020.parquet"))
    if len(shards) != 20:
        raise FileNotFoundError(f"Expected 20 parquet shards under {args.data_dir}, found {len(shards)}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    total_rows, num_classes, row_counts = count_rows_and_classes(shards)
    samples_per_epoch = min(total_rows, args.max_samples) if args.max_samples else total_rows
    steps_per_epoch = (
        estimate_max_sample_batches(samples_per_epoch, args.batch_size, args.num_workers)
        if args.max_samples
        else estimate_loader_batches(row_counts, args.batch_size, args.num_workers)
    )

    dataset = CASIAParquetIterable(
        shards=shards,
        train=True,
        image_size=args.image_size,
        shuffle_buffer=args.shuffle_buffer,
        seed=args.seed,
        max_samples=args.max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )

    device = torch.device(args.device)
    backbone, embedding_size, backbone_name = build_backbone(args.backbone, pretrained_model=None)
    backbone = backbone.to(device)
    head = ArcMarginProduct(embedding_size, num_classes, scale=args.arc_scale, margin=args.arc_margin).to(device)
    optimizer = torch.optim.SGD(
        list(backbone.parameters()) + list(head.parameters()),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * steps_per_epoch)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_epoch = 1
    history: list[EpochMetrics] = []

    if args.resume is not None:
        payload = torch.load(args.resume, map_location="cpu")
        if int(payload.get("num_classes", num_classes)) != num_classes:
            raise ValueError(f"Checkpoint num_classes={payload.get('num_classes')} but data has {num_classes}")
        checkpoint_backbone = canonical_backbone_name(payload.get("backbone", payload.get("model_arch", args.backbone)))
        if checkpoint_backbone != backbone_name:
            raise ValueError(f"Checkpoint backbone={checkpoint_backbone} but requested {backbone_name}")
        backbone.load_state_dict(payload["backbone_state_dict"], strict=True)
        head.load_state_dict(payload["arcface_state_dict"], strict=True)
        history = [EpochMetrics(**x) for x in payload.get("history", [])]
        start_epoch = int(payload.get("epoch", len(history))) + 1
        if args.resume_optimizer:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
            scheduler.load_state_dict(payload["scheduler_state_dict"])

    config = vars(args).copy()
    config.update(
        {
            "total_rows": total_rows,
            "num_classes": num_classes,
            "steps_per_epoch": steps_per_epoch,
            "backbone": backbone_name,
            "embedding_size": embedding_size,
        }
    )
    with (args.output_dir / "training_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, default=str, ensure_ascii=False)

    end_epoch = start_epoch + args.epochs
    for epoch in range(start_epoch, end_epoch):
        dataset.set_epoch(epoch)
        backbone.train()
        head.train()
        running_loss = 0.0
        correct = 0
        seen = 0
        pbar = tqdm(loader, total=steps_per_epoch, desc=f"CASIA epoch {epoch}/{end_epoch - 1}", mininterval=5)
        for step, (images, labels) in enumerate(pbar, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                embeddings = backbone(images)
                logits = head(embeddings, labels)
                loss = F.cross_entropy(logits, labels)
            scale_before_step = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if not (device.type == "cuda" and scaler.get_scale() < scale_before_step):
                scheduler.step()

            batch = labels.numel()
            seen += batch
            running_loss += float(loss.item()) * batch
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            if step == 1 or step % 20 == 0:
                pbar.set_postfix(loss=running_loss / seen, acc=correct / seen, lr=optimizer.param_groups[0]["lr"])
        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=running_loss / max(seen, 1),
            train_accuracy=correct / max(seen, 1),
            samples_seen=seen,
            learning_rate=optimizer.param_groups[0]["lr"],
        )
        history.append(metrics)
        checkpoint = {
            "epoch": epoch,
            "model_arch": backbone_name,
            "backbone": backbone_name,
            "loss": "ArcFace",
            "image_size": args.image_size,
            "embedding_size": embedding_size,
            "num_classes": num_classes,
            "total_rows": total_rows,
            "samples_seen_this_epoch": seen,
            "backbone_state_dict": backbone.state_dict(),
            "arcface_state_dict": head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": [asdict(x) for x in history],
        }
        torch.save(checkpoint, args.output_dir / f"epoch_{epoch:03d}.pth")
        torch.save(checkpoint, args.output_dir / "latest.pth")
        with (args.output_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump([asdict(x) for x in history], f, indent=2, ensure_ascii=False)
        print(asdict(metrics))


if __name__ == "__main__":
    main()
