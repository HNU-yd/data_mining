#!/usr/bin/env python3
"""Fine-tune a scratch face backbone with hard-example self-distillation."""

from __future__ import annotations

import argparse
import json
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
    from train_casia_parquet_arcface import (
        ArcMarginProduct,
        count_rows_and_classes,
        estimate_loader_batches,
        estimate_max_sample_batches,
    )
except ModuleNotFoundError:
    from .face_backbones import build_backbone, canonical_backbone_name
    from .train_casia_parquet_arcface import (
        ArcMarginProduct,
        count_rows_and_classes,
        estimate_loader_batches,
        estimate_max_sample_batches,
    )

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "casia_webface_parquet"
DEFAULT_SOURCE_CHECKPOINT = PROJECT_ROOT / "models" / "advanced_ir18_arcface" / "epoch_020.pth"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "self_hsd_ir18_arcface"


@dataclass
class HSDEpochMetrics:
    epoch: int
    train_loss: float
    arcface_loss: float
    distill_loss: float
    train_accuracy: float
    hard_weight_mean: float
    teacher_student_cosine: float
    samples_seen: int
    learning_rate: float


class CASIAHSDIterable(IterableDataset):
    def __init__(
        self,
        shards: list[Path],
        image_size: int,
        shuffle_buffer: int,
        seed: int,
        max_samples: int | None = None,
    ):
        self.shards = shards
        self.image_size = image_size
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.max_samples = max_samples
        self.epoch = 0
        self.weak_transform = self._build_weak_transform()
        self.strong_transform = self._build_strong_transform()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _build_weak_transform(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size), antialias=True),
                transforms.PILToTensor(),
                transforms.Lambda(lambda x: fixed_image_standardization(x.float())),
            ]
        )

    def _build_strong_transform(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(self.image_size, scale=(0.82, 1.0), antialias=True),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(12),
                transforms.ColorJitter(brightness=0.16, contrast=0.16, saturation=0.10, hue=0.02),
                transforms.RandomGrayscale(p=0.04),
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

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, int]]:
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

    def _decode(self, item: tuple[bytes, int]) -> tuple[torch.Tensor, torch.Tensor, int]:
        image_bytes, label = item
        with Image.open(BytesIO(image_bytes)) as img:
            image = img.convert("RGB")
            return self.weak_transform(image), self.strong_transform(image), label


def load_backbone_from_checkpoint(path: Path, device: torch.device) -> tuple[nn.Module, int, str, dict]:
    payload = torch.load(path, map_location="cpu")
    backbone_name = canonical_backbone_name(payload.get("backbone", payload.get("model_arch", "ir_resnet18")))
    backbone, embedding_size, canonical = build_backbone(backbone_name, pretrained_model=None)
    backbone.load_state_dict(payload["backbone_state_dict"], strict=True)
    return backbone.to(device), embedding_size, canonical, payload


def hard_weighted_arcface_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    hard_weight: float,
    hard_gamma: float,
) -> tuple[torch.Tensor, float]:
    ce = F.cross_entropy(logits, labels, reduction="none")
    with torch.no_grad():
        true_prob = F.softmax(logits.float(), dim=1).gather(1, labels.view(-1, 1)).squeeze(1)
        raw_weights = 1.0 + hard_weight * torch.pow(1.0 - true_prob, hard_gamma)
        weights = raw_weights / raw_weights.mean().clamp_min(1e-6)
    return (ce * weights).mean(), float(raw_weights.mean().item())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--student-checkpoint", type=Path, default=DEFAULT_SOURCE_CHECKPOINT)
    parser.add_argument("--teacher-checkpoint", type=Path, default=DEFAULT_SOURCE_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--arc-scale", type=float, default=64.0)
    parser.add_argument("--arc-margin", type=float, default=0.35)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--hard-weight", type=float, default=1.0)
    parser.add_argument("--hard-gamma", type=float, default=2.0)
    parser.add_argument("--shuffle-buffer", type=int, default=16384)
    parser.add_argument("--max-samples", type=int, default=None, help="debug only; default trains every row")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shards = sorted(args.data_dir.glob("train-*-of-00020.parquet"))
    if len(shards) != 20:
        raise FileNotFoundError(f"Expected 20 parquet shards under {args.data_dir}, found {len(shards)}")
    if not args.student_checkpoint.is_file():
        raise FileNotFoundError(args.student_checkpoint)
    if not args.teacher_checkpoint.is_file():
        raise FileNotFoundError(args.teacher_checkpoint)

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

    dataset = CASIAHSDIterable(
        shards=shards,
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
    student, embedding_size, backbone_name, student_payload = load_backbone_from_checkpoint(args.student_checkpoint, device)
    teacher, teacher_embedding_size, teacher_backbone_name, _ = load_backbone_from_checkpoint(args.teacher_checkpoint, device)
    if teacher_backbone_name != backbone_name or teacher_embedding_size != embedding_size:
        raise ValueError(f"Teacher backbone={teacher_backbone_name}, student backbone={backbone_name}")
    if int(student_payload.get("num_classes", num_classes)) != num_classes:
        raise ValueError(f"Checkpoint num_classes={student_payload.get('num_classes')} but data has {num_classes}")

    head = ArcMarginProduct(embedding_size, num_classes, scale=args.arc_scale, margin=args.arc_margin).to(device)
    head.load_state_dict(student_payload["arcface_state_dict"], strict=True)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.SGD(
        list(student.parameters()) + list(head.parameters()),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * steps_per_epoch)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    source_epoch = int(student_payload.get("epoch", 0))
    start_epoch = source_epoch + 1
    end_epoch = start_epoch + args.epochs
    history: list[HSDEpochMetrics] = []

    config = vars(args).copy()
    config.update(
        {
            "method": "hard_example_self_distillation",
            "source_epoch": source_epoch,
            "total_rows": total_rows,
            "num_classes": num_classes,
            "steps_per_epoch": steps_per_epoch,
            "backbone": backbone_name,
            "embedding_size": embedding_size,
            "teacher_pretrained_backbone": False,
            "student_pretrained_backbone": False,
        }
    )
    with (args.output_dir / "training_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, default=str, ensure_ascii=False)

    for epoch in range(start_epoch, end_epoch):
        dataset.set_epoch(epoch)
        student.train()
        head.train()
        running_total = 0.0
        running_arc = 0.0
        running_distill = 0.0
        running_weight = 0.0
        running_cosine = 0.0
        correct = 0
        seen = 0
        pbar = tqdm(loader, total=steps_per_epoch, desc=f"HSD epoch {epoch}/{end_epoch - 1}", mininterval=5)
        for step, (weak_images, strong_images, labels) in enumerate(pbar, start=1):
            weak_images = weak_images.to(device, non_blocking=True)
            strong_images = strong_images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                teacher_embeddings = teacher(weak_images)
            teacher_norm = F.normalize(teacher_embeddings.float(), p=2, dim=1)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                student_embeddings = student(strong_images)
                logits = head(student_embeddings, labels)
                arc_loss, raw_weight_mean = hard_weighted_arcface_loss(
                    logits,
                    labels,
                    hard_weight=args.hard_weight,
                    hard_gamma=args.hard_gamma,
                )
            student_norm = F.normalize(student_embeddings.float(), p=2, dim=1)
            cosine = F.cosine_similarity(student_norm, teacher_norm, dim=1)
            distill_loss = (1.0 - cosine).mean()
            loss = arc_loss + args.distill_weight * distill_loss

            scale_before_step = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if not (device.type == "cuda" and scaler.get_scale() < scale_before_step):
                scheduler.step()

            batch = labels.numel()
            seen += batch
            running_total += float(loss.item()) * batch
            running_arc += float(arc_loss.item()) * batch
            running_distill += float(distill_loss.item()) * batch
            running_weight += raw_weight_mean * batch
            running_cosine += float(cosine.mean().item()) * batch
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            if step == 1 or step % 20 == 0:
                pbar.set_postfix(
                    loss=running_total / seen,
                    arc=running_arc / seen,
                    kd=running_distill / seen,
                    acc=correct / seen,
                    lr=optimizer.param_groups[0]["lr"],
                )

        metrics = HSDEpochMetrics(
            epoch=epoch,
            train_loss=running_total / max(seen, 1),
            arcface_loss=running_arc / max(seen, 1),
            distill_loss=running_distill / max(seen, 1),
            train_accuracy=correct / max(seen, 1),
            hard_weight_mean=running_weight / max(seen, 1),
            teacher_student_cosine=running_cosine / max(seen, 1),
            samples_seen=seen,
            learning_rate=optimizer.param_groups[0]["lr"],
        )
        history.append(metrics)
        checkpoint = {
            "epoch": epoch,
            "model_arch": backbone_name,
            "backbone": backbone_name,
            "loss": "ArcFace + hard-example self-distillation",
            "image_size": args.image_size,
            "embedding_size": embedding_size,
            "num_classes": num_classes,
            "total_rows": total_rows,
            "samples_seen_this_epoch": seen,
            "source_checkpoint": str(args.student_checkpoint),
            "teacher_checkpoint": str(args.teacher_checkpoint),
            "source_epoch": source_epoch,
            "distill_weight": args.distill_weight,
            "hard_weight": args.hard_weight,
            "hard_gamma": args.hard_gamma,
            "backbone_state_dict": student.state_dict(),
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
