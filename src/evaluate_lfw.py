#!/usr/bin/env python3
"""Evaluate FaceNet/InceptionResnetV1 on the LFW 6000 verification pairs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN, fixed_image_standardization
from PIL import Image
from sklearn.metrics import auc, confusion_matrix, roc_curve
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LFW_ROOT = PROJECT_ROOT / "data" / "raw" / "lfw-deepfunneled"
DEFAULT_PAIRS = PROJECT_ROOT / "design" / "lfw_test_pair.txt"
DEFAULT_OUTPUT = PROJECT_ROOT / "results"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"


@dataclass
class Pair:
    image_a: str
    image_b: str
    label: int


@dataclass
class FoldResult:
    fold: int
    threshold: float
    accuracy: float
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int


class LFWImageDataset(Dataset):
    def __init__(
        self,
        root: Path,
        rel_paths: list[str],
        preprocess: str,
        image_size: int,
        mtcnn_margin: int,
        device: torch.device,
    ):
        self.root = root
        self.rel_paths = rel_paths
        self.preprocess = preprocess
        self.image_size = image_size
        self.device = device
        self.resize = transforms.Resize((image_size, image_size), antialias=True)
        self.to_tensor = transforms.PILToTensor()
        self.mtcnn = None
        if preprocess == "mtcnn":
            self.mtcnn = MTCNN(image_size=image_size, margin=mtcnn_margin, post_process=True, device=device)

    def __len__(self) -> int:
        return len(self.rel_paths)

    def __getitem__(self, index: int) -> tuple[str, torch.Tensor, bool]:
        rel_path = self.rel_paths[index]
        image_path = self.root / rel_path
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            if self.preprocess == "resize":
                tensor = self.to_tensor(self.resize(img)).float()
                tensor = fixed_image_standardization(tensor)
                return rel_path, tensor, True

            assert self.mtcnn is not None
            face = self.mtcnn(img)
            if face is None:
                tensor = self.to_tensor(self.resize(img)).float()
                tensor = fixed_image_standardization(tensor)
                return rel_path, tensor, False
            return rel_path, face.cpu(), True


def parse_pairs(path: Path) -> list[Pair]:
    pairs: list[Pair] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 3:
                raise ValueError(f"Invalid pair line {line_no}: {line!r}")
            pairs.append(Pair(parts[0], parts[1], int(parts[2])))
    if len(pairs) != 6000:
        raise ValueError(f"Expected 6000 LFW pairs, got {len(pairs)}")
    labels = np.array([p.label for p in pairs])
    if labels.sum() != 3000 or len(labels) - labels.sum() != 3000:
        raise ValueError("Expected 3000 positive and 3000 negative pairs")
    return pairs


def unique_image_paths(pairs: Iterable[Pair]) -> list[str]:
    paths = sorted({p.image_a for p in pairs} | {p.image_b for p in pairs})
    return paths


def collate_images(batch: list[tuple[str, torch.Tensor, bool]]) -> tuple[list[str], torch.Tensor, list[bool]]:
    paths, tensors, detected = zip(*batch)
    return list(paths), torch.stack(list(tensors), dim=0), list(detected)


def build_model(model_name: str, device: torch.device, checkpoint: Path | None = None) -> InceptionResnetV1:
    pretrained = None if checkpoint else model_name
    model = InceptionResnetV1(pretrained=pretrained).eval().to(device)
    if checkpoint:
        payload = torch.load(checkpoint, map_location="cpu")
        state = payload.get("backbone_state_dict", payload.get("model_state_dict", payload))
        missing, unexpected = model.load_state_dict(state, strict=False)
        allowed_unexpected = [k for k in unexpected if k.startswith("logits.")]
        if missing or len(allowed_unexpected) != len(unexpected):
            raise RuntimeError(f"Checkpoint load issue. missing={missing}, unexpected={unexpected}")
    return model


def extract_embeddings(
    model: InceptionResnetV1,
    dataset: LFWImageDataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    tta_flip: bool,
) -> tuple[dict[str, np.ndarray], dict[str, bool]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_images,
        pin_memory=(device.type == "cuda"),
    )
    embeddings: dict[str, np.ndarray] = {}
    detected: dict[str, bool] = {}
    with torch.inference_mode():
        for rel_paths, batch, batch_detected in tqdm(loader, desc="Extract embeddings"):
            batch = batch.to(device, non_blocking=True)
            if tta_flip:
                flipped = torch.flip(batch, dims=[3])
                out_all = model(torch.cat([batch, flipped], dim=0))
                out = (out_all[: batch.size(0)] + out_all[batch.size(0) :]) / 2.0
            else:
                out = model(batch)
            out = torch.nn.functional.normalize(out, p=2, dim=1)
            out_np = out.detach().cpu().numpy().astype(np.float32)
            for rel_path, emb, ok in zip(rel_paths, out_np, batch_detected):
                embeddings[rel_path] = emb
                detected[rel_path] = bool(ok)
    return embeddings, detected


def pair_scores(pairs: list[Pair], embeddings: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    scores = np.empty(len(pairs), dtype=np.float32)
    labels = np.empty(len(pairs), dtype=np.int64)
    for i, pair in enumerate(pairs):
        emb_a = embeddings[pair.image_a]
        emb_b = embeddings[pair.image_b]
        scores[i] = float(np.dot(emb_a, emb_b))
        labels[i] = pair.label
    return scores, labels


def best_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    order = np.argsort(scores)
    sorted_scores = scores[order]
    candidates = np.empty(len(sorted_scores) + 1, dtype=np.float32)
    candidates[0] = sorted_scores[0] - 1e-6
    candidates[-1] = sorted_scores[-1] + 1e-6
    if len(sorted_scores) > 1:
        candidates[1:-1] = (sorted_scores[:-1] + sorted_scores[1:]) / 2.0

    best_acc = -1.0
    best_thr = float(candidates[0])
    for threshold in candidates:
        pred = (scores >= threshold).astype(np.int64)
        acc = float((pred == labels).mean())
        if acc > best_acc:
            best_acc = acc
            best_thr = float(threshold)
    return best_thr, best_acc


def fold_indices(num_pairs: int = 6000, folds: int = 10) -> list[np.ndarray]:
    if num_pairs != 6000 or folds != 10:
        raise ValueError("This helper is for the official 6000-pair, 10-fold LFW protocol")
    fold_ids = []
    for fold in range(folds):
        pos = np.arange(fold * 300, (fold + 1) * 300)
        neg = np.arange(3000 + fold * 300, 3000 + (fold + 1) * 300)
        fold_ids.append(np.concatenate([pos, neg]))
    return fold_ids


def evaluate_10_fold(scores: np.ndarray, labels: np.ndarray) -> tuple[list[FoldResult], np.ndarray]:
    ids = fold_indices(len(scores), 10)
    all_indices = np.arange(len(scores))
    predictions = np.zeros_like(labels)
    results: list[FoldResult] = []
    for fold, test_idx in enumerate(ids, start=1):
        train_idx = np.setdiff1d(all_indices, test_idx)
        threshold, _ = best_threshold(scores[train_idx], labels[train_idx])
        pred = (scores[test_idx] >= threshold).astype(np.int64)
        predictions[test_idx] = pred
        tn, fp, fn, tp = confusion_matrix(labels[test_idx], pred, labels=[0, 1]).ravel()
        results.append(
            FoldResult(
                fold=fold,
                threshold=threshold,
                accuracy=float((pred == labels[test_idx]).mean()),
                true_positive=int(tp),
                false_positive=int(fp),
                true_negative=int(tn),
                false_negative=int(fn),
            )
        )
    return results, predictions


def save_scores_csv(path: Path, pairs: list[Pair], scores: np.ndarray, labels: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_a", "image_b", "label", "cosine_score"])
        for pair, score, label in zip(pairs, scores, labels):
            writer.writerow([pair.image_a, pair.image_b, int(label), f"{float(score):.8f}"])


def save_fold_csv(path: Path, fold_results: list[FoldResult]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(fold_results[0]).keys()))
        writer.writeheader()
        for result in fold_results:
            writer.writerow(asdict(result))


def plot_roc(path: Path, labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = float(auc(fpr, tpr))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("LFW ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return roc_auc


def plot_confusion(path: Path, labels: np.ndarray, predictions: np.ndarray) -> list[list[int]]:
    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Different", "Same"])
    ax.set_yticks([0, 1], labels=["Different", "Same"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title("LFW Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return cm.astype(int).tolist()


def plot_score_histogram(path: Path, labels: np.ndarray, scores: np.ndarray, threshold: float) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(scores[labels == 1], bins=60, alpha=0.75, label="Same person")
    ax.hist(scores[labels == 0], bins=60, alpha=0.75, label="Different people")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5, label=f"Global threshold {threshold:.4f}")
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Pair Count")
    ax.set_title("LFW Pair Score Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def verify_files_exist(lfw_root: Path, pairs: list[Pair]) -> None:
    missing = []
    for rel_path in unique_image_paths(pairs):
        if not (lfw_root / rel_path).is_file():
            missing.append(rel_path)
            if len(missing) >= 10:
                break
    if missing:
        raise FileNotFoundError(f"Missing LFW images under {lfw_root}: {missing}")


def save_model_artifact(model_name: str, model_dir: Path, checkpoint: Path | None = None) -> Path:
    if checkpoint is not None:
        return checkpoint.resolve()
    checkpoint_name = {
        "casia-webface": "20180408-102900-casia-webface.pt",
        "vggface2": "20180402-114759-vggface2.pt",
    }[model_name]
    source = Path(os.environ.get("TORCH_HOME", str(model_dir / "torch"))) / "checkpoints" / checkpoint_name
    target = model_dir / f"facenet_{model_name.replace('-', '_')}.pth"
    if source.exists():
        shutil.copy2(source, target)
    return target


def write_metrics(
    path: Path,
    args: argparse.Namespace,
    labels: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
    fold_results: list[FoldResult],
    global_threshold: float,
    global_accuracy: float,
    roc_auc: float,
    confusion: list[list[int]],
    detected: dict[str, bool],
    model_artifact: Path,
) -> None:
    fold_accuracies = [r.accuracy for r in fold_results]
    metrics = {
        "model": "local_checkpoint" if args.checkpoint else args.model,
        "pretrained_model_arg": args.model,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "preprocess": args.preprocess,
        "mtcnn_margin": int(args.mtcnn_margin),
        "tta_flip": bool(args.tta_flip),
        "image_size": int(args.image_size),
        "device": args.device,
        "lfw_root": str(args.lfw_root),
        "pairs_file": str(args.pairs_file),
        "num_pairs": int(len(labels)),
        "num_positive_pairs": int(labels.sum()),
        "num_negative_pairs": int(len(labels) - labels.sum()),
        "num_unique_images": int(len(detected)),
        "mtcnn_or_resize_success_rate": float(np.mean(list(detected.values()))),
        "ten_fold_accuracy_mean": float(np.mean(fold_accuracies)),
        "ten_fold_accuracy_std": float(np.std(fold_accuracies)),
        "ten_fold_accuracies": fold_accuracies,
        "global_best_threshold": float(global_threshold),
        "global_best_accuracy": float(global_accuracy),
        "roc_auc": float(roc_auc),
        "confusion_matrix_10_fold": confusion,
        "model_artifact": str(model_artifact),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lfw-root", type=Path, default=DEFAULT_LFW_ROOT)
    parser.add_argument("--pairs-file", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--model", choices=["casia-webface", "vggface2"], default="casia-webface")
    parser.add_argument("--checkpoint", type=Path, default=None, help="local scratch-trained backbone checkpoint")
    parser.add_argument("--preprocess", choices=["resize", "mtcnn"], default="resize")
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--mtcnn-margin", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--tta-flip", action="store_true", help="average embeddings from original and horizontal flip")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_torch_home = args.model_dir / "torch"
    os.environ.setdefault("TORCH_HOME", str(project_torch_home))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    pairs = parse_pairs(args.pairs_file)
    verify_files_exist(args.lfw_root, pairs)

    device = torch.device(args.device)
    model = build_model(args.model, device, args.checkpoint)
    rel_paths = unique_image_paths(pairs)
    dataset = LFWImageDataset(args.lfw_root, rel_paths, args.preprocess, args.image_size, args.mtcnn_margin, device)
    embeddings, detected = extract_embeddings(model, dataset, args.batch_size, args.num_workers, device, args.tta_flip)
    scores, labels = pair_scores(pairs, embeddings)

    fold_results, predictions = evaluate_10_fold(scores, labels)
    global_threshold, global_accuracy = best_threshold(scores, labels)
    roc_auc = plot_roc(args.output_dir / "roc_curve.png", labels, scores)
    confusion = plot_confusion(args.output_dir / "confusion_matrix.png", labels, predictions)
    plot_score_histogram(args.output_dir / "score_histogram.png", labels, scores, global_threshold)

    save_scores_csv(args.output_dir / "pair_scores.csv", pairs, scores, labels)
    save_fold_csv(args.output_dir / "fold_metrics.csv", fold_results)
    np.savez_compressed(
        args.output_dir / "lfw_embeddings.npz",
        paths=np.array(rel_paths),
        embeddings=np.stack([embeddings[p] for p in rel_paths]),
    )
    model_artifact = save_model_artifact(args.model, args.model_dir, args.checkpoint)
    write_metrics(
        args.output_dir / "metrics.json",
        args,
        labels,
        scores,
        predictions,
        fold_results,
        global_threshold,
        global_accuracy,
        roc_auc,
        confusion,
        detected,
        model_artifact,
    )

    mean_acc = np.mean([r.accuracy for r in fold_results])
    std_acc = np.std([r.accuracy for r in fold_results])
    print(f"LFW 10-fold accuracy: {mean_acc:.4%} ± {std_acc:.4%}")
    print(f"ROC AUC: {roc_auc:.6f}")
    print(f"Global best accuracy: {global_accuracy:.4%} @ threshold {global_threshold:.6f}")
    print(f"Results written to: {args.output_dir}")


if __name__ == "__main__":
    main()
