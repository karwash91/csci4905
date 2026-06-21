"""Analyze learned CNN feature space with PCA."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

MPL_CACHE_DIR = Path("outputs/.matplotlib").resolve()
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from train_cnn import CriticalityCNN, ForestFireDataset, IDX_TO_LABEL, get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--model-path", type=Path, default=Path("outputs/baseline_cnn/model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/feature_space"))
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def extract_features(model: CriticalityCNN, loader: DataLoader, device: torch.device):
    """Extract the CNN feature vector before the classifier head."""
    model.eval()
    features, labels = [], []

    with torch.no_grad():
        for grids, batch_labels in loader:
            feature_batch = model.features(grids.to(device)).flatten(1)
            features.append(feature_batch.cpu().numpy())
            labels.append(batch_labels.numpy())

    return np.concatenate(features), np.concatenate(labels)


def pca_2d(features: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """Project features to two dimensions using NumPy SVD."""
    centered = features - features.mean(axis=0, keepdims=True)
    scaled = centered / (features.std(axis=0, keepdims=True) + 1e-8)
    _, singular_values, components = np.linalg.svd(scaled, full_matrices=False)

    embedding = scaled @ components[:2].T
    variance = singular_values**2
    explained = (variance[:2] / variance.sum()).tolist()
    return embedding, explained


def centroid_summary(embedding: np.ndarray, labels: np.ndarray) -> dict:
    """Compute simple class centroids in PCA space."""
    centroids = {}
    for idx, label in IDX_TO_LABEL.items():
        class_points = embedding[labels == idx]
        centroids[label] = class_points.mean(axis=0).round(4).tolist()
    return centroids


def plot_embedding(embedding: np.ndarray, labels: np.ndarray, output_path: Path) -> None:
    colors = {
        0: "#2f80ed",
        1: "#f2a93b",
        2: "#d94f45",
    }

    plt.figure(figsize=(8, 6))
    for idx, label in IDX_TO_LABEL.items():
        points = embedding[labels == idx]
        plt.scatter(
            points[:, 0],
            points[:, 1],
            s=14,
            alpha=0.65,
            color=colors[idx],
            label=label,
            edgecolors="none",
        )

    plt.title("CNN Feature Space PCA")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = ForestFireDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size)
    device = get_device()

    model = CriticalityCNN().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    features, labels = extract_features(model, loader, device)
    embedding, explained = pca_2d(features)
    centroids = centroid_summary(embedding, labels)

    plot_embedding(embedding, labels, args.output_dir / "cnn_feature_pca.png")

    summary = {
        "feature_shape": list(features.shape),
        "pca_explained_variance": explained,
        "class_centroids": centroids,
    }
    (args.output_dir / "feature_space_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Feature shape: {features.shape}")
    print(f"PCA explained variance: {explained}")
    print(f"Class centroids: {centroids}")
    print(f"Saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()

