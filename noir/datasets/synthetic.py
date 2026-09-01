"""Synthetic dataset generators for real machine learning training."""

from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset


class SpiralDataset(Dataset):
    """Generates a non-linear multi-class spiral dataset in N-dimensional space."""

    def __init__(
        self,
        num_samples_per_class: int = 500,
        num_classes: int = 4,
        noise: float = 0.2,
        embedding_dim: int = 16,
        seed: int = 42,
    ):
        np.random.seed(seed)
        total_samples = num_samples_per_class * num_classes

        # 2D base spiral coordinates
        x_2d = np.zeros((total_samples, 2), dtype=np.float32)
        y = np.zeros(total_samples, dtype=np.int64)

        for c in range(num_classes):
            ix = range(num_samples_per_class * c, num_samples_per_class * (c + 1))
            r = np.linspace(0.0, 1.0, num_samples_per_class)
            t = np.linspace(c * 4, (c + 1) * 4, num_samples_per_class) + np.random.randn(num_samples_per_class) * noise
            x_2d[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
            y[ix] = c

        # Project 2D coordinates into higher-dimensional feature space
        if embedding_dim > 2:
            projection_matrix = np.random.randn(2, embedding_dim).astype(np.float32) / np.sqrt(2)
            noise_matrix = np.random.randn(total_samples, embedding_dim).astype(np.float32) * 0.05
            x_high = np.dot(x_2d, projection_matrix) + noise_matrix
        else:
            x_high = x_2d

        self.features = torch.from_numpy(x_high).float()
        self.labels = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


class ManifoldClassificationDataset(Dataset):
    """Multi-modal non-linear manifold classification dataset."""

    def __init__(
        self,
        num_samples: int = 2000,
        input_dim: int = 16,
        num_classes: int = 4,
        seed: int = 42,
    ):
        np.random.seed(seed)
        samples_per_class = num_samples // num_classes

        features_list = []
        labels_list = []

        for c in range(num_classes):
            center = np.random.randn(input_dim) * 2.0
            cov = np.diag(np.random.uniform(0.5, 1.5, size=input_dim))
            feat = np.random.multivariate_normal(center, cov, size=samples_per_class).astype(np.float32)
            # Add non-linear transformation
            feat = np.sin(feat * 1.2) + 0.3 * feat
            labels = np.full(samples_per_class, c, dtype=np.int64)

            features_list.append(feat)
            labels_list.append(labels)

        self.features = torch.from_numpy(np.vstack(features_list)).float()
        self.labels = torch.from_numpy(np.concatenate(labels_list)).long()

        # Shuffle
        perm = torch.randperm(len(self.labels))
        self.features = self.features[perm]
        self.labels = self.labels[perm]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]
