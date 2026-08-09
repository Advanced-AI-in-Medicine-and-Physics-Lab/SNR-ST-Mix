from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageGeneDataset(Dataset):
    def __init__(self, paths: list[Path], genes: np.ndarray, transform: Callable):
        self.paths, self.genes, self.transform = list(paths), genes.astype(np.float32), transform

    def __len__(self) -> int:
        return len(self.paths)

    def _load(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._load(index), torch.from_numpy(self.genes[index])


class SNRMixDataset(ImageGeneDataset):
    """Label-neighborhood SNR-ST-Mix used by the main experiment."""

    def __init__(self, paths: list[Path], genes: np.ndarray, distances: np.ndarray,
                 transform: Callable, alpha: float = 1.0, sigma: float | None = None,
                 k_neighbors: int = 10, include_self: bool = True):
        super().__init__(paths, genes, transform)
        self.distances = np.asarray(distances, dtype=np.float32)
        if self.distances.shape != (len(self), len(self)):
            raise ValueError("distances must have shape (N, N)")
        if len(self) < 2:
            raise ValueError("SNR-ST-Mix requires at least two training samples")
        self.alpha = float(alpha)
        self.sigma = max(float(sigma or np.median(self.distances) / 2), 1e-8)
        self.include_self = include_self
        self.k_neighbors = min(int(k_neighbors), len(self) if include_self else len(self) - 1)
        if self.include_self and self.k_neighbors < 2:
            raise ValueError("k_neighbors must be at least 2 when include_self=True")
        self.probabilities = self._conditional_probabilities()
        self.neighbors = self._nearest_neighbors()

    def _conditional_probabilities(self) -> np.ndarray:
        probabilities = np.exp(-self.distances / (2 * self.sigma**2))
        np.fill_diagonal(probabilities, 0)
        row_sums = probabilities.sum(axis=1, keepdims=True)
        zero_rows = row_sums[:, 0] <= 0
        if np.any(zero_rows):
            probabilities[zero_rows] = 1
            np.fill_diagonal(probabilities, 0)
            row_sums = probabilities.sum(axis=1, keepdims=True)
        return probabilities / np.clip(row_sums, 1e-8, None)

    def _nearest_neighbors(self) -> np.ndarray:
        order = np.argsort(self.distances, axis=1)
        if self.include_self:
            return order[:, :self.k_neighbors]
        rows = [row[row != i][:self.k_neighbors] for i, row in enumerate(order)]
        return np.stack(rows)

    def __getitem__(self, index: int):
        xi, yi = super().__getitem__(index)
        neighbors = self.neighbors[index]
        weights = self.probabilities[index, neighbors].copy()
        weights[neighbors == index] = 0
        if weights.sum() <= 0:
            weights = (neighbors != index).astype(np.float32)
        weights /= weights.sum()
        partner = int(np.random.choice(neighbors, p=weights))
        xj, yj = super().__getitem__(partner)
        lam = float(np.random.beta(self.alpha, self.alpha))
        weight = float(self.probabilities[index, partner])
        return (lam * xi + (1 - lam) * xj, lam * yi + (1 - lam) * yj,
                xi, yi, xj, yj, lam, index, partner, weight)
