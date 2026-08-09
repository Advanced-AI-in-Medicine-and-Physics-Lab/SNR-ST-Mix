from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import pairwise_distances


@dataclass
class SpatialExpressionData:
    image_paths: list[Path]
    expression: np.ndarray
    coordinates: np.ndarray
    barcodes: list[str]
    gene_names: np.ndarray


def _numeric_id(path: Path) -> int:
    match = re.search(r"(\d+)", path.name)
    return int(match.group(1)) if match else -1


def load_hest_sample(data_root: str | Path, patch_root: str | Path, sample: str,
                     num_genes: int = 250) -> SpatialExpressionData:
    """Load and align HEST metadata, expression, coordinates, and PNG patches."""
    import h5py
    import scanpy as sc

    data_root, patch_root = Path(data_root), Path(patch_root)
    with h5py.File(data_root / "patches" / f"{sample}.h5", "r") as handle:
        raw_barcodes = handle["barcode"][:]
        coordinates = handle["coords"][:]

    barcodes = [
        item.decode("utf-8") if isinstance(item, bytes) else str(item)
        for item in raw_barcodes.flatten()
    ]
    adata = sc.read_h5ad(data_root / "st" / f"{sample}.h5ad")
    sc.pp.filter_cells(adata, min_genes=1)
    sc.pp.filter_genes(adata, min_cells=1)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    matched_source = [i for i, barcode in enumerate(barcodes) if barcode in adata.obs_names]
    matched_barcodes = [barcodes[i] for i in matched_source]
    row_indices = [adata.obs_names.get_loc(barcode) for barcode in matched_barcodes]
    expression = adata.X[row_indices].astype(np.float32)
    if not isinstance(expression, np.ndarray):
        expression = expression.toarray()

    top_indices = np.argsort(expression.mean(axis=0))[-num_genes:]
    expression = expression[:, top_indices]
    gene_names = np.asarray(adata.var_names[top_indices])
    coordinates = coordinates[matched_source]

    source_to_position = {source: position for position, source in enumerate(matched_source)}
    image_paths, positions = [], []
    for path in sorted((patch_root / sample).glob("*.png"), key=_numeric_id):
        source_index = _numeric_id(path)
        if source_index in source_to_position:
            image_paths.append(path)
            positions.append(source_to_position[source_index])
    if not image_paths:
        raise FileNotFoundError(f"No aligned PNG patches found in {patch_root / sample}")

    return SpatialExpressionData(
        image_paths=image_paths,
        expression=expression[positions],
        coordinates=coordinates[positions],
        barcodes=[matched_barcodes[i] for i in positions],
        gene_names=gene_names,
    )


def random_split_indices(size: int, train_fraction: float, val_fraction: float,
                         seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if train_fraction + val_fraction >= 1:
        raise ValueError("train_fraction + val_fraction must be less than 1")
    indices = np.random.default_rng(seed).permutation(size)
    train_end = int(size * train_fraction)
    val_end = train_end + int(size * val_fraction)
    return indices[:train_end], indices[train_end:val_end], indices[val_end:]


def distance_matrix(values: np.ndarray) -> np.ndarray:
    return pairwise_distances(values, metric="euclidean").astype(np.float32)
