import numpy as np
import torch


def pearson_loss(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-8):
    prediction = prediction - prediction.mean(dim=0, keepdim=True)
    target = target - target.mean(dim=0, keepdim=True)
    numerator = (prediction * target).sum(dim=0)
    denominator = prediction.square().sum(dim=0).add(eps).sqrt()
    denominator *= target.square().sum(dim=0).add(eps).sqrt()
    return -(numerator / (denominator + eps)).mean()


def mean_gene_pearson(target: np.ndarray, prediction: np.ndarray) -> float:
    correlations = [
        np.corrcoef(target[:, gene], prediction[:, gene])[0, 1]
        for gene in range(target.shape[1])
        if np.std(target[:, gene]) > 0 and np.std(prediction[:, gene]) > 0
    ]
    return float(np.nanmean(correlations)) if correlations else float("nan")

