#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
import yaml
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from snr_st_mix.augmentation import ImageGeneDataset, SNRMixDataset
from snr_st_mix.config import apply_overrides, load_config
from snr_st_mix.data import distance_matrix, load_hest_sample, random_split_indices
from snr_st_mix.models import GeneExpressionPredictor
from snr_st_mix.trainer import Trainer
from snr_st_mix.utils import create_logger, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Train SNR-ST-Mix")
    parser.add_argument("--config", required=True, help="Path to a YAML configuration")
    parser.add_argument("overrides", nargs="*", help="Optional dotted key=value overrides")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.overrides)
    seed_everything(config["seed"])
    output_dir = Path(config["output_dir"]) / config["data"]["sample"] / f"seed_{config['seed']}"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    logger = create_logger(output_dir)

    data = load_hest_sample(config["data"]["root"], config["data"]["patch_root"],
                            config["data"]["sample"], config["data"]["num_genes"])
    train_idx, val_idx, test_idx = random_split_indices(
        len(data.expression), config["data"]["train_fraction"],
        config["data"]["val_fraction"], config["seed"])
    np.savez(output_dir / "split_indices.npz", train=train_idx, validation=val_idx, test=test_idx)

    transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])
    paths = np.asarray(data.image_paths, dtype=object)
    augmentation = config["augmentation"]
    train_set = SNRMixDataset(
        paths[train_idx].tolist(), data.expression[train_idx],
        distance_matrix(data.expression[train_idx]), transform,
        alpha=augmentation["alpha"], sigma=augmentation.get("sigma"),
        k_neighbors=augmentation["k_neighbors"], include_self=augmentation["include_self"])
    val_set = ImageGeneDataset(paths[val_idx].tolist(), data.expression[val_idx], transform)
    test_set = ImageGeneDataset(paths[test_idx].tolist(), data.expression[test_idx], transform)
    loader_args = {"batch_size": config["training"]["batch_size"],
                   "num_workers": config["training"]["num_workers"],
                   "pin_memory": torch.cuda.is_available()}
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    test_loader = DataLoader(test_set, shuffle=False, **loader_args)

    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    model = GeneExpressionPredictor(data.expression.shape[1], **config["model"]).to(device)
    training = config["training"]
    optimizer = AdamW(model.parameters(), lr=training["learning_rate"],
                      betas=tuple(training["betas"]), eps=training["eps"],
                      weight_decay=training["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=training["epochs"] * max(1, len(train_loader)),
                                  eta_min=training["minimum_learning_rate"])
    trainer = Trainer(model, optimizer, scheduler, device, output_dir, logger, config["loss_weights"])
    trainer.fit(train_loader, val_loader, training["epochs"])
    metrics = trainer.test_best(test_loader)
    logger.info("Test metrics: %s", json.dumps(metrics))


if __name__ == "__main__":
    main()
