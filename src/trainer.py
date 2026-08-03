from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .metrics import mean_gene_pearson, pearson_loss


class Trainer:
    def __init__(self, model, optimizer, scheduler, device, output_dir: Path,
                 logger, loss_weights: dict[str, float]):
        self.model, self.optimizer, self.scheduler = model, optimizer, scheduler
        self.device, self.output_dir, self.logger = device, output_dir, logger
        self.weights = loss_weights
        self.mse, self.mae = nn.MSELoss(), nn.L1Loss()

    @torch.no_grad()
    def evaluate(self, loader) -> dict[str, float]:
        self.model.eval()
        targets, predictions = [], []
        for images, genes in loader:
            targets.append(genes.numpy())
            predictions.append(self.model(images.to(self.device)).cpu().numpy())
        target, prediction = np.concatenate(targets), np.concatenate(predictions)
        return {
            "mse": float(np.mean((prediction - target) ** 2)),
            "mae": float(np.mean(np.abs(prediction - target))),
            "pcc": mean_gene_pearson(target, prediction),
        }

    def _train_epoch(self, loader) -> dict[str, float]:
        self.model.train()
        totals = {key: 0.0 for key in ("total", "vicinal", "consistency", "edge", "correlation")}
        for batch in loader:
            mixed_x, mixed_y, xi, yi, xj, yj, lam, _, _, edge_weight = batch
            mixed_x, mixed_y = mixed_x.to(self.device), mixed_y.to(self.device)
            xi, yi, xj, yj = xi.to(self.device), yi.to(self.device), xj.to(self.device), yj.to(self.device)
            lam = torch.as_tensor(lam, dtype=torch.float32, device=self.device).view(-1, 1)
            edge_weight = torch.as_tensor(edge_weight, dtype=torch.float32, device=self.device).view(-1, 1)

            batch_size = mixed_x.size(0)
            outputs = self.model(torch.cat((mixed_x, xi, xj)))
            mixed_hat, yi_hat, yj_hat = outputs[:batch_size], outputs[batch_size:2*batch_size], outputs[2*batch_size:]
            losses = {
                "vicinal": self.mse(mixed_hat, mixed_y),
                "consistency": self.mse(mixed_hat, lam * yi_hat + (1 - lam) * yj_hat),
                "edge": (edge_weight * ((yi_hat - yj_hat) - (yi - yj)).square()).mean(),
                "correlation": pearson_loss(mixed_hat, mixed_y),
            }
            loss = (
                losses["vicinal"]
                + self.weights["consistency"] * losses["consistency"]
                + self.weights["edge"] * losses["edge"]
                + self.weights["correlation"] * losses["correlation"]
            )
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            totals["total"] += loss.item()
            for name, value in losses.items():
                totals[name] += value.item()
        return {name: value / len(loader) for name, value in totals.items()}

    def fit(self, train_loader, val_loader, epochs: int) -> list[dict[str, float]]:
        history, best_mse = [], float("inf")
        for epoch in range(1, epochs + 1):
            training = self._train_epoch(train_loader)
            validation = self.evaluate(val_loader)
            row = {"epoch": epoch, **{f"train_{k}": v for k, v in training.items()},
                   **{f"val_{k}": v for k, v in validation.items()}}
            history.append(row)
            self.logger.info("Epoch %03d | train loss %.4f | val MSE %.4f | val PCC %.4f",
                             epoch, training["total"], validation["mse"], validation["pcc"])
            if validation["mse"] < best_mse:
                best_mse = validation["mse"]
                torch.save({"epoch": epoch, "model_state": self.model.state_dict(),
                            "validation": validation}, self.output_dir / "best_model.pt")
        pd.DataFrame(history).to_csv(self.output_dir / "history.csv", index=False)
        return history

    def test_best(self, loader) -> dict[str, float]:
        checkpoint = torch.load(self.output_dir / "best_model.pt", map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        metrics = self.evaluate(loader)
        with (self.output_dir / "test_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        return metrics
