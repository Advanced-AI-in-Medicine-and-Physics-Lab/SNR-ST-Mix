import os
import torch
import sys
import re
import glob
import random
import numpy as np
import pandas as pd
import scanpy as sc
from PIL import Image
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.spatial.distance import cdist
import torchvision.transforms as T
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import timm
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import logging
import h5py
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Read h5 metadata
with h5py.File("/media/hongyi/data/HEST/hest_data/patches/TENX13.h5", "r") as f:
    raw_barcodes = f["barcode"][:]
    coords_all   = f["coords"][:]
    n_total      =f["img"][:].shape[0]

barcodes = [b.decode("utf-8") if isinstance(b, bytes) else b for b in raw_barcodes.flatten()]

# Read AnnData & QC
adata = sc.read_h5ad("/media/hongyi/data/HEST/hest_data/st/TENX13.h5ad")
sc.pp.filter_cells(adata, min_genes=1)
sc.pp.filter_genes(adata, min_cells=1)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Only keep valid barcodes and corresponding coordinates
valid_barcodes = [bc for bc in barcodes if bc in adata.obs_names]
adata_sub = adata[valid_barcodes].copy()

matched_indices = [i for i, bc in enumerate(barcodes) if bc in adata_sub.obs_names]
barcodes = [barcodes[i] for i in matched_indices]
coords = coords_all[matched_indices]

# Ensure gene expression is in the same order
barcode_idx = [adata_sub.obs_names.get_loc(bc) for bc in barcodes]
gene_exp = adata_sub.X[barcode_idx].astype(np.float32)
gene_name = adata_sub.var_names
if not isinstance(gene_exp, np.ndarray):
    gene_exp = gene_exp.toarray()

# Select 250 genes by mean expression
gene_means = gene_exp.mean(axis=0)  # shape: (num_genes,)
top_k = 250
top_gene_indices = np.argsort(gene_means)[-top_k:]
gene_exp_top = gene_exp[:, top_gene_indices]  # shape: (num_spots, 250)
gene_names_top = gene_name[top_gene_indices]

# Reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# extract image paths and sort numerically
def numerical_sort_key(filename):
    m = re.search(r'(\d+)', os.path.basename(filename))
    return int(m.group(1)) if m else -1

# build ordered image list and align labels
image_dir = '/media/hongyi/data/HEST/hest_patches/TENX13'
all_images = sorted(glob.glob(os.path.join(image_dir, '*.png')), key=numerical_sort_key)
id2pos = {idx: pos for pos, idx in enumerate(matched_indices)}

images_ordered, pos_order = [], []
for p in all_images:
    n = numerical_sort_key(p)
    if n in id2pos:
        images_ordered.append(p)
        pos_order.append(id2pos[n])

# Ensure 1–1 alignment between images and labels
images = images_ordered
gene_exp_top = gene_exp_top[pos_order]          # reorder labels to match images
coords = coords[pos_order]
barcodes_matched = [barcodes[i] for i in pos_order]

# transforms (ImageNet stats for ViT)
processor = T.Compose([
    T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
])

# label distances (for C-Mixup kernel)
label_dist = pairwise_distances(gene_exp_top, metric='euclidean').astype(np.float32)  # (N, N)
# Sensible sigma for Gaussian kernel: ~1/4 median inter-label distance
SIGMA = float(np.median(label_dist)/2)

# Randomly split datasets
TRAIN_SPLIT  = 0.30
VAL_SPLIT    = 0.20

num_samples = len(gene_exp_top)
indices = np.arange(num_samples)
np.random.seed(SEED)
np.random.shuffle(indices)

train_size = int(num_samples * TRAIN_SPLIT)
val_size = int(num_samples * VAL_SPLIT)
test_size = num_samples - val_size - train_size

train_idx, val_idx, test_idx = (
    indices[:train_size],
    indices[train_size:train_size+val_size],
    indices[train_size+val_size:]
)

# --------------- training hyperparams ---------------
BATCH_SIZE   = 8
EPOCHS       = 30
LR           = 3e-5
BETAS        = (0.9, 0.98)
EPS          = 1e-6
WEIGHT_DECAY = 1e-4
ALPHA        = 1.0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gene_dim = gene_exp_top.shape[1]

class LabelMixupDataset(Dataset):
    def __init__(self, img_paths, genes, dist_matrix, proc, alpha=ALPHA, sigma=SIGMA):
        self.paths = img_paths
        self.genes = genes
        self.proc = proc
        self.dist_matrix = dist_matrix
        self.alpha = alpha
        self.sigma = sigma
        self.P = self._build_conditional_probs()

    def _build_conditional_probs(self):
        P = np.exp(-self.dist_matrix / (2 * self.sigma ** 2))
        np.fill_diagonal(P, 0)

        # Compute row sums and detect zero rows
        row_sums = P.sum(axis=1, keepdims=True)
        zero_rows = (row_sums == 0).flatten()

        # Fix zero rows: assign uniform distribution (excluding self)
        if np.any(zero_rows):
            print(f"[Warning] Found {zero_rows.sum()} rows with zero similarity. Assigning uniform probs.")
            P[zero_rows] = 1.0 / (P.shape[1] - 1)
            np.fill_diagonal(P, 0)
            row_sums = P.sum(axis=1, keepdims=True)  # Recompute

        # Safe division
        row_sums = np.clip(row_sums, 1e-8, np.inf)
        P /= row_sums

        # Sanity check
        assert not np.isnan(P).any(), "NaNs still present in conditional prob matrix"
        return P


    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        xi = self.proc(Image.open(self.paths[idx]).convert("RGB"))
        yi = torch.from_numpy(self.genes[idx])

        # Conditional sampling for mixup
        pj = self.P[idx]
        pj = pj / pj.sum()  # Ensure it sums exactly to 1
        j = np.random.choice(len(pj), p=pj)
        xj = self.proc(Image.open(self.paths[j]).convert("RGB"))
        yj = torch.from_numpy(self.genes[j])

        # Mixup
        lam = np.random.beta(self.alpha, self.alpha)
        x_mix = lam * xi + (1 - lam) * xj
        y_mix = lam * yi + (1 - lam) * yj

        return x_mix, y_mix

class ImageGeneDataset(Dataset):
    def __init__(self, images, genes, proc, cache_cap: int = 0):
        self.paths = list(images)
        self.genes = genes.astype(np.float32) if hasattr(genes, "astype") else genes
        self.proc  = proc
        self.cache_cap = int(cache_cap)
        self._cache = {} if self.cache_cap > 0 else None

    def __len__(self): return len(self.paths)

    def _load_tensor(self, path: str):
        if self._cache is not None:
            t = self._cache.get(path)
            if t is not None:
                return t.clone()
        with Image.open(path) as im:
            im = im.convert("RGB")
            t = self.proc(im)
        if self._cache is not None:
            if len(self._cache) >= self.cache_cap:
                self._cache.pop(next(iter(self._cache)))
            self._cache[path] = t
            return t.clone()
        return t

    def __getitem__(self, idx):
        img_t = self._load_tensor(self.paths[idx])
        gene  = torch.from_numpy(self.genes[idx])
        return img_t, gene

# Datasets
train_set = LabelMixupDataset([images[i] for i in train_idx], gene_exp_top[train_idx], label_dist[np.ix_(train_idx, train_idx)], processor)
val_set  = ImageGeneDataset([images[i] for i in val_idx],  gene_exp_top[val_idx],  processor)
test_set = ImageGeneDataset([images[i] for i in test_idx], gene_exp_top[test_idx], processor)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)


SAVE_ROOT = "/home/hongyi/project/mixup/model/Ablation/TENX13/mixup_label"
LOSS_ROOT = "/home/hongyi/project/mixup/loss_recording/Ablation/TENX13"

os.makedirs(SAVE_ROOT, exist_ok=True)
os.makedirs(LOSS_ROOT, exist_ok=True)

# --- Model ---
class ViTLargeGenePredictor(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.backbone = timm.create_model('vit_large_patch16_224', pretrained=True, num_classes=0)
        self.dropout  = nn.Dropout(0.2)
        self.head = nn.Linear(self.backbone.num_features, out_dim)
    def forward(self, images):
        feat = self.backbone(images)
        feat = self.dropout(feat)
        return self.head(feat)
    
model = ViTLargeGenePredictor(gene_dim).to(device)

criterion_mse = nn.MSELoss()
criterion_mae = nn.L1Loss()

optimizer  = optim.AdamW(
    model.parameters(),
    lr=LR, betas=BETAS, eps=EPS, weight_decay=WEIGHT_DECAY
)
total_steps = EPOCHS * max(1, len(train_loader))
scheduler   = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-7)

def setup_logger(output_dir: str, verbose: bool = True) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%m-%d %H:%M:%S")
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO if verbose else logging.WARNING)
    ch.setFormatter(fmt)
    fh = logging.FileHandler(os.path.join(output_dir, "train.log"), mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

@torch.no_grad()
def evaluate(loader):
    model.eval()
    mse_sum, mae_sum, n_sample = 0.0, 0.0, 0
    pearsons = []
    for imgs, genes in loader:
        imgs, genes = imgs.to(device), genes.to(device)
        preds = model(imgs)

        loss_mse = criterion_mse(preds, genes)
        loss_mae = criterion_mae(preds, genes)  # optional: leave MAE as-is

        mse_sum += loss_mse.item() * genes.size(0)
        mae_sum += loss_mae.item() * genes.size(0)
        n_sample += genes.size(0)

        # Calculate Pearson correlation for each gene
        g_true, g_pred = genes.cpu().numpy(), preds.cpu().numpy()
        for i in range(gene_dim):
            if np.std(g_true[:, i]) > 0 and np.std(g_pred[:, i]) > 0:
                pearsons.append(np.corrcoef(g_true[:, i], g_pred[:, i])[0, 1])

    mse  = mse_sum / n_sample
    mae  = mae_sum / n_sample
    pcc  = float(np.nanmean(pearsons)) if pearsons else np.nan
    return mse, mae, pcc

# --- Training Loop ---
logger = setup_logger(SAVE_ROOT, verbose=True)

train_mse_list, train_mae_list = [], []
val_mse_list, val_mae_list, val_pcc_list = [], [], []

best_val = float("inf")
logger.info("Starting training")
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_mse, running_mae, steps = 0.0, 0.0, 0
    
    for imgs, genes in train_loader:
        imgs, genes = imgs.to(device), genes.to(device)
        preds = model(imgs)

        loss_mse = criterion_mse(preds, genes)
        loss_mae = criterion_mae(preds, genes)

        optimizer.zero_grad()
        loss_mse.backward()
        optimizer.step()
        scheduler.step()
        running_mse += loss_mse.item()
        running_mae += loss_mae.item()
        steps += 1
        if steps % 50 == 0:
            logger.info(
                    f"Epoch {epoch:03d} | Step {steps}/{len(train_loader)} | "
                    f"BatchMSE {loss_mse.item():.4f} | BatchMAE {loss_mae.item():.4f} | "
                    f"AvgMSE {running_mse/steps:.4f} | AvgMAE {running_mae/steps:.4f} | "
                    f"LR {optimizer.param_groups[0]['lr']:.2e}"
                )
    
    val_mse, val_mae, val_pcc = evaluate(val_loader)
    logger.info(
            f"Epoch {epoch:03d} DONE | TrainAvgMSE {running_mse/steps:.4f} "
            f"| TrainAvgMAE {running_mae/steps:.4f} | ValMSE {val_mse:.4f} "
            f"| ValMAE {val_mae:.4f} | ValPCC {val_pcc:.4f}"
        )
    
    train_mse_list.append(running_mse / steps)
    train_mae_list.append(running_mae / steps)
    val_mse_list.append(val_mse)
    val_mae_list.append(val_mae)
    val_pcc_list.append(val_pcc)

    # save best + a resumable "last.pt"
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_val_mse": min(best_val, val_mse),
    }, os.path.join(SAVE_ROOT, "last.pt"))

    if val_mse < best_val:
        best_val = val_mse
        best_checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "mse": val_mse,
            "mae": val_mae,
            "pearson": val_pcc
        }
        torch.save(best_checkpoint, os.path.join(SAVE_ROOT, "best_model.pt"))
        logger.info(f"New best checkpoint saved @ epoch {epoch} (Val MSE {best_val:.4f})")

logger.info(f"Training finished. Best validation MSE = {best_val:.4f}")

# Load best model for test evaluation
checkpoint = torch.load(os.path.join(SAVE_ROOT, "best_model.pt"), map_location="cpu", weights_only=True)
model.load_state_dict(checkpoint["model_state"])
model.eval()

test_mse, test_mae, test_pcc = evaluate(test_loader)
logger.info(f"[TEST] MSE {test_mse:.4f} | MAE {test_mae:.4f} | PCC {test_pcc:.4f}")

# Save loss records
epochs = range(1, EPOCHS + 1)
record = {
    'epoch': epochs,
    'train_mse': train_mse_list,
    'train_mae': train_mae_list,
    'val_mse': val_mse_list,
    'val_mae': val_mae_list,
    'val_pcc': val_pcc_list
}

df = pd.DataFrame(record)
df.to_csv(os.path.join(LOSS_ROOT, "mixup_label.csv"), index=False)

# Persist test metrics as JSON for easy parsing later
import json
with open(os.path.join(SAVE_ROOT, "test_metrics.json"), "w") as f:
    json.dump({
        "test_mse": float(test_mse),
        "test_mae": float(test_mae),
        "test_pcc": float(test_pcc),
        "best_val_mse": float(best_val)
    }, f, indent=2)