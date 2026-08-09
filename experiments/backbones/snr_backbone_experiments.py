"""
Backbone experiments for reviewer response:
1) Img2ST-Net-inspired fully convolutional regression backbone
2) MagNet-inspired multi-scale attention graph backbone
Each can be trained with either standard supervision or SNR-ST-Mix.

Adapt paths/configs from your current train.py.
"""
import os, re, glob, json, random, logging, sys
from dataclasses import dataclass
from typing import Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from PIL import Image
from sklearn.metrics import pairwise_distances

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as T
import timm
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


@dataclass
class Config:
    sample_id: str = "TENX14"
    patch_h5: str = "/home/hongyi/project/HEST/hest_data/patches/TENX14.h5"
    st_h5ad: str = "/home/hongyi/project/HEST/hest_data/st/TENX14.h5ad"
    image_dir: str = "/home/hongyi/project/HEST/hest_patches/TENX14"
    save_root: str = "/home/hongyi/project/Backbones/TENX14"
    loss_root: str = "/home/hongyi/project/Backbones/TENX14"
    top_k_genes: int = 250
    seed: int = 0
    train_split: float = 0.30
    val_split: float = 0.20
    batch_size: int = 8
    epochs: int = 200
    lr: float = 3e-5
    weight_decay: float = 1e-4
    alpha: float = 1.0
    k_neighbors: int = 10
    lambda_cons: float = 0.5
    lambda_edge: float = 50.0
    lambda_corr: float = 0.05
    num_workers: int = 4
    cache_cap: int = 0
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def numerical_sort_key(filename):
    m = re.search(r"(\d+)", os.path.basename(filename))
    return int(m.group(1)) if m else -1


def setup_logger(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger(output_dir)
    logger.setLevel(logging.DEBUG); logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%m-%d %H:%M:%S")
    ch = logging.StreamHandler(stream=sys.stdout); ch.setFormatter(fmt); ch.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(output_dir, "train.log"), mode="a"); fh.setFormatter(fmt); fh.setLevel(logging.DEBUG)
    logger.addHandler(ch); logger.addHandler(fh)
    return logger


def load_hest_data(cfg: Config):
    with h5py.File(cfg.patch_h5, "r") as f:
        raw_barcodes = f["barcode"][:]
        coords_all = f["coords"][:]
    barcodes = [b.decode("utf-8") if isinstance(b, bytes) else b for b in raw_barcodes.flatten()]

    adata = sc.read_h5ad(cfg.st_h5ad)
    sc.pp.filter_cells(adata, min_genes=1)
    sc.pp.filter_genes(adata, min_cells=1)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    valid_barcodes = [bc for bc in barcodes if bc in adata.obs_names]
    adata_sub = adata[valid_barcodes].copy()
    matched_indices = [i for i, bc in enumerate(barcodes) if bc in adata_sub.obs_names]
    barcodes = [barcodes[i] for i in matched_indices]
    coords = coords_all[matched_indices].astype(np.float32)

    barcode_idx = [adata_sub.obs_names.get_loc(bc) for bc in barcodes]
    gene_exp = adata_sub.X[barcode_idx].astype(np.float32)
    if not isinstance(gene_exp, np.ndarray):
        gene_exp = gene_exp.toarray()

    top_gene_indices = np.argsort(gene_exp.mean(axis=0))[-cfg.top_k_genes:]
    genes = gene_exp[:, top_gene_indices].astype(np.float32)

    all_images = sorted(glob.glob(os.path.join(cfg.image_dir, "*.png")), key=numerical_sort_key)
    id2pos = {idx: pos for pos, idx in enumerate(matched_indices)}
    images_ordered, pos_order = [], []
    for p in all_images:
        n = numerical_sort_key(p)
        if n in id2pos:
            images_ordered.append(p); pos_order.append(id2pos[n])

    images = images_ordered
    genes = genes[pos_order]
    coords = coords[pos_order]
    return images, genes, coords


class ImageGeneCoordDataset(Dataset):
    def __init__(self, paths, genes, coords, proc, cache_cap=0):
        self.paths = list(paths); self.genes = genes.astype(np.float32); self.coords = coords.astype(np.float32); self.proc = proc
        self.cache_cap = int(cache_cap); self._cache = {} if cache_cap > 0 else None
    def __len__(self): return len(self.paths)
    def _load_tensor(self, path):
        if self._cache is not None and path in self._cache: return self._cache[path].clone()
        with Image.open(path) as im: x = self.proc(im.convert("RGB"))
        if self._cache is not None:
            if len(self._cache) >= self.cache_cap: self._cache.pop(next(iter(self._cache)))
            self._cache[path] = x
        return x.clone() if self._cache is not None else x
    def __getitem__(self, idx):
        return self._load_tensor(self.paths[idx]), torch.from_numpy(self.genes[idx]), torch.from_numpy(self.coords[idx]), idx


class SNRMixCoordDataset(ImageGeneCoordDataset):
    def __init__(self, paths, genes, coords, dist_matrix, proc, alpha=1.0, sigma=None, k=10, cache_cap=0):
        super().__init__(paths, genes, coords, proc, cache_cap)
        self.D = np.asarray(dist_matrix, dtype=np.float32)
        self.alpha = float(alpha)
        self.sigma = float(np.median(self.D) / 2) if sigma is None else float(sigma)
        self.k = int(k)
        N = len(self.paths)
        assert self.D.shape == (N, N)
        self.P = np.exp(-self.D / (2 * self.sigma ** 2)); np.fill_diagonal(self.P, 0)
        self.P = self.P / np.clip(self.P.sum(axis=1, keepdims=True), 1e-8, np.inf)
        k_eff = min(self.k + 1, N)
        nn_idx = np.argpartition(self.D, kth=k_eff - 1, axis=1)[:, :k_eff]
        self.knn_idx = []
        for i, row in enumerate(nn_idx):
            row = row[row != i]
            self.knn_idx.append(row[np.argsort(self.D[i, row])[: self.k]])
        self.knn_idx = np.stack(self.knn_idx, axis=0)
    def __getitem__(self, idx):
        xi, yi, ci, _ = super().__getitem__(idx)
        neigh = self.knn_idx[idx]
        pk = self.P[idx, neigh].copy(); pk = pk / np.clip(pk.sum(), 1e-8, np.inf)
        j = int(np.random.choice(neigh, p=pk))
        xj, yj, cj, _ = super().__getitem__(j)
        lam = float(np.random.beta(self.alpha, self.alpha))
        xmix = lam * xi + (1 - lam) * xj
        ymix = lam * yi + (1 - lam) * yj
        cmix = lam * ci + (1 - lam) * cj
        return xmix, ymix, cmix, xi, yi, ci, xj, yj, cj, torch.tensor(lam, dtype=torch.float32), idx, j, torch.tensor(float(self.P[idx, j]), dtype=torch.float32)


class Img2STNetLite(nn.Module):
    """Img2ST-Net-inspired FCN: dense convolutional feature map -> global spot expression.
    This is a fair spot-level adaptation for 224x224 patches, not the official HD map code.
    """
    def __init__(self, out_dim, encoder="resnet50", pretrained=True, dropout=0.2):
        super().__init__()
        self.backbone = timm.create_model(encoder, pretrained=pretrained, features_only=True, out_indices=(1,2,3,4))
        chs = self.backbone.feature_info.channels()
        self.proj = nn.ModuleList([nn.Conv2d(c, 256, 1) for c in chs])
        self.fuse = nn.Sequential(
            nn.Conv2d(256 * len(chs), 512, 3, padding=1), nn.BatchNorm2d(512), nn.GELU(), nn.Dropout2d(dropout),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.GELU(),
            nn.Conv2d(512, out_dim, 1),
        )
    def forward(self, x, coords=None):
        feats = self.backbone(x)
        target_hw = feats[0].shape[-2:]
        ups = [F.interpolate(p(f), size=target_hw, mode="bilinear", align_corners=False) for p, f in zip(self.proj, feats)]
        gene_map = self.fuse(torch.cat(ups, dim=1))
        return gene_map.mean(dim=(2,3))


class BatchGraphAttention(nn.Module):
    def __init__(self, dim, heads=4, k=6):
        super().__init__(); self.heads=heads; self.k=k; self.scale=(dim//heads)**-0.5
        self.qkv = nn.Linear(dim, dim*3); self.out = nn.Linear(dim, dim); self.norm = nn.LayerNorm(dim)
    def forward(self, h, coords):
        B, D = h.shape
        if B <= 1: return h
        k = min(self.k, B)
        dist = torch.cdist(coords.float(), coords.float())
        knn = dist.topk(k=k, largest=False).indices
        qkv = self.qkv(self.norm(h)).view(B, 3, self.heads, D//self.heads)
        q, kvec, v = qkv[:,0], qkv[:,1], qkv[:,2]
        outs = []
        for i in range(B):
            nb = knn[i]
            att = (q[i:i+1] * kvec[nb]).sum(-1) * self.scale
            att = att.softmax(dim=0)
            outs.append((att.unsqueeze(-1) * v[nb]).sum(0).reshape(D))
        return h + self.out(torch.stack(outs, dim=0))


class MagNetLite(nn.Module):
    """MagNet-inspired backbone: multi-resolution visual features + coordinate kNN graph attention.
    This captures the reviewer-facing architectural idea without requiring the official repo dependencies.
    """
    def __init__(self, out_dim, encoder="resnet50", pretrained=True, hidden=512, graph_layers=2, graph_k=6, dropout=0.2):
        super().__init__()
        self.backbone = timm.create_model(encoder, pretrained=pretrained, features_only=True, out_indices=(1,2,3,4))
        chs = self.backbone.feature_info.channels()
        self.level_proj = nn.ModuleList([nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c, hidden), nn.GELU()) for c in chs])
        self.level_att = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)
        self.graph = nn.ModuleList([BatchGraphAttention(hidden, heads=4, k=graph_k) for _ in range(graph_layers)])
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout), nn.Linear(hidden, out_dim))
    def forward(self, x, coords=None):
        feats = self.backbone(x)
        tokens = torch.stack([p(f) for p, f in zip(self.level_proj, feats)], dim=1)
        h, _ = self.level_att(tokens, tokens, tokens)
        h = h.mean(dim=1)
        if coords is not None:
            for layer in self.graph: h = layer(h, coords)
        return self.head(h)


def pearson_loss(pred, target, eps=1e-8):
    pc = pred - pred.mean(dim=0, keepdim=True); tc = target - target.mean(dim=0, keepdim=True)
    r = (pc*tc).sum(0) / ((pc.square().sum(0)+eps).sqrt() * (tc.square().sum(0)+eps).sqrt() + eps)
    return -r.mean()

@torch.no_grad()
def evaluate(model, loader, device, gene_dim):
    model.eval(); mse_sum=mae_sum=n=0; pearsons=[]
    for imgs, genes, coords, _ in loader:
        imgs, genes, coords = imgs.to(device), genes.to(device), coords.to(device)
        preds = model(imgs, coords)
        mse_sum += F.mse_loss(preds, genes).item()*genes.size(0)
        mae_sum += F.l1_loss(preds, genes).item()*genes.size(0); n += genes.size(0)
        yt, yp = genes.cpu().numpy(), preds.cpu().numpy()
        for g in range(gene_dim):
            if np.std(yt[:,g]) > 0 and np.std(yp[:,g]) > 0: pearsons.append(np.corrcoef(yt[:,g], yp[:,g])[0,1])
    return mse_sum/n, mae_sum/n, float(np.nanmean(pearsons)) if pearsons else np.nan


def train_one(cfg: Config, model_name="img2st", use_snr_mix=False):
    seed_all(cfg.seed); device=torch.device(cfg.device)
    run_name = f"{model_name}_{'snr' if use_snr_mix else 'base'}"
    save_dir = os.path.join(cfg.save_root, run_name); loss_dir = os.path.join(cfg.loss_root, run_name)
    os.makedirs(save_dir, exist_ok=True); os.makedirs(loss_dir, exist_ok=True); logger=setup_logger(save_dir)

    images, genes, coords = load_hest_data(cfg); gene_dim = genes.shape[1]
    proc = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(), T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
    idx = np.arange(len(genes)); np.random.shuffle(idx)
    tr = int(len(idx)*cfg.train_split); va = int(len(idx)*cfg.val_split)
    train_idx, val_idx, test_idx = idx[:tr], idx[tr:tr+va], idx[tr+va:]

    if use_snr_mix:
        D = pairwise_distances(genes[train_idx], metric="euclidean").astype(np.float32)
        train_set = SNRMixCoordDataset([images[i] for i in train_idx], genes[train_idx], coords[train_idx], D, proc, cfg.alpha, k=cfg.k_neighbors, cache_cap=cfg.cache_cap)
    else:
        train_set = ImageGeneCoordDataset([images[i] for i in train_idx], genes[train_idx], coords[train_idx], proc, cfg.cache_cap)
    val_set = ImageGeneCoordDataset([images[i] for i in val_idx], genes[val_idx], coords[val_idx], proc)
    test_set = ImageGeneCoordDataset([images[i] for i in test_idx], genes[test_idx], coords[test_idx], proc)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    model = Img2STNetLite(gene_dim) if model_name.lower()=="img2st" else MagNetLite(gene_dim)
    model = model.to(device)
    opt = optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9,0.98), eps=1e-6, weight_decay=cfg.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=cfg.epochs*max(1,len(train_loader)), eta_min=1e-7)
    records=[]; best=float("inf")

    for epoch in range(1, cfg.epochs+1):
        model.train(); losses=[]; mses=[]; maes=[]
        for batch in train_loader:
            if use_snr_mix:
                xmix, ymix, cmix, xi, yi, ci, xj, yj, cj, lam, _, _, wij = batch
                xmix, ymix, cmix = xmix.to(device), ymix.to(device), cmix.to(device)
                xi, yi, ci = xi.to(device), yi.to(device), ci.to(device)
                xj, yj, cj = xj.to(device), yj.to(device), cj.to(device)
                lam = lam.to(device).view(-1,1); wij = wij.to(device).view(-1,1)
                X = torch.cat([xmix, xi, xj], 0); C = torch.cat([cmix, ci, cj], 0)
                Y = model(X, C); B=xmix.size(0)
                yhat_mix, yhat_i, yhat_j = Y[:B], Y[B:2*B], Y[2*B:]
                L_vic = F.mse_loss(yhat_mix, ymix)
                L_cons = F.mse_loss(yhat_mix, lam*yhat_i + (1-lam)*yhat_j)
                L_edge = (wij * ((yhat_i-yhat_j)-(yi-yj)).square()).mean()
                L_corr = pearson_loss(yhat_mix, ymix)
                loss = L_vic + cfg.lambda_cons*L_cons + cfg.lambda_edge*L_edge + cfg.lambda_corr*L_corr
                pred_for_log, target_for_log = yhat_mix, ymix
            else:
                imgs, gene, coord, _ = batch
                imgs, gene, coord = imgs.to(device), gene.to(device), coord.to(device)
                pred = model(imgs, coord)
                loss = F.mse_loss(pred, gene) + cfg.lambda_corr * pearson_loss(pred, gene)
                pred_for_log, target_for_log = pred, gene
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            losses.append(loss.item()); mses.append(F.mse_loss(pred_for_log, target_for_log).item()); maes.append(F.l1_loss(pred_for_log, target_for_log).item())
        vmse, vmae, vpcc = evaluate(model, val_loader, device, gene_dim)
        logger.info(f"Epoch {epoch:03d} | train_loss {np.mean(losses):.4f} | train_mse {np.mean(mses):.4f} | val_mse {vmse:.4f} | val_mae {vmae:.4f} | val_pcc {vpcc:.4f}")
        records.append(dict(epoch=epoch, train_loss=np.mean(losses), train_mse=np.mean(mses), train_mae=np.mean(maes), val_mse=vmse, val_mae=vmae, val_pcc=vpcc))
        if vmse < best:
            best=vmse; torch.save({"epoch":epoch,"model_state":model.state_dict(),"mse":vmse,"mae":vmae,"pearson":vpcc}, os.path.join(save_dir,"best_model.pt"))
    pd.DataFrame(records).to_csv(os.path.join(loss_dir,"metrics.csv"), index=False)
    ckpt=torch.load(os.path.join(save_dir,"best_model.pt"), map_location=device); model.load_state_dict(ckpt["model_state"])
    tmse,tmae,tpcc = evaluate(model, test_loader, device, gene_dim)
    with open(os.path.join(save_dir,"test_metrics.json"),"w") as f: json.dump({"test_mse":tmse,"test_mae":tmae,"test_pcc":tpcc,"best_val_mse":best}, f, indent=2)
    logger.info(f"[TEST] MSE {tmse:.4f} | MAE {tmae:.4f} | PCC {tpcc:.4f}")


if __name__ == "__main__":
    cfg = Config()
    # Suggested reviewer experiments:
    # train_one(cfg, "img2st", use_snr_mix=False)
    # train_one(cfg, "img2st", use_snr_mix=True)
    # train_one(cfg, "magnet", use_snr_mix=False)
    # train_one(cfg, "magnet", use_snr_mix=True)
    train_one(cfg, "magnet", use_snr_mix=True)