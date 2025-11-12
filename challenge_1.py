import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
from tqdm import tqdm
from braindecode.models import EEGNeX

from dir import (
    make_descending_soft_labels,
    dir_loss,
    MultiExpertDIR,
    compute_quantile_edges,
    make_group_id_quantile,
)

# =========================================
# 0) Config
# =========================================
GPU_ID         = 3
MAX_EPOCHS     = 100
PATIENCE       = 15
LR             = 5e-3
WEIGHT_DECAY   = 1e-4
BATCH_SIZE     = 256
NUM_WORKERS    = 8
SEED           = 100

NUM_GROUPS     = 10
BETA_SOFT      = 1.0
LAMBDA_SOFT    = 0.5
LAMBDA_MSE     = 1.0
TEACHER_FORCE  = True

DATA_ROOT      = "/home/mip/gaia_nas_second/eeg_challenge/data_2s_epoch"
OUT_ROOT       = "/home/mip/disk4/sy/neuripseeg/out/ch1_new"


# =========================================
# 1) Utils
# =========================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def device_select(gpu_id: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")

@torch.no_grad()
def robust_zscore(X: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # X: [B, C, T]
    med = torch.median(X, dim=-1, keepdim=True).values        # [B, C, 1]
    mad = torch.median(torch.abs(X - med), dim=-1, keepdim=True).values
    return (X - med) / (1.4826 * (mad + eps))

class BalancedSAM:
    def __init__(self, base_optimizer, rho=0.05, beta=1.0):
        self.opt = base_optimizer
        self.rho = rho
        self.beta = beta
        self._e = None

    @torch.no_grad()
    def _grad_norm(self):
        gn2 = None
        for group in self.opt.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                val = p.grad.pow(2).sum()
                gn2 = val if gn2 is None else gn2 + val
        if gn2 is None:
            return 1.0
        return gn2.sqrt().item() + 1e-12

    @torch.no_grad()
    def first_step(self, scale: float):
        grad_norm = self._grad_norm()
        eps_scale = self.rho * scale / grad_norm
        self._e = []
        for group in self.opt.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    self._e.append(None)
                    continue
                e = p.grad * eps_scale
                p.add_(e)
                self._e.append(e)

    @torch.no_grad()
    def second_step(self):
        idx = 0
        for group in self.opt.param_groups:
            for p in group["params"]:
                e = self._e[idx]; idx += 1
                if e is not None:
                    p.sub_(e)
        self.opt.step()
        self.opt.zero_grad()

def make_loaders(train_t, valid_t, test_t, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, seed=SEED):
    def _worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    train_loader = DataLoader(train_t, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, worker_init_fn=_worker_init_fn)
    valid_loader = DataLoader(valid_t, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, worker_init_fn=_worker_init_fn)
    test_loader  = DataLoader(test_t,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, worker_init_fn=_worker_init_fn)
    return train_loader, valid_loader, test_loader


# =========================================
# 2) Data
# =========================================
def load_split(data_root: str):
    Xs_train, Ys_train = [], []
    Xs_valid, Ys_valid = [], []
    Xs_test,  Ys_test  = [], []

    for i in range(1, 12):
        p = os.path.join(data_root, f"R{i}", "contrastChangeDetection_ch1.pt")
        data = torch.load(p, weights_only=False)
        X = data["X"]
        y = data["response_time"]

        if i == 11:
            Xs_test.append(X);  Ys_test.append(y)
        elif i == 10:
            Xs_valid.append(X); Ys_valid.append(y)
        else:
            Xs_train.append(X); Ys_train.append(y)

    Xs_train = np.concatenate(Xs_train, axis=0)
    Ys_train = np.concatenate(Ys_train, axis=0)
    Xs_valid = np.concatenate(Xs_valid, axis=0)
    Ys_valid = np.concatenate(Ys_valid, axis=0)
    Xs_test  = np.concatenate(Xs_test,  axis=0)
    Ys_test  = np.concatenate(Ys_test,  axis=0)

    # to tensor
    Xs_train = torch.tensor(Xs_train, dtype=torch.float32)
    Ys_train = torch.tensor(Ys_train, dtype=torch.float32)
    Xs_valid = torch.tensor(Xs_valid, dtype=torch.float32)
    Ys_valid = torch.tensor(Ys_valid, dtype=torch.float32)
    Xs_test  = torch.tensor(Xs_test,  dtype=torch.float32)
    Ys_test  = torch.tensor(Ys_test,  dtype=torch.float32)

    return (Xs_train, Ys_train), (Xs_valid, Ys_valid), (Xs_test, Ys_test)


# =========================================
# 3) Model
# =========================================
class EEG_DIR_Model(nn.Module):
    def __init__(self, encoder: nn.Module, head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, X):
        z = self.encoder(X)  # [B, 32]
        logits, p_g, y_hat_all = self.head(z, return_all=True)
        return logits, p_g, y_hat_all

def build_model(device: torch.device, num_groups: int) -> nn.Module:
    encoder = EEGNeX(n_chans=129, n_times=200, n_outputs=32).to(device)
    head = MultiExpertDIR(d_embed=32, num_groups=num_groups, hidden=32).to(device)
    return EEG_DIR_Model(encoder, head).to(device)


# =========================================
# 4) Train one epoch
# =========================================
def train_one_epoch(model, train_loader, optimizer, bsam, quantile_edges, w_group, device):
    model.train()
    running, sum_mse, sum_soft = 0.0, 0.0, 0.0

    for X, y in train_loader:
        optimizer.zero_grad()
        X = X.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.float32).unsqueeze(1)

        logits, p_g, y_hat_all = model(robust_zscore(X))
        with torch.no_grad():
            g_true = make_group_id_quantile(y, quantile_edges)
            q_soft = make_descending_soft_labels(g_true, NUM_GROUPS, beta=BETA_SOFT)
        sw = w_group[g_true]

        loss, logs = dir_loss(
            logits, y_hat_all, y_true=y,
            g_true=g_true, q_soft=q_soft,
            lambda_soft=LAMBDA_SOFT, lambda_mse=LAMBDA_MSE,
            teacher_force=TEACHER_FORCE, sample_weight=sw
        )
        loss.backward()

        # BSAM 1st
        batch_scale = (sw.mean().item()) ** bsam.beta
        bsam.first_step(scale=batch_scale)

        # BSAM 2nd forward
        logits, p_g, y_hat_all = model(robust_zscore(X))
        loss_pert, logs_pert = dir_loss(
            logits, y_hat_all, y_true=y,
            g_true=g_true, q_soft=q_soft,
            lambda_soft=LAMBDA_SOFT, lambda_mse=LAMBDA_MSE,
            teacher_force=TEACHER_FORCE, sample_weight=sw
        )
        loss_pert.backward()
        bsam.second_step()

        running += float(loss_pert.item())
        sum_mse += float(logs_pert["L_mse"])
        sum_soft += float(logs_pert["L_soft"])

    n = max(1, len(train_loader))
    return running / n, sum_mse / n, sum_soft / n


# =========================================
# 5) Evaluate (valid or test)  — inference + metrics
# =========================================
@torch.no_grad()
def evaluate(model, loader, quantile_edges, w_group, device):
    model.eval()
    total, sum_mse, sum_soft = 0.0, 0.0, 0.0
    preds, gts = [], []

    for X, y in loader:
        X = X.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.float32).unsqueeze(1)

        logits, p_g, y_hat_all = model(robust_zscore(X))
        g_true = make_group_id_quantile(y, quantile_edges)
        q_soft = make_descending_soft_labels(g_true, NUM_GROUPS, beta=BETA_SOFT)
        sw = w_group[g_true]

        loss, logs = dir_loss(
            logits, y_hat_all, y_true=y,
            g_true=g_true, q_soft=q_soft,
            lambda_soft=LAMBDA_SOFT, lambda_mse=LAMBDA_MSE,
            teacher_force=TEACHER_FORCE, sample_weight=sw
        )
        total += float(loss.item())
        sum_mse += float(logs["L_mse"])
        sum_soft += float(logs["L_soft"])

        y_pred = (p_g * y_hat_all).sum(dim=1, keepdim=True)  # [B,1]
        preds.append(y_pred.cpu())
        gts.append(y.cpu())

    n = max(1, len(loader))
    avg_total = total / n
    avg_mse   = sum_mse / n
    avg_soft  = sum_soft / n

    preds = np.concatenate([t.numpy() for t in preds], axis=0).reshape(-1)
    gts   = np.concatenate([t.numpy() for t in gts],   axis=0).reshape(-1)
    mse   = float(np.mean((preds - gts) ** 2))
    rmse  = float(np.sqrt(mse))
    nrmse = float(rmse / (gts.std() + 1e-8))

    return {
        "avg_total": avg_total,
        "avg_mse": avg_mse,
        "avg_soft": avg_soft,
        "mse": mse,
        "rmse": rmse,
        "nrmse": nrmse,
    }


# =========================================
# 6) Main
# =========================================
def main():
    # Run id and out dir
    now = datetime.now()
    run_id = now.strftime("%y%m%d%H%M%S")
    out_dir = os.path.join(OUT_ROOT, run_id)
    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, f"weights_challenge_1_best_{run_id}.pt")
    last_path = os.path.join(out_dir, f"weights_challenge_1_last_{run_id}.pt")

    # seed & device
    set_seed(SEED)
    device = device_select(GPU_ID)
    print(f"Device -> {device}")

    # data
    (Xs_train, Ys_train), (Xs_valid, Ys_valid), (Xs_test, Ys_test) = load_split(DATA_ROOT)
    print("Train", Xs_train.shape, Ys_train.shape)
    print("Valid", Xs_valid.shape, Ys_valid.shape)
    print("Test ", Xs_test.shape, Ys_test.shape)

    # stat for grouping
    quantile_edges = compute_quantile_edges(Ys_train, NUM_GROUPS)
    np.save(os.path.join(out_dir, "quantile_edges.npy"), quantile_edges)
    with torch.no_grad():
        g_train = make_group_id_quantile(Ys_train, quantile_edges)
        counts = np.bincount(g_train.cpu().numpy(), minlength=NUM_GROUPS).astype(np.float64)
        inv = 1.0 / np.maximum(counts, 1.0)
        w_group = (inv / inv.mean()).astype(np.float32)
    w_group = torch.tensor(w_group, device=device)  # [G]

    # datasets & loaders
    train_ds = TensorDataset(Xs_train, Ys_train)
    valid_ds = TensorDataset(Xs_valid, Ys_valid)
    test_ds  = TensorDataset(Xs_test,  Ys_test)
    train_loader, valid_loader, test_loader = make_loaders(train_ds, valid_ds, test_ds)

    # model & opt
    model = build_model(device, NUM_GROUPS)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    bsam = BalancedSAM(optimizer, rho=0.05, beta=1.0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS - 1)

    # train loop
    best_valid = float("inf")
    no_improve = 0
    for epoch in tqdm(range(1, MAX_EPOCHS + 1), desc="Train"):
        tr_total, tr_mse, tr_soft = train_one_epoch(
            model, train_loader, optimizer, bsam, quantile_edges, w_group, device
        )
        if scheduler is not None:
            scheduler.step()

        val_stats = evaluate(model, valid_loader, quantile_edges, w_group, device)
        print(
            f"[e{epoch:03d}] "
            f"train total {tr_total:.5f} | mse {tr_mse:.5f} | soft {tr_soft:.5f}  ||  "
            f"valid total {val_stats['avg_total']:.5f} | mse {val_stats['avg_mse']:.5f} | soft {val_stats['avg_soft']:.5f} "
            f"| R4 mse {val_stats['mse']:.5f} rmse {val_stats['rmse']:.5f} nrmse {val_stats['nrmse']:.5f}"
        )

        # early stopping 기준은 valid total
        if val_stats["avg_total"] < best_valid - 1e-6:
            best_valid = val_stats["avg_total"]
            no_improve = 0
            torch.save(model.state_dict(), best_path)
            print(f"New best {best_valid:.6f} -> {best_path}")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    # save last
    torch.save(model.state_dict(), last_path)
    print(f"Saved last to {last_path}")

    # load best for test
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"Loaded best: {best_path}")

    test_stats = evaluate(model, test_loader, quantile_edges, w_group, device)
    print(
        f"[TEST-{run_id}] "
        f"R4 mse {test_stats['mse']:.6f} rmse {test_stats['rmse']:.6f} nrmse {test_stats['nrmse']:.6f}"
    )

if __name__ == "__main__":
    main()
