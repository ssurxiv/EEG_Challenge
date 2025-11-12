import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
from tqdm import tqdm

from model import CascadedEEGModel  # 네 모델 파일
# from braindecode.models import EEGNeX  # encoder 교체 시 참고

# =========================================
# 0) Config
# =========================================
GPU_ID         = 3
MAX_EPOCHS     = 100
PATIENCE       = 15
LR             = 1e-5
WEIGHT_DECAY   = 1e-5
BATCH_SIZE     = 1024
NUM_WORKERS    = 8
SEED           = 42

DATA_PREFIX    = 'ch2_add'
DATA_ROOT      = "/home/mip/gaia_nas_second/eeg_challenge/data_2s_epoch"
OUT_ROOT       = "/home/mip/disk4/sy/neuripseeg/out/ch2_pretrain"

# factor 정의
FACTOR_ORDER   = ['internalizing', 'attention', 'p-factor', 'age', 'sex', 'externalizing']
IDX            = {k: i for i, k in enumerate(FACTOR_ORDER)}
REG_KEYS       = ['internalizing', 'attention', 'p-factor', 'age']

# sign-direction 보조손실 하이퍼
W_AUX          = 0.5
W_MAIN         = 1.0
W_SIGN         = 1.0
W_DIR          = 1.0


# =========================================
# 1) Utils
# =========================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def device_select(gpu_id: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")

@torch.no_grad()
def robust_zscore(X: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # X: [B, C, T]
    med = torch.median(X, dim=-1, keepdim=True).values
    mad = torch.median(torch.abs(X - med), dim=-1, keepdim=True).values
    return (X - med) / (1.4826 * (mad + eps))

def to_sex01(arr_like):
    # 'F'→1, 그 외 0로 매핑. 데이터셋에 맞춰 필요 시 조정.
    return np.array([1 if (s == 'F' or s == 1 or s == '1' or s is True) else 0
                     for s in arr_like], dtype=np.float32)

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
def load_split_ch2(data_root: str, data_prefix: str):
    """
    R1–R3 → train, R4 → valid, R5 → test
    반환 텐서는 정규화(robust z) 적용 전 원시 텐서
    """
    Xs_train, Ys_train, Zs_train = [], [], []
    Xs_valid, Ys_valid, Zs_valid = [], [], []
    Xs_test,  Ys_test,  Zs_test  = [], [], []

    for i in [1, 2, 3, 4, 5]:
        p = os.path.join(data_root, f"R{i}", f"contrastChangeDetection_{data_prefix}.pt")
        data = torch.load(p, weights_only=False)
        X = data['X']  # [N, C, T]
        y = np.asarray(data['externalizing'], dtype=np.float32).reshape(-1)  # [N]

        # Z = 5열 (intl, attn, p-factor, age, sex)
        cols = []
        for k in FACTOR_ORDER[:-1]:  # exclude 'externalizing'
            if k == 'sex':
                cols.append(to_sex01(data['sex']))
            else:
                cols.append(np.asarray(data[k], dtype=np.float32))
        Z = np.stack(cols, axis=-1)  # [N, 5]

        if i == 5:
            Xs_test.append(X);  Ys_test.append(y);  Zs_test.append(Z)
        elif i == 4:
            Xs_valid.append(X); Ys_valid.append(y); Zs_valid.append(Z)
        else:
            Xs_train.append(X); Ys_train.append(y); Zs_train.append(Z)

    # 합치기
    Xs_train = np.concatenate(Xs_train, axis=0); Ys_train = np.concatenate(Ys_train, axis=0); Zs_train = np.concatenate(Zs_train, axis=0)
    Xs_valid = np.concatenate(Xs_valid, axis=0); Ys_valid = np.concatenate(Ys_valid, axis=0); Zs_valid = np.concatenate(Zs_valid, axis=0)
    Xs_test  = np.concatenate(Xs_test,  axis=0); Ys_test  = np.concatenate(Ys_test,  axis=0); Zs_test  = np.concatenate(Zs_test,  axis=0)

    # 텐서화
    Xs_train = torch.tensor(Xs_train, dtype=torch.float32)
    Ys_train = torch.tensor(Ys_train, dtype=torch.float32)
    Zs_train = torch.tensor(Zs_train, dtype=torch.float32)

    Xs_valid = torch.tensor(Xs_valid, dtype=torch.float32)
    Ys_valid = torch.tensor(Ys_valid, dtype=torch.float32)
    Zs_valid = torch.tensor(Zs_valid, dtype=torch.float32)

    Xs_test  = torch.tensor(Xs_test,  dtype=torch.float32)
    Ys_test  = torch.tensor(Ys_test,  dtype=torch.float32)
    Zs_test  = torch.tensor(Zs_test,  dtype=torch.float32)

    return (Xs_train, Ys_train, Zs_train), (Xs_valid, Ys_valid, Zs_valid), (Xs_test, Ys_test, Zs_test)


# =========================================
# 3) Model
# =========================================
def build_model(device: torch.device) -> nn.Module:
    model = CascadedEEGModel(
        n_channels=129, n_times=200, d_eeg=32,
        h_factor=32, h_ext=32, dropout=0.0,
        stop_grad_factors=False,
        use_residual_fusion=True,
        use_layernorm_on_z=True,
        ext_nonneg=False,
        feed_sex_prob_to_ext=True
    ).to(device)
    return model


# =========================================
# 4) Train one epoch
# =========================================
class JointLoss:
    """4 regression + sex(bce) + externalizing + sign + direction"""
    def __init__(self, device, pos_weight: float):
        self.reg_loss = nn.HuberLoss(delta=1.0, reduction="mean")
        self.bce_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=device)
        )

    def __call__(self, out: dict, y_ext: torch.Tensor, Z: torch.Tensor,
                 ext_mean: torch.Tensor, ext_std: torch.Tensor):
        """
        out: model(X)의 dict 출력
        y_ext: [B]
        Z:     [B, 5] (intl, attn, pfac, age, sex)
        """
        B = y_ext.size(0)
        # 4 regression
        reg_pred = torch.cat([out[k] for k in REG_KEYS], dim=-1).view(B, 4)
        reg_gt   = torch.stack([Z[:, IDX[k]] for k in REG_KEYS], dim=-1)
        loss_reg = self.reg_loss(reg_pred, reg_gt)

        # sex
        sex_logit = out['sex_logit'].view(-1)
        sex_gt    = Z[:, IDX['sex']]
        loss_bce  = self.bce_loss(sex_logit, sex_gt)

        # externalizing
        ext_pred  = out['externalizing'].view(-1)
        loss_ext  = self.reg_loss(ext_pred, y_ext)

        # sign-direction
        sign_temp  = ext_std + 1e-8
        dir_margin = 0.10 * ext_std
        amb_tau    = 0.15 * ext_std

        y_c      = y_ext - ext_mean
        y_pred_c = ext_pred - ext_mean
        sign_lbl = (y_c > 0).float()
        amb_mask = (y_c.abs() > amb_tau).float()

        loss_sign_vec = F.binary_cross_entropy_with_logits(
            y_pred_c / sign_temp, sign_lbl, reduction='none'
        )
        denom = amb_mask.sum().clamp_min(1.0)
        loss_sign = (loss_sign_vec * amb_mask).sum() / denom

        sgn = torch.sign(y_c)
        hinge_vec = torch.relu(dir_margin - sgn * y_pred_c)
        loss_dir = (hinge_vec * amb_mask).sum() / denom

        total = (W_AUX * (loss_reg + loss_bce)
                 + W_MAIN * loss_ext
                 + W_SIGN * loss_sign
                 + W_DIR * loss_dir)

        logs = dict(total=total.item(), reg=loss_reg.item(), sex=loss_bce.item(),
                    ext=loss_ext.item(), sgn=loss_sign.item(), dir=loss_dir.item())
        return total, logs


def train_one_epoch(model, loader, optimizer, loss_fn, device, ext_mean, ext_std, grad_clip=1.0):
    model.train()
    n_train = 0
    agg = dict(total=0.0, reg=0.0, sex=0.0, ext=0.0, sgn=0.0, dir=0.0)

    for X, y_ext, Z in loader:
        X     = X.to(device=device, dtype=torch.float32)
        y_ext = y_ext.to(device=device, dtype=torch.float32).view(-1)
        Z     = Z.to(device=device, dtype=torch.float32)
        B = X.size(0)

        optimizer.zero_grad()
        out = model(robust_zscore(X), return_latent=False, detach_factors=False)
        loss, logs = loss_fn(out, y_ext, Z, ext_mean, ext_std)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        n_train += B
        for k in agg:
            agg[k] += logs[k] * B

    for k in agg:
        agg[k] /= max(1, n_train)
    return agg


# =========================================
# 5) Evaluate (valid or test)
# =========================================
@torch.no_grad()
def evaluate(model, loader, loss_fn, device, ext_mean, ext_std):
    model.eval()
    n_val = 0
    agg = dict(total=0.0, reg=0.0, sex=0.0, ext=0.0, sgn=0.0, dir=0.0)

    reg_preds_all, reg_gts_all = [], []
    sex_logits_all, sex_gts_all = [], []
    ext_preds_all, ext_gts_all  = [], []

    for X, y_ext, Z in loader:
        X     = X.to(device=device, dtype=torch.float32)
        y_ext = y_ext.to(device=device, dtype=torch.float32).view(-1)
        Z     = Z.to(device=device, dtype=torch.float32)
        B = X.size(0)

        out = model(robust_zscore(X), return_latent=False, detach_factors=False)
        loss, logs = loss_fn(out, y_ext, Z, ext_mean, ext_std)

        n_val += B
        for k in agg:
            agg[k] += logs[k] * B

        reg_preds_all.append(torch.cat([out[k] for k in REG_KEYS], dim=-1).cpu())
        reg_gts_all.append(torch.stack([Z[:, IDX[k]] for k in REG_KEYS], dim=-1).cpu())
        sex_logits_all.append(out['sex_logit'].view(-1).cpu())
        sex_gts_all.append(Z[:, IDX['sex']].cpu())
        ext_preds_all.append(out['externalizing'].view(-1).cpu())
        ext_gts_all.append(y_ext.cpu())

    for k in agg:
        agg[k] /= max(1, n_val)

    # metrics
    reg_preds = torch.cat(reg_preds_all, dim=0).numpy()
    reg_gts   = torch.cat(reg_gts_all,   dim=0).numpy()
    rmse_each  = np.sqrt(np.mean((reg_preds - reg_gts) ** 2, axis=0))
    nrmse_each = rmse_each / (np.std(reg_gts, axis=0) + 1e-8)
    nrmse_mean = float(nrmse_each.mean())

    sex_logits = torch.cat(sex_logits_all, dim=0)
    sex_gts    = torch.cat(sex_gts_all,    dim=0)
    sex_pred   = (torch.sigmoid(sex_logits) > 0.5).float()
    sex_acc    = float((sex_pred == sex_gts).float().mean().item())

    ext_preds = torch.cat(ext_preds_all, dim=0).numpy()
    ext_gts   = torch.cat(ext_gts_all,   dim=0).numpy()
    mse_ext = float(np.mean((ext_preds - ext_gts) ** 2))
    rmse_ext = float(np.sqrt(mse_ext))
    nrmse_ext = rmse_ext / (np.std(ext_gts) + 1e-8)

    metrics = dict(
        nrmse_each=nrmse_each, nrmse_mean=nrmse_mean,
        sex_acc=sex_acc, nrmse_ext=nrmse_ext,
        ext_preds_head=ext_preds[:20], ext_gts_head=ext_gts[:20],
    )
    return agg, metrics


# =========================================
# 6) Main
# =========================================
def main():
    # Run id & out dir
    now = datetime.now()
    run_id = now.strftime("%y%m%d%H%M%S")
    out_dir = os.path.join(OUT_ROOT, run_id)
    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, f"weights_joint_best_{run_id}.pt")
    last_path = os.path.join(out_dir, f"weights_joint_last_{run_id}.pt")

    print("ID:", run_id)

    # seed & device
    set_seed(SEED)
    device = device_select(GPU_ID)
    print("Device:", device)

    # data
    (Xs_train, Ys_train, Zs_train), (Xs_valid, Ys_valid, Zs_valid), (Xs_test, Zs_test_y, Zs_test) = load_split_ch2(DATA_ROOT, DATA_PREFIX)
    print("Train shapes:", Xs_train.shape, Ys_train.shape, Zs_train.shape)
    print("Valid shapes:", Xs_valid.shape, Ys_valid.shape, Zs_valid.shape)
    print("Test  shapes:", Xs_test.shape,  Zs_test_y.shape,  Zs_test.shape)

    # robust z-score는 배치에서 적용하므로 여기서는 텐서만 묶어 DataLoader 생성
    train_ds = TensorDataset(Xs_train, Ys_train, Zs_train)
    valid_ds = TensorDataset(Xs_valid, Ys_valid, Zs_valid)
    test_ds  = TensorDataset(Xs_test,  Zs_test_y, Zs_test)
    train_loader, valid_loader, test_loader = make_loaders(train_ds, valid_ds, test_ds)

    # 통계 (sign/dir용)
    ext_mean = torch.tensor(Ys_train.mean().item(), dtype=torch.float32).to(device)
    ext_std  = torch.tensor(Ys_train.std(unbiased=True).item(), dtype=torch.float32).clamp_min(1e-6).to(device)

    # pos_weight 계산
    pos_ratio = float(Zs_train[:, IDX['sex']].mean().item())
    neg_ratio = 1.0 - pos_ratio
    pos_weight = (neg_ratio / max(pos_ratio, 1e-6))

    # model & opt & loss
    model = build_model(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = JointLoss(device, pos_weight=pos_weight)

    # train loop
    best_val = float("inf")
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in tqdm(range(1, MAX_EPOCHS + 1), desc="[JOINT]"):
        tr_logs = train_one_epoch(model, train_loader, optimizer, loss_fn, device, ext_mean, ext_std)
        va_logs, va_metrics = evaluate(model, valid_loader, loss_fn, device, ext_mean, ext_std)

        # 로깅
        print(
            f"[JOINT][e{epoch:03d}] \n"
            f"Train | total {tr_logs['total']:.5f} | reg {tr_logs['reg']:.5f} | sex {tr_logs['sex']:.5f} | ext {tr_logs['ext']:.5f} | sgn {tr_logs['sgn']:.5f} | dir {tr_logs['dir']:.5f} |\n "
            f"Valid | total {va_logs['total']:.5f} | reg {va_logs['reg']:.5f} | sex {va_logs['sex']:.5f} | ext {va_logs['ext']:.5f} | sgn {va_logs['sgn']:.5f} | dir {va_logs['dir']:.5f} | "
            f"NRMSE4 mean {va_metrics['nrmse_mean']:.4f} | "
            f"NRMSE[intl {va_metrics['nrmse_each'][0]:.3f}, attn {va_metrics['nrmse_each'][1]:.3f}, "
            f"pfac {va_metrics['nrmse_each'][2]:.3f}, age {va_metrics['nrmse_each'][3]:.3f}] | "
            f"sex_acc {va_metrics['sex_acc']:.3f} | ext_NRMSE {va_metrics['nrmse_ext']:.4f}"
        )

        # Early stopping: externalizing NRMSE 기준
        key_val = va_metrics['nrmse_ext']
        if key_val < best_val - 1e-6:
            best_val = key_val
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_path)
            print(f"[JOINT] New best ext_NRMSE {best_val:.6f} @ epoch {best_epoch} -> {best_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"[JOINT] Early stopping at epoch {epoch}")
                break

    # 마지막 저장
    torch.save(model.state_dict(), last_path)
    print(f"[JOINT] Final model saved: {last_path}")

    # 베스트 로드
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"[JOINT] Loaded best(e{best_epoch}): {best_path}")

    # 테스트
    _, te_metrics = evaluate(model, test_loader, loss_fn, device, ext_mean, ext_std)
    print(
        f"[JOINT][TEST-{run_id}]"
        f"NRMSE4 mean {te_metrics['nrmse_mean']:.4f} | "
        f"NRMSE[intl {te_metrics['nrmse_each'][0]:.3f}, attn {te_metrics['nrmse_each'][1]:.3f}, "
        f"pfac {te_metrics['nrmse_each'][2]:.3f}, age {te_metrics['nrmse_each'][3]:.3f}] | "
        f"sex_acc {te_metrics['sex_acc']:.3f} | ext_NRMSE {te_metrics['nrmse_ext']:.4f}"
    )
    print(te_metrics['ext_preds_head'])
    print(te_metrics['ext_gts_head'])


if __name__ == "__main__":
    main()
