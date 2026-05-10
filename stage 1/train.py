"""
Обучение LSTM-наукастера.

Стратегия:
  - 5-fold CV по event_id (group-aware) внутри SC23+24.
  - Финальная модель — на всём SC23+24, оценка на SC25 — в evaluate.py.
  - Early stopping по валидационному total loss.
  - Чекпоинты: stage 1/results/checkpoints/{tag}/fold{k}.pt
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from dataset import (SPEDataset, PriorNormalizer, collate_fn, event_kfold,
                     load_index, K_MIN, N_MAX)
from model   import LSTMNowcaster, nowcaster_loss

CHECKPOINTS_DIR = ROOT / "results" / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Конфигурация по умолчанию ────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "hidden_size":  64,
    "num_layers":   2,
    "dropout":      0.0,
    "lam_peak":     1.0,
    "lr":           1e-3,
    "weight_decay": 0.0,
    "batch_size":   32,
    "max_epochs":   200,
    "patience":     25,
    "k_min":        K_MIN,
    "n_max":        N_MAX,
    "seed":         42,
}


# ── Тренировочные процедуры ──────────────────────────────────────────────────

def run_epoch(model, loader, opt, device, lam_peak: float, train: bool):
    model.train(train)
    sums = {"loss": 0.0, "L_profile": 0.0, "L_peak": 0.0, "n": 0}
    for batch in loader:
        enc      = batch["encoder_input"].to(device)
        enc_mask = batch["encoder_mask"].to(device)
        tgt      = batch["target"].to(device)
        tgt_mask = batch["target_mask"].to(device)
        prior    = batch["prior"].to(device)
        log_jm   = batch["log_J_max"].to(device)

        if train:
            opt.zero_grad()
        with torch.set_grad_enabled(train):
            preds = model(enc, enc_mask, prior)
            loss, parts = nowcaster_loss(preds, tgt, tgt_mask, log_jm, lam_peak)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                opt.step()

        bsz = enc.size(0)
        sums["loss"]      += float(loss.item()) * bsz
        sums["L_profile"] += parts["L_profile"] * bsz
        sums["L_peak"]    += parts["L_peak"]    * bsz
        sums["n"]         += bsz

    n = max(sums["n"], 1)
    return {k: (v / n if k != "n" else v) for k, v in sums.items()}


def train_fold(model, train_loader, val_loader, cfg: dict, device,
               ckpt_path: Path, log_path: Path) -> dict:
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                           weight_decay=cfg["weight_decay"])

    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0
    history = []

    for ep in range(1, cfg["max_epochs"] + 1):
        tr = run_epoch(model, train_loader, opt, device, cfg["lam_peak"], train=True)
        va = run_epoch(model, val_loader,   opt, device, cfg["lam_peak"], train=False)

        rec = {"epoch": ep, "train": tr, "val": va}
        history.append(rec)
        print(f"  ep {ep:3d}  train={tr['loss']:.4f} (prof={tr['L_profile']:.3f}, peak={tr['L_peak']:.3f})  "
              f"val={va['loss']:.4f}  best={best_val:.4f}@{best_epoch}")

        if va["loss"] < best_val - 1e-5:
            best_val = va["loss"]
            best_epoch = ep
            bad_epochs = 0
            torch.save({"state_dict": model.state_dict(),
                        "epoch": ep, "val_loss": best_val, "config": cfg},
                       ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["patience"]:
                print(f"  early stopping at epoch {ep}, best @ {best_epoch}")
                break

    log_path.write_text(json.dumps(history, indent=2))
    return {"best_val": best_val, "best_epoch": best_epoch, "epochs_run": len(history)}


def make_loaders(idx_df, train_eids: list[str], val_eids: list[str],
                 normalizer: PriorNormalizer, cfg: dict, num_workers: int = 0):
    train_subset = idx_df[idx_df["event_id"].isin(train_eids)].reset_index(drop=True)
    val_subset   = idx_df[idx_df["event_id"].isin(val_eids)].reset_index(drop=True)

    ds_tr = SPEDataset(train_subset, normalizer, k_min=cfg["k_min"], n_max=cfg["n_max"])
    ds_va = SPEDataset(val_subset,   normalizer, k_min=cfg["k_min"], n_max=cfg["n_max"])

    print(f"    train: {len(ds_tr.profiles)} событий → {len(ds_tr)} примеров")
    print(f"    val:   {len(ds_va.profiles)} событий → {len(ds_va)} примеров")

    dl_tr = DataLoader(ds_tr, batch_size=cfg["batch_size"], shuffle=True,
                       collate_fn=collate_fn, num_workers=num_workers)
    dl_va = DataLoader(ds_va, batch_size=cfg["batch_size"], shuffle=False,
                       collate_fn=collate_fn, num_workers=num_workers)
    return dl_tr, dl_va


def train_cv(tag: str = "default", cfg: dict | None = None,
             n_splits: int = 5, only_fold: int | None = None) -> None:
    cfg = {**DEFAULT_CONFIG, **(cfg or {})}
    torch.manual_seed(cfg["seed"])

    idx = load_index()
    idx = idx.dropna(subset=["J_max_prior", "T_delta_prior"]).reset_index(drop=True)
    print(f"Доступно событий с prior: {len(idx)}")

    normalizer = PriorNormalizer().fit(idx)
    folds = event_kfold(idx, n_splits=n_splits)

    out_dir = CHECKPOINTS_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    summary = []
    for fold, (tr_eids, va_eids) in enumerate(folds):
        if only_fold is not None and fold != only_fold:
            continue
        print(f"\n=== Fold {fold} ===")
        dl_tr, dl_va = make_loaders(idx, tr_eids, va_eids, normalizer, cfg)

        model = LSTMNowcaster(
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
            n_max=cfg["n_max"],
        ).to(device)

        info = train_fold(model, dl_tr, dl_va, cfg, device,
                          ckpt_path=out_dir / f"fold{fold}.pt",
                          log_path=out_dir / f"fold{fold}_history.json")
        info["fold"] = fold
        summary.append(info)

    (out_dir / "cv_summary.json").write_text(json.dumps(summary, indent=2))
    if summary:
        avg = sum(s["best_val"] for s in summary) / len(summary)
        print(f"\nAverage best_val across folds: {avg:.4f}")


def train_final(tag: str = "final", cfg: dict | None = None) -> None:
    """Обучение на всём SC23+24, с холд-аутом ~10% событий для early stopping."""
    cfg = {**DEFAULT_CONFIG, **(cfg or {})}
    torch.manual_seed(cfg["seed"])

    idx = load_index()
    idx = idx.dropna(subset=["J_max_prior", "T_delta_prior"]).reset_index(drop=True)
    train = idx[idx["split"] == "train"].copy()

    # 90/10 hold-out по event_id (только для early stopping)
    rng = torch.Generator().manual_seed(cfg["seed"])
    perm = torch.randperm(len(train), generator=rng).tolist()
    n_val = max(int(0.1 * len(train)), 5)
    val_eids = train.iloc[perm[:n_val]]["event_id"].tolist()
    tr_eids  = train.iloc[perm[n_val:]]["event_id"].tolist()

    normalizer = PriorNormalizer().fit(idx)
    out_dir = CHECKPOINTS_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dl_tr, dl_va = make_loaders(idx, tr_eids, val_eids, normalizer, cfg)
    model = LSTMNowcaster(
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        n_max=cfg["n_max"],
    ).to(device)

    train_fold(model, dl_tr, dl_va, cfg, device,
               ckpt_path=out_dir / "final.pt",
               log_path=out_dir / "final_history.json")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "cv"
    if cmd == "cv":
        only = int(sys.argv[2]) if len(sys.argv) > 2 else None
        train_cv(only_fold=only)
    elif cmd == "final":
        train_final()
    else:
        print("Использование: python train.py [cv [fold]|final]")
