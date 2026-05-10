"""
PyTorch Dataset со скользящим окном для LSTM-наукастинга.

Из одного профиля длиной L (≥ k_min) генерируется (L − k_min) примеров:
  encoder видит шаги [0, k), target — шаги [k, L), для k в [k_min, L).

Все примеры одного события маркируются `event_id` — чтобы в k-fold
все они попадали в один фолд (нет утечки внутри события).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ROOT         = Path(__file__).parent
PROJECT_ROOT = ROOT.parent
PROFILE_DIR  = ROOT / "results" / "profiles"
INDEX_PATH   = ROOT / "results" / "profiles_index.parquet"
PRIOR_PATH   = ROOT / "results" / "prior_oof.parquet"

K_MIN = 6     # минимум 30 минут наблюдений до прогноза
N_MAX = 250   # максимум 21 ч (5-мин шаги) на профиль


# ── Нормализация prior ───────────────────────────────────────────────────────

class PriorNormalizer:
    """Простая стандартизация J_max_prior (в log10) и T_delta_prior (в часах)."""

    def __init__(self):
        self.j_mean = 0.0
        self.j_std  = 1.0
        self.t_mean = 0.0
        self.t_std  = 1.0

    def fit(self, df: pd.DataFrame) -> "PriorNormalizer":
        train = df[df["split"] == "train"]
        if "J_max_prior" in train.columns and train["J_max_prior"].notna().any():
            self.j_mean = float(train["J_max_prior"].mean())
            self.j_std  = float(train["J_max_prior"].std()) or 1.0
        if "T_delta_prior" in train.columns and train["T_delta_prior"].notna().any():
            self.t_mean = float(train["T_delta_prior"].mean())
            self.t_std  = float(train["T_delta_prior"].std()) or 1.0
        return self

    def transform(self, j_prior: float, t_prior: float) -> tuple[float, float]:
        return ((j_prior - self.j_mean) / self.j_std,
                (t_prior - self.t_mean) / self.t_std)


# ── Загрузка профилей и метаданных ───────────────────────────────────────────

def load_profile_array(event_id: str) -> np.ndarray:
    """Возвращает массив [L, 2] — log_J и delta_log_J."""
    df = pd.read_parquet(PROFILE_DIR / f"{event_id}.parquet")
    return df[["log_J", "delta_log_J"]].to_numpy().astype(np.float32)


def load_index() -> pd.DataFrame:
    """Сводный индекс: только успешно построенные профили + prior."""
    idx = pd.read_parquet(INDEX_PATH)
    idx = idx[idx["status"] == "ok"].reset_index(drop=True)
    if PRIOR_PATH.exists():
        prior = pd.read_parquet(PRIOR_PATH)
        idx = idx.merge(prior[["event_id", "J_max_prior", "T_delta_prior",
                               "J_max_real", "T_delta_real"]],
                        on="event_id", how="left")
    return idx


# ── Dataset ──────────────────────────────────────────────────────────────────

class SPEDataset(Dataset):
    """Скользящее окно по профилям СПС.

    Параметры
    ---------
    index_df : pd.DataFrame  (subset из load_index)
    normalizer : PriorNormalizer (fit на train)
    k_min : int — минимум шагов в энкодере
    n_max : int — обрезка длины профиля сверху
    """

    def __init__(self, index_df: pd.DataFrame, normalizer: PriorNormalizer,
                 k_min: int = K_MIN, n_max: int = N_MAX):
        self.normalizer = normalizer
        self.k_min = k_min
        self.n_max = n_max

        self.profiles: dict[str, np.ndarray] = {}
        self.examples: list[tuple[str, int]] = []  # (event_id, k)

        for _, row in index_df.iterrows():
            eid = row["event_id"]
            arr = load_profile_array(eid)
            if len(arr) > self.n_max:
                arr = arr[: self.n_max]
            L = len(arr)
            if L <= k_min:
                continue
            self.profiles[eid] = arr

            j_p = row.get("J_max_prior", np.nan)
            t_p = row.get("T_delta_prior", np.nan)
            if pd.isna(j_p) or pd.isna(t_p):
                continue
            j_norm, t_norm = self.normalizer.transform(float(j_p), float(t_p))

            self.metadata = getattr(self, "metadata", {})
            self.metadata[eid] = {
                "prior":     np.array([j_norm, t_norm], dtype=np.float32),
                "group":     row["group"],
                "split":     row["split"],
                "log_J_max": float(row["log_J_real"]),
            }
            for k in range(k_min, L):
                self.examples.append((eid, k))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        eid, k = self.examples[idx]
        arr   = self.profiles[eid]
        meta  = self.metadata[eid]

        encoder_input = torch.from_numpy(arr[:k])                # [k, 2]
        target        = torch.from_numpy(arr[k:, 0])             # [L-k]
        prior         = torch.from_numpy(meta["prior"])          # [2]

        return {
            "encoder_input": encoder_input,
            "target":        target,
            "prior":         prior,
            "log_J_max":     torch.tensor(meta["log_J_max"], dtype=torch.float32),
            "event_id":      eid,
            "group":         meta["group"],
            "k":             k,
        }


# ── collate_fn: padding до max в batch ───────────────────────────────────────

def collate_fn(batch: list[dict]) -> dict:
    B = len(batch)
    max_k    = max(item["encoder_input"].shape[0] for item in batch)
    max_tgt  = max(item["target"].shape[0]        for item in batch)

    enc      = torch.zeros(B, max_k,   2, dtype=torch.float32)
    enc_mask = torch.zeros(B, max_k,      dtype=torch.float32)
    tgt      = torch.zeros(B, max_tgt,    dtype=torch.float32)
    tgt_mask = torch.zeros(B, max_tgt,    dtype=torch.float32)
    prior    = torch.zeros(B, 2,          dtype=torch.float32)
    log_jmax = torch.zeros(B,             dtype=torch.float32)

    for i, item in enumerate(batch):
        k = item["encoder_input"].shape[0]
        L = item["target"].shape[0]
        enc[i, :k] = item["encoder_input"]
        enc_mask[i, :k] = 1.0
        tgt[i, :L] = item["target"]
        tgt_mask[i, :L] = 1.0
        prior[i] = item["prior"]
        log_jmax[i] = item["log_J_max"]

    return {
        "encoder_input": enc,
        "encoder_mask":  enc_mask,
        "target":        tgt,
        "target_mask":   tgt_mask,
        "prior":         prior,
        "log_J_max":     log_jmax,
        "event_ids":     [item["event_id"] for item in batch],
        "groups":        [item["group"]    for item in batch],
        "ks":            [item["k"]        for item in batch],
    }


# ── Group-aware K-fold по event_id ───────────────────────────────────────────

def event_kfold(index_df: pd.DataFrame, n_splits: int = 5,
                random_state: int = 42) -> list[tuple[list[str], list[str]]]:
    """Возвращает список (train_event_ids, val_event_ids) для каждого фолда.

    Только train-события (split=='train'). Стратифицируем по группе W/E.
    """
    from sklearn.model_selection import StratifiedKFold

    train = index_df[index_df["split"] == "train"].copy()
    # Стратификация по комбинации group × log_J_max-bin
    train["jbin"] = pd.cut(train["log_J_real"], bins=3, labels=False).astype(int)
    train["strat"] = train["group"].astype(str) + "_" + train["jbin"].astype(str)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    eids = train["event_id"].to_numpy()
    out  = []
    for tr_i, va_i in skf.split(eids, train["strat"].to_numpy()):
        out.append((eids[tr_i].tolist(), eids[va_i].tolist()))
    return out


# ── Smoke ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    idx = load_index()
    print(f"Профилей со статусом ok: {len(idx)}")
    if not PRIOR_PATH.exists():
        print("Сначала запусти prior.py чтобы получить J_max_prior / T_delta_prior")
        sys.exit(0)

    normalizer = PriorNormalizer().fit(idx)
    print(f"Нормализатор: J μ={normalizer.j_mean:.3f} σ={normalizer.j_std:.3f}, "
          f"T μ={normalizer.t_mean:.3f} σ={normalizer.t_std:.3f}")

    train_idx = idx[idx["split"] == "train"]
    ds = SPEDataset(train_idx, normalizer)
    print(f"Train: {len(ds.profiles)} событий → {len(ds)} обучающих примеров")

    sample = ds[0]
    print("Пример [0]:")
    for k, v in sample.items():
        if hasattr(v, "shape"):
            print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")
        else:
            print(f"  {k}: {v}")

    folds = event_kfold(idx, n_splits=5)
    for i, (tr, va) in enumerate(folds):
        print(f"  fold {i}: train events={len(tr)}, val events={len(va)}")
