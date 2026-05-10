"""
Out-of-fold prior для LSTM-наукастинга.

Конфигурация моделей по итогам hybrid_no_vel_jmax1 (SUMMARY.md):
  West/J_max   — Linear, набор «Флюэс вместо пика»  + Density-weights
  West/T_delta — SVR,    набор «Флюэс вместо пика»  + Density-weights (SVR не юзает w)
  East/J_max   — Forest, набор «Базовая»            + Target-weights (α=1.5, clip=1)
  East/T_delta — SVR,    набор «Флюэс вместо пика»  + Target-weights

Алгоритм:
  Для (group, target):
    1. Взять train (SC23+24) и test (SC25) подмножества группы.
    2. На train: 5-fold CV → out-of-fold предсказания для всех train-событий.
    3. Обучить на всём train с теми же весами → предсказать на test.
  Сохранить `prior_oof.parquet` со столбцами:
    event_id, group, J_max_prior (log10), T_delta_prior, J_max_real, T_delta_real, split
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

ROOT         = Path(__file__).parent
PROJECT_ROOT = ROOT.parent
RESULTS_DIR  = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from catalog import load_catalog                # noqa: E402

warnings.filterwarnings("ignore")

# ── Конфигурация моделей ─────────────────────────────────────────────────────

REG_MODELS = {
    "Linear": LinearRegression(),
    "Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "SVR":    SVR(kernel="rbf", C=10.0, epsilon=0.1),
}
NO_WEIGHT_MODELS = {"SVR"}

CONFIG: list[dict] = [
    {"group": "West", "target": "Jmax",    "model": "Linear",
     "features": ["helio_lon", "log_fluence"], "log_target": True,
     "weighting": "density"},
    {"group": "West", "target": "T_delta", "model": "SVR",
     "features": ["helio_lon", "log_fluence"], "log_target": False,
     "weighting": "density"},
    {"group": "East", "target": "Jmax",    "model": "Forest",
     "features": ["helio_lon", "log_goes_peak_flux"], "log_target": True,
     "weighting": "target"},
    {"group": "East", "target": "T_delta", "model": "SVR",
     "features": ["helio_lon", "log_fluence"], "log_target": False,
     "weighting": "target"},
]

ALPHA_TW       = 1.5
TW_CLIP_LOWER  = 1.0   # = JMAX_MIN из hybrid_no_vel_jmax1
DENSITY_CLIP   = (0.1, 10.0)
N_SPLITS       = 5
RANDOM_STATE   = 42


# ── Веса ──────────────────────────────────────────────────────────────────────

def target_weights(jmax: np.ndarray) -> np.ndarray:
    y_log = np.log10(np.clip(jmax, TW_CLIP_LOWER, None))
    raw   = (y_log - y_log.min() + 0.5) ** ALPHA_TW
    return raw / raw.mean()


def density_weights(X_tr_s: np.ndarray, X_te_s: np.ndarray) -> np.ndarray:
    if len(X_te_s) < 3 or len(X_tr_s) < 3:
        return np.ones(len(X_tr_s))
    try:
        lo, hi = DENSITY_CLIP
        kde_tr = gaussian_kde(X_tr_s.T, bw_method="scott")
        kde_te = gaussian_kde(X_te_s.T, bw_method="scott")
        p_tr   = np.clip(kde_tr(X_tr_s.T), 1e-10, None)
        w      = np.clip(kde_te(X_tr_s.T) / p_tr, lo, hi)
        return w / w.mean()
    except Exception:
        return np.ones(len(X_tr_s))


def get_weights(weighting: str, X_tr_s, X_te_s, jmax_tr) -> np.ndarray:
    if weighting == "density":
        return density_weights(X_tr_s, X_te_s)
    if weighting == "target":
        return target_weights(jmax_tr)
    return np.ones(len(X_tr_s))


# ── Подготовка X / y с сохранением индексов ──────────────────────────────────

def _prep_xy(df: pd.DataFrame, feat_cols: list[str], tgt_col: str,
             log_tgt: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Возвращает (X, y, jmax, valid_mask) где valid_mask — индексы исходного df."""
    cols = list(dict.fromkeys(feat_cols + [tgt_col, "Jmax"]))
    work = df[cols].copy()
    for c in cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    valid = work[cols].apply(np.isfinite).all(axis=1).to_numpy()
    work_v = work[valid]
    X = work_v[feat_cols].to_numpy()
    y = work_v[tgt_col].to_numpy()
    jmax = work_v["Jmax"].to_numpy()
    if log_tgt:
        y = np.log10(np.maximum(y, 1e-6))
    return X, y, jmax, valid


# ── OOF prior для одной (group, target) пары ─────────────────────────────────

def compute_one(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    group       = cfg["group"]
    target_col  = cfg["target"]
    model_name  = cfg["model"]
    feat_cols   = cfg["features"]
    log_target  = cfg["log_target"]
    weighting   = cfg["weighting"]

    sub = df[df["group"] == group].copy().reset_index(drop=True)
    train_mask = sub["split"] == "train"
    test_mask  = sub["split"] == "test"

    train = sub[train_mask].reset_index(drop=True)
    test  = sub[test_mask].reset_index(drop=True)

    X_tr, y_tr, jmax_tr, valid_tr = _prep_xy(train, feat_cols, target_col, log_target)
    X_te, y_te, _,        valid_te = _prep_xy(test,  feat_cols, target_col, log_target)

    sx = StandardScaler().fit(X_tr)
    X_tr_s = sx.transform(X_tr)
    X_te_s = sx.transform(X_te) if len(X_te) else np.empty((0, X_tr.shape[1]))

    sy = StandardScaler().fit(y_tr.reshape(-1, 1))
    y_tr_s = sy.transform(y_tr.reshape(-1, 1)).ravel()

    # ── OOF на train ──
    oof = np.full(len(y_tr), np.nan)
    kf  = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    for fold, (tr_i, va_i) in enumerate(kf.split(X_tr_s)):
        w = get_weights(weighting, X_tr_s[tr_i], X_tr_s[va_i], jmax_tr[tr_i])
        mdl = clone(REG_MODELS[model_name])
        if model_name in NO_WEIGHT_MODELS:
            mdl.fit(X_tr_s[tr_i], y_tr_s[tr_i])
        else:
            mdl.fit(X_tr_s[tr_i], y_tr_s[tr_i], sample_weight=w)
        oof[va_i] = sy.inverse_transform(
            mdl.predict(X_tr_s[va_i]).reshape(-1, 1)
        ).ravel()

    # ── финальная модель на полном train → предсказание на test ──
    test_pred = np.full(len(y_te), np.nan)
    if len(X_te_s) > 0:
        w_full = get_weights(weighting, X_tr_s, X_te_s, jmax_tr)
        mdl = clone(REG_MODELS[model_name])
        if model_name in NO_WEIGHT_MODELS:
            mdl.fit(X_tr_s, y_tr_s)
        else:
            mdl.fit(X_tr_s, y_tr_s, sample_weight=w_full)
        test_pred = sy.inverse_transform(
            mdl.predict(X_te_s).reshape(-1, 1)
        ).ravel()

    # ── метрики ──
    rmse_oof  = float(np.sqrt(np.mean((oof - y_tr) ** 2))) if len(oof) else np.nan
    rmse_test = float(np.sqrt(np.mean((test_pred - y_te) ** 2))) if len(y_te) else np.nan
    print(f"  [{group:<5}/{target_col:<7}/{model_name:<6}] "
          f"OOF RMSE={rmse_oof:.3f}  Test RMSE={rmse_test:.3f}  "
          f"(n_train={len(y_tr)}, n_test={len(y_te)})")

    # ── собрать DataFrame: event_id × prediction ──
    pred_col = "J_max_prior" if target_col == "Jmax" else "T_delta_prior"
    real_col = "J_max_real"  if target_col == "Jmax" else "T_delta_real"

    rows = []
    j = 0
    for i in range(len(train)):
        if not valid_tr[i]:
            continue
        rows.append({
            "event_id": train.iloc[i]["event_id"],
            "group":    group,
            "split":    "train",
            pred_col:   float(oof[j]),
            real_col:   float(y_tr[j]),
        })
        j += 1
    j = 0
    for i in range(len(test)):
        if not valid_te[i]:
            continue
        rows.append({
            "event_id": test.iloc[i]["event_id"],
            "group":    group,
            "split":    "test",
            pred_col:   float(test_pred[j]),
            real_col:   float(y_te[j]),
        })
        j += 1
    return pd.DataFrame(rows)


def compute_all() -> pd.DataFrame:
    df = load_catalog()
    print(f"Каталог: {len(df)} событий, train={int((df['split']=='train').sum())}, "
          f"test={int((df['split']=='test').sum())}")

    parts = [compute_one(df, cfg) for cfg in CONFIG]

    # Слить J_max и T_delta предсказания по event_id
    merged = None
    for part in parts:
        if merged is None:
            merged = part
        else:
            merged = merged.merge(part, on=["event_id", "group", "split"], how="outer")

    out = RESULTS_DIR / "prior_oof.parquet"
    merged.to_parquet(out, index=False)
    print(f"\n✓ Сохранено: {out}  ({len(merged)} событий)")

    print("\nСтатистики prior:")
    for col in ("J_max_prior", "T_delta_prior", "J_max_real", "T_delta_real"):
        if col in merged.columns:
            n = merged[col].notna().sum()
            print(f"  {col}: n={n}, mean={merged[col].mean():.3f}, std={merged[col].std():.3f}")
    return merged


if __name__ == "__main__":
    compute_all()
