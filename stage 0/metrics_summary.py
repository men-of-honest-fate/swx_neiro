"""
metrics_summary.py
==================
Сводная таблица метрик (RMSLE/RMSE, R², CC) для всех:
  pipeline × approach (веса) × group × target × feature_set × model.

Покрывает:
  pipeline     = ew (с исх. 4 fsets), ew_no_vel, ew_cme_angles
  approach     = baseline (без весов), density (KDE), target (α=1.5),
                 hybrid (West=density, East=target)
  group        = West, East, All (pooled)
  target       = Jmax (log10), T_delta (часы)
  model        = Linear, Forest, Boosting, SVR

Запуск: python metrics_summary.py
Результат: metrics_summary.csv
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from spe_utils import build_features, COL_CYCLE

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent

REG_MODELS = {
    "Linear":   LinearRegression(),
    "Forest":   RandomForestRegressor(n_estimators=200, random_state=42),
    "Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
    "SVR":      SVR(kernel="rbf", C=10.0, epsilon=0.1),
}
NO_WEIGHT_MODELS = {"SVR"}

ALPHA_TW  = 1.5
CLIP_WEST = (0.1, 10.0)


FS_BASE = [
    ("Базовая",           ["helio_lon", "log_goes_peak_flux", "log_cme_velocity"]),
    ("Флюэс вместо пика", ["helio_lon", "log_fluence", "log_cme_velocity"]),
    ("Обе координаты",    ["helio_lon", "helio_lat", "log_goes_peak_flux", "log_cme_velocity"]),
    ("Координаты+флюэс",  ["helio_lon", "helio_lat", "log_fluence", "log_cme_velocity"]),
]
FS_NO_VEL = [
    ("Базовая",           ["helio_lon", "log_goes_peak_flux"]),
    ("Флюэс вместо пика", ["helio_lon", "log_fluence"]),
    ("Обе координаты",    ["helio_lon", "helio_lat", "log_goes_peak_flux"]),
    ("Координаты+флюэс",  ["helio_lon", "helio_lat", "log_fluence"]),
]
FS_CME_ANGLES = [
    ("Базовая",           ["helio_lon", "log_goes_peak_flux", "log_cme_velocity", "cme_pa_deg", "cme_width_deg"]),
    ("Флюэс вместо пика", ["helio_lon", "log_fluence", "log_cme_velocity", "cme_pa_deg", "cme_width_deg"]),
    ("Обе координаты",    ["helio_lon", "helio_lat", "log_goes_peak_flux", "log_cme_velocity", "cme_pa_deg", "cme_width_deg"]),
    ("Координаты+флюэс",  ["helio_lon", "helio_lat", "log_fluence", "log_cme_velocity", "cme_pa_deg", "cme_width_deg"]),
]


def load_splits():
    df = build_features(pd.read_excel(
        PROJECT_ROOT / "data" / "ОБЪЕДИНЕННЫЙ КАТАЛОГ СПС 23-25.xlsx",
        sheet_name="Флюэс GOES"
    ))
    cycle  = pd.to_numeric(df[COL_CYCLE], errors="coerce")
    tdelta = pd.to_numeric(df["T_delta"], errors="coerce")
    rise   = pd.to_numeric(df["goes_rise_min"], errors="coerce")
    mask = (
        (df["Jmax"].fillna(0) >= 10)
        & (tdelta.fillna(0) <= 40)
        & (rise.fillna(0) <= 120)
    )
    full   = df[mask].copy()
    tr_all = full[cycle.isin([23, 24])].copy()
    te_all = full[cycle.isin([25])].copy()
    return {
        "West": (tr_all[tr_all["helio_lon"] > 0].copy(),
                 te_all[te_all["helio_lon"] > 0].copy()),
        "East": (tr_all[tr_all["helio_lon"] < 0].copy(),
                 te_all[te_all["helio_lon"] < 0].copy()),
        "All":  (tr_all.copy(), te_all.copy()),
    }


def prep_xy(df, feat_cols, tgt_col, log_tgt):
    all_cols = list(dict.fromkeys(feat_cols + [tgt_col, "Jmax"]))
    w = df[all_cols].copy()
    for c in all_cols:
        w[c] = pd.to_numeric(w[c], errors="coerce")
    w = w[w.apply(np.isfinite).all(axis=1)]
    X    = w[feat_cols].to_numpy()
    y    = w[tgt_col].to_numpy()
    jmax = w["Jmax"].to_numpy()
    if log_tgt:
        y = np.log10(np.maximum(y, 1e-6))
    return X, y, jmax


def prep_xy_test(df, feat_cols, tgt_col, log_tgt):
    cols = list(dict.fromkeys(feat_cols + [tgt_col]))
    w = df[cols].copy()
    for c in cols:
        w[c] = pd.to_numeric(w[c], errors="coerce")
    w = w[w.apply(np.isfinite).all(axis=1)]
    X = w[feat_cols].to_numpy()
    y = w[tgt_col].to_numpy()
    if log_tgt:
        y = np.log10(np.maximum(y, 1e-6))
    return X, y


def density_weights(X_tr_s, X_te_s):
    if len(X_te_s) < 3:
        return np.ones(len(X_tr_s))
    try:
        lo, hi = CLIP_WEST
        kde_tr = gaussian_kde(X_tr_s.T, bw_method="scott")
        kde_te = gaussian_kde(X_te_s.T, bw_method="scott")
        p_tr   = np.clip(kde_tr(X_tr_s.T), 1e-10, None)
        w      = np.clip(kde_te(X_tr_s.T) / p_tr, lo, hi)
        return w / w.mean()
    except Exception:
        return np.ones(len(X_tr_s))


def target_weights(jmax):
    yl  = np.log10(np.clip(jmax, 10.0, None))
    raw = (yl - yl.min() + 0.5) ** ALPHA_TW
    return raw / raw.mean()


def compute_weights(approach, group, splits_raw, feat_cols, tgt_col, log_tgt,
                    X_tr_s, X_te_s, jmax_tr):
    """
    Возвращает вектор весов длиной len(X_tr_s) в соответствии с подходом.
    Для 'hybrid' на 'All' совмещает: West-часть — density, East-часть — target.
    """
    if approach == "baseline":
        return np.ones(len(X_tr_s))
    if approach == "density":
        return density_weights(X_tr_s, X_te_s)
    if approach == "target":
        return target_weights(jmax_tr)
    if approach == "hybrid":
        if group == "West":
            return density_weights(X_tr_s, X_te_s)
        if group == "East":
            return target_weights(jmax_tr)
        # All: собираем отдельно West/East
        tr_w, _ = splits_raw["West"]
        tr_e, _ = splits_raw["East"]
        Xw, yw, jw = prep_xy(tr_w, feat_cols, tgt_col, log_tgt)
        Xe, ye, je = prep_xy(tr_e, feat_cols, tgt_col, log_tgt)
        n_w, n_e = len(Xw), len(Xe)
        if n_w + n_e != len(X_tr_s):
            # fallback: порядок не совпал — равномерно
            return np.ones(len(X_tr_s))
        w = np.ones(n_w + n_e)
        if n_w:
            w[:n_w] = density_weights(X_tr_s[:n_w], X_te_s)
        if n_e:
            w[n_w:] = target_weights(je)
        return w / w.mean()
    return np.ones(len(X_tr_s))


def fit_and_score(X_tr, y_tr, X_te, y_te, jmax_tr, approach, group,
                  log_tgt, splits_raw, feat_cols, tgt_col):
    rows = []
    if len(X_tr) < 3 or len(X_te) < 2:
        return rows
    sx = StandardScaler().fit(X_tr)
    sy = StandardScaler().fit(y_tr.reshape(-1, 1))
    X_tr_s = sx.transform(X_tr)
    X_te_s = sx.transform(X_te)
    y_tr_s = sy.transform(y_tr.reshape(-1, 1)).ravel()

    w = compute_weights(approach, group, splits_raw, feat_cols, tgt_col,
                        log_tgt, X_tr_s, X_te_s, jmax_tr)
    # Подгон длины
    w = w[:len(X_tr_s)]

    for mname, mdl in REG_MODELS.items():
        try:
            m = clone(mdl)
            if mname not in NO_WEIGHT_MODELS:
                m.fit(X_tr_s, y_tr_s, sample_weight=w)
            else:
                m.fit(X_tr_s, y_tr_s)
            y_pred = sy.inverse_transform(m.predict(X_te_s).reshape(-1, 1)).ravel()
            rmse = float(np.sqrt(np.mean((y_pred - y_te) ** 2)))
            r2   = float(r2_score(y_te, y_pred)) if len(y_te) >= 2 else np.nan
            cc   = (float(np.corrcoef(y_te, y_pred)[0, 1])
                    if len(y_te) >= 2 and np.std(y_pred) > 0 and np.std(y_te) > 0
                    else np.nan)
            rows.append({
                "model": mname,
                "n_train": len(X_tr),
                "n_test":  len(y_te),
                "metric_name": "RMSLE" if log_tgt else "RMSE",
                "metric_value": rmse,
                "R2": r2,
                "CC": cc,
            })
        except Exception:
            rows.append({
                "model": mname,
                "n_train": len(X_tr),
                "n_test":  len(y_te),
                "metric_name": "RMSLE" if log_tgt else "RMSE",
                "metric_value": np.nan,
                "R2": np.nan,
                "CC": np.nan,
            })
    return rows


def run_experiment(pipeline, feat_sets, approaches, splits_raw, targets, out_rows):
    for approach in approaches:
        for fs_label, fs_cols in feat_sets:
            for tgt_col, log_tgt in targets:
                for group in ("West", "East", "All"):
                    tr_df, te_df = splits_raw[group]
                    X_tr, y_tr, jmax_tr = prep_xy(tr_df, fs_cols, tgt_col, log_tgt)
                    X_te, y_te         = prep_xy_test(te_df, fs_cols, tgt_col, log_tgt)
                    # Для Jmax убираем выброс >30000 pfu (в log10)
                    if log_tgt and len(y_te) > 0:
                        m = y_te < np.log10(30000)
                        X_te, y_te = X_te[m], y_te[m]

                    rows = fit_and_score(
                        X_tr, y_tr, X_te, y_te, jmax_tr,
                        approach, group, log_tgt, splits_raw, fs_cols, tgt_col,
                    )
                    for r in rows:
                        out_rows.append({
                            "pipeline": pipeline,
                            "approach": approach,
                            "group":    group,
                            "target":   tgt_col,
                            "feature_set": fs_label,
                            **r,
                        })


def main():
    splits = load_splits()
    targets = [("Jmax", True), ("T_delta", False)]

    experiments = [
        ("ew",            FS_BASE,       ["baseline", "density", "target", "hybrid"]),
        ("ew_no_vel",     FS_NO_VEL,     ["baseline", "hybrid"]),
        ("ew_cme_angles", FS_CME_ANGLES, ["baseline", "hybrid"]),
    ]

    rows = []
    for pipeline, fsets, approaches in experiments:
        print(f"Pipeline: {pipeline}  ({len(approaches)} approaches, {len(fsets)} fsets)")
        run_experiment(pipeline, fsets, approaches, splits, targets, rows)

    df = pd.DataFrame(rows)
    df = df[[
        "pipeline", "approach", "group", "target", "feature_set",
        "model", "n_train", "n_test",
        "metric_name", "metric_value", "R2", "CC",
    ]]
    df = df.round({"metric_value": 4, "R2": 3, "CC": 3})

    out = ROOT / "metrics_summary.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {out}  ({len(df)} rows)")
    print(df.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
