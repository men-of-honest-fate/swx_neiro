"""
compare_weighted_density.py
===========================
Подход 2: importance weighting через density ratio p_test(x) / p_train(x).

Для каждого набора признаков:
  1. Оцениваем KDE на обучающей и тестовой выборке (multivariate, Scott bw).
  2. w_i = kde_test(x_i) / kde_train(x_i)  для каждого train-события.
  3. Clip весов в [CLIP_LO, CLIP_HI] — против взрывного роста при ~100 событиях.
  4. Нормализуем к среднему = 1.

Веса зависят от набора признаков (разные x → разные плотности).
SVR не поддерживает sample_weight — обучается без весов (помечено *).

Результаты: results_weighted/density_weighted_results.xlsx

Запуск: python compare_weighted_density.py
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import gaussian_kde, spearmanr

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.metrics import r2_score

from spe_utils import build_features, COL_CYCLE

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).parent
OUT_DIR  = ROOT / "results_weighted"
OUT_XLSX = OUT_DIR / "density_weighted_results.xlsx"

# ── Конфигурация ──────────────────────────────────────────────────────────────

CLIP_LO = 0.1   # минимальный вес (не даём редким train-событиям вес < 0.1)
CLIP_HI = 10.0  # максимальный вес (защита от шума KDE при малой выборке)

FEATURE_SETS = [
    ("Базовая",           ["helio_lon", "log_goes_peak_flux", "log_cme_velocity"]),
    ("Флюэс вместо пика", ["helio_lon", "log_fluence",        "log_cme_velocity"]),
    ("Обе координаты",    ["helio_lon", "helio_lat", "log_goes_peak_flux", "log_cme_velocity"]),
    ("Координаты+флюэс",  ["helio_lon", "helio_lat", "log_fluence",        "log_cme_velocity"]),
]

REG_MODELS = {
    "Linear":   LinearRegression(),
    "Forest":   RandomForestRegressor(n_estimators=200, random_state=42),
    "Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
    "SVR*":     SVR(kernel="rbf", C=10.0),
}
NO_WEIGHT_MODELS = {"SVR*"}


# ── Загрузка данных ───────────────────────────────────────────────────────────

def load():
    df = build_features(
        pd.read_excel(PROJECT_ROOT / "data" / "ОБЪЕДИНЕННЫЙ КАТАЛОГ СПС 23-25.xlsx",
                      sheet_name="Флюэс GOES")
    )
    cycle = pd.to_numeric(df[COL_CYCLE], errors="coerce")
    mask  = df["Jmax"].fillna(0) >= 10
    train = df[cycle.isin([23, 24]) & mask].copy()
    test  = df[cycle.isin([25])     & mask].copy()
    print(f"Train SC23+SC24: {len(train)}  |  Test SC25: {len(test)}")
    return train, test


# ── Density ratio weights ─────────────────────────────────────────────────────

def density_ratio_weights(X_train: np.ndarray, X_test: np.ndarray,
                           clip_lo: float = CLIP_LO,
                           clip_hi: float = CLIP_HI) -> np.ndarray:
    """
    w_i = kde_test(x_i) / kde_train(x_i), clipped и нормализованный.
    X_train, X_test — уже StandardScaled (важно для KDE bandwidth).
    """
    # Для 1D — транспонирование не нужно, gaussian_kde принимает (n_features, n_samples)
    kde_tr = gaussian_kde(X_train.T, bw_method="scott")
    kde_te = gaussian_kde(X_test.T,  bw_method="scott")

    p_tr = kde_tr(X_train.T)   # плотность train-распределения в точках train
    p_te = kde_te(X_train.T)   # плотность test-распределения в точках train

    # Защита от деления на ноль
    p_tr = np.clip(p_tr, 1e-10, None)

    w = p_te / p_tr
    w = np.clip(w, clip_lo, clip_hi)
    w = w / w.mean()            # нормировка: sum(w) ≈ n
    return w


# ── Подготовка X / y ─────────────────────────────────────────────────────────

def prep_xy(df, feat_cols, tgt_col, log_tgt):
    work = df[feat_cols + [tgt_col]].copy()
    for c in feat_cols + [tgt_col]:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    mask = work.apply(np.isfinite).all(axis=1)
    work = work[mask]
    X = work[feat_cols].to_numpy()
    y = work[tgt_col].to_numpy()
    if log_tgt:
        y = np.log10(np.maximum(y, 1e-6))
    return X, y, work.index


# ── Метрики ───────────────────────────────────────────────────────────────────

def metrics(y_true, y_pred, log_tgt):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) < 2:
        return {}
    rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
    r2   = float(r2_score(yt, yp))
    rho  = float(spearmanr(yt, yp).statistic)
    return {"primary": rmse, "r2": r2, "spearman": rho}


# ── CV leave-one-cycle-out ────────────────────────────────────────────────────

def cycle_splits_from_idx(train_df, valid_idx):
    """
    valid_idx — индексы строк в train_df, которые прошли фильтр (не NaN).
    Возвращает список (tr_local_idx, val_local_idx) в пространстве valid_idx.
    """
    cycle = pd.to_numeric(
        train_df.loc[valid_idx, COL_CYCLE], errors="coerce"
    ).values
    cycles = sorted(set(c for c in cycle if not np.isnan(c)))
    splits = []
    for c in cycles:
        val = np.where(cycle == c)[0]
        tr  = np.where(cycle != c)[0]
        if len(tr) > 0 and len(val) > 0:
            splits.append((tr, val))
    if not splits:
        from sklearn.model_selection import KFold
        splits = list(KFold(5, shuffle=True, random_state=42).split(valid_idx))
    return splits


# ── Обучение с density weights ────────────────────────────────────────────────

def fit_score_density(train_df, test_df, feat_cols, tgt_col, log_tgt):
    X_tr, y_tr, idx_tr = prep_xy(train_df, feat_cols, tgt_col, log_tgt)
    X_te, y_te, _      = prep_xy(test_df,  feat_cols, tgt_col, log_tgt)

    sx = StandardScaler().fit(X_tr)
    sy = StandardScaler().fit(y_tr.reshape(-1, 1))
    X_tr_s = sx.transform(X_tr)
    X_te_s = sx.transform(X_te) if len(X_te) > 0 else np.empty((0, X_tr_s.shape[1]))
    y_tr_s = sy.transform(y_tr.reshape(-1, 1)).ravel()

    # Веса вычисляются на масштабированных признаках
    w_full = density_ratio_weights(X_tr_s, X_te_s) if len(X_te_s) > 0 \
             else np.ones(len(X_tr_s))

    splits = cycle_splits_from_idx(train_df, idx_tr)

    cv_res, te_res = {}, {}

    for mname, mdl in REG_MODELS.items():
        use_w = mname not in NO_WEIGHT_MODELS

        # CV — при каждом split веса пересчитываются только по train-части фолда
        # (тестовая часть фолда = val_idx → плотность test не меняется, OK)
        y_cv_s = np.full(len(y_tr_s), np.nan)
        for tr_idx, val_idx in splits:
            m = clone(mdl)
            if use_w and len(X_te_s) > 0:
                # Пересчёт density ratio на подвыборке fold-train vs полный test
                w_fold = density_ratio_weights(X_tr_s[tr_idx], X_te_s)
            else:
                w_fold = np.ones(len(tr_idx))
            if use_w:
                m.fit(X_tr_s[tr_idx], y_tr_s[tr_idx], sample_weight=w_fold)
            else:
                m.fit(X_tr_s[tr_idx], y_tr_s[tr_idx])
            y_cv_s[val_idx] = m.predict(X_tr_s[val_idx])
        y_cv = sy.inverse_transform(y_cv_s.reshape(-1, 1)).ravel()
        cv_res[mname] = metrics(y_tr, y_cv, log_tgt)

        # Full train → test
        m_full = clone(mdl)
        if use_w:
            m_full.fit(X_tr_s, y_tr_s, sample_weight=w_full)
        else:
            m_full.fit(X_tr_s, y_tr_s)
        if len(X_te_s) > 0:
            y_te_pred = sy.inverse_transform(
                m_full.predict(X_te_s).reshape(-1, 1)).ravel()
            te_res[mname] = metrics(y_te, y_te_pred, log_tgt)
        else:
            te_res[mname] = {}

    return cv_res, te_res


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print(f"Подход 2: density ratio  (clip=[{CLIP_LO}, {CLIP_HI}])")
    train, test = load()

    targets = [("Jmax", True), ("T_delta", False)]
    rows = []

    for fs_label, fs_cols in FEATURE_SETS:
        for tgt_col, log_tgt in targets:
            print(f"  [{tgt_col}] {fs_label} ...", end=" ", flush=True)
            try:
                cv_res, te_res = fit_score_density(train, test, fs_cols, tgt_col, log_tgt)
                for mname in REG_MODELS:
                    cv_m = cv_res.get(mname, {})
                    te_m = te_res.get(mname, {})
                    rows.append(dict(
                        approach=f"density_w(clip={CLIP_LO}-{CLIP_HI})",
                        target=tgt_col, feature_set=fs_label, model=mname,
                        weighted=(mname not in NO_WEIGHT_MODELS),
                        cv_primary=cv_m.get("primary", np.nan),
                        test_primary=te_m.get("primary", np.nan),
                        test_r2=te_m.get("r2", np.nan),
                        test_spearman=te_m.get("spearman", np.nan),
                    ))
                print("OK")
            except Exception as e:
                print(f"ERROR: {e}")

    df = pd.DataFrame(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
        df[df["target"] == "Jmax"].to_excel(w, sheet_name="jmax_reg",    index=False)
        df[df["target"] == "T_delta"].to_excel(w, sheet_name="tdelta_reg", index=False)
    print(f"\nСохранено: {OUT_XLSX}")

    fs_labels = [fs for fs, _ in FEATURE_SETS]
    for tgt_col, log_tgt in targets:
        metric_name = "RMSLE log₁₀" if log_tgt else "RMSE ч"
        sub = df[df["target"] == tgt_col]
        print(f"\n── {tgt_col} ({metric_name}) — тест SC25 ──")
        print(f"  {'Набор признаков':<26}  {'Лучшая модель':>14}  {'test_primary':>12}")
        print(f"  {'-'*26}  {'-'*14}  {'-'*12}")
        for fs in fs_labels:
            fsub = sub[sub["feature_set"] == fs].dropna(subset=["test_primary"])
            if fsub.empty:
                continue
            best = fsub.loc[fsub["test_primary"].idxmin()]
            print(f"  {fs:<26}  {best['model']:>14}  {best['test_primary']:>12.3f}")


if __name__ == "__main__":
    main()
