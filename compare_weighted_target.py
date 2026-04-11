"""
compare_weighted_target.py
==========================
Подход 1: взвешивание потерь по значению J_max (аналог MEMPSEP-II).

Идея: события с высоким J_max физически важнее и редки в выборке SC25,
поэтому при обучении увеличиваем их вес в MSE.

  w_i = (log10(Jmax_i) - log10(Jmax_min) + 0.5)^alpha

alpha = 1.5  (по умолчанию) — умеренное усиление, не даёт крайних весов.

Для регрессии T_delta применяем те же веса (события с высоким Jmax
физически значимее независимо от прогнозируемой переменной).

SVR не поддерживает sample_weight — обучается без весов (помечено *).

Результаты: results_weighted/target_weighted_results.xlsx

Запуск: python compare_weighted_target.py
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

from spe_utils import build_features, COL_CYCLE

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).parent
OUT_DIR  = ROOT / "results_weighted"
OUT_XLSX = OUT_DIR / "target_weighted_results.xlsx"

# ── Конфигурация ──────────────────────────────────────────────────────────────

ALPHA = 1.5   # степень взвешивания; 1.0 = линейно в log-пространстве

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
    "SVR*":     SVR(kernel="rbf", C=10.0),   # * = без весов
}
NO_WEIGHT_MODELS = {"SVR*"}   # не поддерживают sample_weight


# ── Загрузка данных ───────────────────────────────────────────────────────────

def load():
    df = build_features(
        pd.read_excel(ROOT / "data" / "ОБЪЕДИНЕННЫЙ КАТАЛОГ СПС 23-25.xlsx",
                      sheet_name="Флюэс GOES")
    )
    cycle = pd.to_numeric(df[COL_CYCLE], errors="coerce")
    mask  = df["Jmax"].fillna(0) >= 10
    train = df[cycle.isin([23, 24]) & mask].copy()
    test  = df[cycle.isin([25])     & mask].copy()
    print(f"Train SC23+SC24: {len(train)}  |  Test SC25: {len(test)}")
    return train, test


# ── Веса по J_max ─────────────────────────────────────────────────────────────

def jmax_weights(jmax_series: pd.Series, alpha: float = ALPHA) -> np.ndarray:
    """
    w_i = (log10(Jmax_i) - log10(Jmax_min) + 0.5)^alpha
    Нормализованы к среднему = 1.
    """
    y_log = np.log10(np.clip(jmax_series.values, 10.0, None))
    raw = (y_log - y_log.min() + 0.5) ** alpha
    return raw / raw.mean()


# ── Подготовка матриц X / y ───────────────────────────────────────────────────

def prep_xy(df, feat_cols, tgt_col, log_tgt):
    # Уникальный список колонок (избегаем дублирования когда tgt_col=="Jmax")
    all_cols = list(dict.fromkeys(feat_cols + [tgt_col, "Jmax"]))
    work = df[all_cols].copy()
    for c in all_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    mask = work[all_cols].apply(np.isfinite).all(axis=1)
    work = work[mask]
    X = work[feat_cols].to_numpy()
    y = work[tgt_col].to_numpy()
    w = jmax_weights(work["Jmax"])
    if log_tgt:
        y = np.log10(np.maximum(y, 1e-6))
    return X, y, w, work.index


# ── Метрики ───────────────────────────────────────────────────────────────────

def metrics(y_true, y_pred, log_tgt):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) < 2:
        return {}
    rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
    r2   = float(r2_score(yt, yp))
    rho  = float(spearmanr(yt, yp).statistic)
    if log_tgt:
        return {"primary": rmse, "r2": r2, "spearman": rho}   # RMSLE в log-пространстве
    return {"primary": rmse, "r2": r2, "spearman": rho}


# ── CV leave-one-cycle-out ────────────────────────────────────────────────────

def cycle_splits(df):
    cycle = pd.to_numeric(df[COL_CYCLE], errors="coerce").values
    cycles = sorted(set(c for c in cycle if not np.isnan(c)))
    splits = []
    for c in cycles:
        val = np.where(cycle == c)[0]
        tr  = np.where(cycle != c)[0]
        if len(tr) > 0 and len(val) > 0:
            splits.append((tr, val))
    return splits


# ── Обучение с весами ─────────────────────────────────────────────────────────

def fit_score_weighted(train_df, test_df, feat_cols, tgt_col, log_tgt):
    X_tr, y_tr, w_tr, idx_tr = prep_xy(train_df, feat_cols, tgt_col, log_tgt)
    # Тест: без весов
    te_cols = list(dict.fromkeys(feat_cols + [tgt_col]))
    work_te = test_df[te_cols].copy()
    for c in te_cols:
        work_te[c] = pd.to_numeric(work_te[c], errors="coerce")
    mask_te = work_te.apply(np.isfinite).all(axis=1)
    work_te = work_te[mask_te]
    X_te = work_te[feat_cols].to_numpy()
    y_te = work_te[tgt_col].to_numpy()
    if log_tgt:
        y_te = np.log10(np.maximum(y_te, 1e-6))

    sx = StandardScaler().fit(X_tr)
    sy = StandardScaler().fit(y_tr.reshape(-1, 1))
    X_tr_s = sx.transform(X_tr)
    X_te_s = sx.transform(X_te) if len(X_te) > 0 else np.empty((0, X_tr_s.shape[1]))
    y_tr_s = sy.transform(y_tr.reshape(-1, 1)).ravel()

    splits = cycle_splits(train_df.loc[idx_tr])
    if len(splits) < 1:
        from sklearn.model_selection import KFold
        splits = list(KFold(5, shuffle=True, random_state=42).split(X_tr_s))

    cv_res, te_res = {}, {}

    for mname, mdl in REG_MODELS.items():
        use_w = mname not in NO_WEIGHT_MODELS

        # CV
        y_cv_s = np.full(len(y_tr_s), np.nan)
        for tr_idx, val_idx in splits:
            m = clone(mdl)
            if use_w:
                m.fit(X_tr_s[tr_idx], y_tr_s[tr_idx], sample_weight=w_tr[tr_idx])
            else:
                m.fit(X_tr_s[tr_idx], y_tr_s[tr_idx])
            y_cv_s[val_idx] = m.predict(X_tr_s[val_idx])
        y_cv = sy.inverse_transform(y_cv_s.reshape(-1, 1)).ravel()
        cv_res[mname] = metrics(y_tr, y_cv, log_tgt)

        # Full train → test
        m_full = clone(mdl)
        if use_w:
            m_full.fit(X_tr_s, y_tr_s, sample_weight=w_tr)
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

    print(f"Подход 1: взвешивание по J_max  (alpha={ALPHA})")
    train, test = load()

    targets = [("Jmax", True), ("T_delta", False)]
    rows = []

    for fs_label, fs_cols in FEATURE_SETS:
        for tgt_col, log_tgt in targets:
            print(f"  [{tgt_col}] {fs_label} ...", end=" ", flush=True)
            try:
                cv_res, te_res = fit_score_weighted(train, test, fs_cols, tgt_col, log_tgt)
                for mname in REG_MODELS:
                    cv_m = cv_res.get(mname, {})
                    te_m = te_res.get(mname, {})
                    rows.append(dict(
                        approach=f"target_w(α={ALPHA})",
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

    # Сводка
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
