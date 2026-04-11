"""
compare_ew_weighted_density.py
================================
Density ratio weighting p_test(x)/p_train(x) + раздельные модели West / East.

Clip: West=[0.1, 10.0], East=[0.2, 5.0]  — East осторожнее (n_test=6, KDE шумная).

Результаты: results_ew_weighted/density_weighted_ew_results.xlsx
Запуск: python compare_ew_weighted_density.py
"""

import sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import gaussian_kde, spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib; matplotlib.use("Agg")

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.metrics import r2_score

from spe_utils import build_features, COL_CYCLE

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).parent
OUT_DIR  = ROOT / "results_ew_weighted"
OUT_XLSX = OUT_DIR / "density_weighted_ew_results.xlsx"

# Более агрессивный clip для East из-за малого числа тестовых событий
CLIP = {"West": (0.1, 10.0), "East": (0.2, 5.0)}

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
NO_WEIGHT = {"SVR*"}


def load_splits():
    df = build_features(
        pd.read_excel(ROOT / "data" / "ОБЪЕДИНЕННЫЙ КАТАЛОГ СПС 23-25.xlsx",
                      sheet_name="Флюэс GOES")
    )
    cycle = pd.to_numeric(df[COL_CYCLE], errors="coerce")
    full  = df[df["Jmax"].fillna(0) >= 10].copy()
    tr_all = full[cycle.isin([23, 24])].copy()
    te_all = full[cycle.isin([25])].copy()
    splits = {
        "West": (tr_all[tr_all["helio_lon"] > 0].copy(), te_all[te_all["helio_lon"] > 0].copy()),
        "East": (tr_all[tr_all["helio_lon"] < 0].copy(), te_all[te_all["helio_lon"] < 0].copy()),
    }
    for g, (tr, te) in splits.items():
        print(f"  {g}: Train={len(tr)}  Test={len(te)}")
    return splits


def density_ratio_weights(X_train, X_test, clip_lo, clip_hi):
    if len(X_test) < 3:
        return np.ones(len(X_train))
    try:
        kde_tr = gaussian_kde(X_train.T, bw_method="scott")
        kde_te = gaussian_kde(X_test.T,  bw_method="scott")
        p_tr = np.clip(kde_tr(X_train.T), 1e-10, None)
        p_te = kde_te(X_train.T)
        w = np.clip(p_te / p_tr, clip_lo, clip_hi)
        return w / w.mean()
    except Exception:
        return np.ones(len(X_train))


def prep_xy(df, feat_cols, tgt_col, log_tgt):
    cols = list(dict.fromkeys(feat_cols + [tgt_col]))
    work = df[cols].copy()
    for c in cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work[work.apply(np.isfinite).all(axis=1)]
    X = work[feat_cols].to_numpy()
    y = work[tgt_col].to_numpy()
    if log_tgt:
        y = np.log10(np.maximum(y, 1e-6))
    return X, y, work.index


def metrics(y_true, y_pred, log_tgt):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) < 2:
        return {}
    return {
        "primary":  float(np.sqrt(np.mean((yp - yt) ** 2))),
        "r2":       float(r2_score(yt, yp)),
        "spearman": float(spearmanr(yt, yp).statistic),
    }


def cycle_splits(df, idx):
    cycle = pd.to_numeric(df.loc[idx, COL_CYCLE], errors="coerce").values
    cycles = sorted(set(c for c in cycle if not np.isnan(c)))
    splits = [(np.where(cycle != c)[0], np.where(cycle == c)[0]) for c in cycles
              if len(np.where(cycle != c)[0]) > 0 and len(np.where(cycle == c)[0]) > 0]
    if not splits:
        from sklearn.model_selection import KFold
        splits = list(KFold(5, shuffle=True, random_state=42).split(idx))
    return splits


def fit_score(train_df, test_df, feat_cols, tgt_col, log_tgt, group):
    X_tr, y_tr, idx_tr = prep_xy(train_df, feat_cols, tgt_col, log_tgt)
    X_te, y_te, _      = prep_xy(test_df,  feat_cols, tgt_col, log_tgt)

    if len(X_tr) < 3:
        raise ValueError("Слишком мало обучающих примеров")

    sx = StandardScaler().fit(X_tr)
    sy = StandardScaler().fit(y_tr.reshape(-1, 1))
    X_tr_s = sx.transform(X_tr)
    X_te_s = sx.transform(X_te) if len(X_te) > 0 else np.empty((0, X_tr_s.shape[1]))
    y_tr_s = sy.transform(y_tr.reshape(-1, 1)).ravel()

    clip_lo, clip_hi = CLIP[group]
    w_full = density_ratio_weights(X_tr_s, X_te_s, clip_lo, clip_hi) \
             if len(X_te_s) > 0 else np.ones(len(X_tr_s))

    splits = cycle_splits(train_df, idx_tr)
    cv_res, te_res = {}, {}

    for mname, mdl in REG_MODELS.items():
        use_w = mname not in NO_WEIGHT
        y_cv_s = np.full(len(y_tr_s), np.nan)
        for tr_i, val_i in splits:
            m = clone(mdl)
            if use_w and len(X_te_s) > 0:
                w_fold = density_ratio_weights(X_tr_s[tr_i], X_te_s, clip_lo, clip_hi)
            else:
                w_fold = np.ones(len(tr_i))
            kw = {"sample_weight": w_fold} if use_w else {}
            m.fit(X_tr_s[tr_i], y_tr_s[tr_i], **kw)
            y_cv_s[val_i] = m.predict(X_tr_s[val_i])
        y_cv = sy.inverse_transform(y_cv_s.reshape(-1, 1)).ravel()
        cv_res[mname] = metrics(y_tr, y_cv, log_tgt)

        m_full = clone(mdl)
        kw = {"sample_weight": w_full} if use_w else {}
        m_full.fit(X_tr_s, y_tr_s, **kw)
        if len(X_te_s) > 0:
            y_te_pred = sy.inverse_transform(m_full.predict(X_te_s).reshape(-1, 1)).ravel()
            te_res[mname] = metrics(y_te, y_te_pred, log_tgt)
        else:
            te_res[mname] = {}

    return cv_res, te_res


def main():
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print(f"EW + Density ratio weighting  (clip: West={CLIP['West']}, East={CLIP['East']})")
    splits = load_splits()
    rows = []

    for group, (train, test) in splits.items():
        print(f"\n══ {group} ══")
        for fs_label, fs_cols in FEATURE_SETS:
            for tgt_col, log_tgt in [("Jmax", True), ("T_delta", False)]:
                print(f"  [{tgt_col}] {fs_label} ...", end=" ", flush=True)
                try:
                    cv_res, te_res = fit_score(train, test, fs_cols, tgt_col, log_tgt, group)
                    for mname in REG_MODELS:
                        rows.append(dict(
                            approach=f"density_w({CLIP[group]})", group=group,
                            target=tgt_col, feature_set=fs_label, model=mname,
                            weighted=(mname not in NO_WEIGHT),
                            cv_primary=cv_res.get(mname, {}).get("primary", np.nan),
                            test_primary=te_res.get(mname, {}).get("primary", np.nan),
                            test_r2=te_res.get(mname, {}).get("r2", np.nan),
                            test_spearman=te_res.get(mname, {}).get("spearman", np.nan),
                        ))
                    print("OK")
                except Exception as e:
                    print(f"ERROR: {e}")

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
        for tgt in ["Jmax", "T_delta"]:
            for grp in ["West", "East"]:
                sheet = f"{grp}_{tgt}"
                sub = df[(df["group"] == grp) & (df["target"] == tgt)]
                if not sub.empty:
                    sub.to_excel(w, sheet_name=sheet, index=False)
    print(f"\nСохранено: {OUT_XLSX}")

    for group in ["West", "East"]:
        for tgt_col, log_tgt in [("Jmax", True), ("T_delta", False)]:
            metric = "RMSLE log₁₀" if log_tgt else "RMSE ч"
            sub = df[(df["group"] == group) & (df["target"] == tgt_col)]
            print(f"\n── [{group}] {tgt_col} ({metric}) ──")
            for fs_label, _ in FEATURE_SETS:
                fsub = sub[sub["feature_set"] == fs_label].dropna(subset=["test_primary"])
                if fsub.empty:
                    continue
                best = fsub.loc[fsub["test_primary"].idxmin()]
                print(f"  {fs_label:<26}  {best['model']:>14}  {best['test_primary']:>8.3f}")


if __name__ == "__main__":
    main()
