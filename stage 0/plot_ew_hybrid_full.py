"""
plot_ew_hybrid_full.py
=======================
Гибридный подход: лучший способ взвешивания для каждой группы:
  West → Density weighting  (p_test(x)/p_train(x), clip=[0.1, 10])
  East → Target weighting   (w ∝ log10(Jmax)^1.5)

Генерирует те же типы графиков, что plot_ew_full.py:
  A) Scatter 2×2  — scatter4_{tgt}_{group}_{fs}.png
  B) Permutation importance — imp_{tgt}_{group}_{fs}.png
  C) Confusion matrix T_delta — cm_tdelta_{group}.png
  D) Compact reg bar chart (Hybrid vs Baseline) — compact_reg_hybrid_ew.png
  E) Scatter best config — scatter_best_hybrid_ew.png

Результаты: results_ew_hybrid/plots/
Запуск: python plot_ew_hybrid_full.py
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               RandomForestClassifier)
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, r2_score


def _cc(y_true, y_pred):
    if len(y_true) < 2 or np.std(y_pred) == 0 or np.std(y_true) == 0:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])

from spe_utils import build_features, COL_CYCLE
from pipelines.tdelta_clf.utils import tdelta_to_class

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).parent
PLOTS_DIR = ROOT / "results_ew_hybrid" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_SETS = [
    ("Базовая",            ["helio_lon", "log_goes_peak_flux", "log_cme_velocity"]),
    ("Флюэс вместо пика",  ["helio_lon", "log_fluence", "log_cme_velocity"]),
    ("Обе координаты",     ["helio_lon", "helio_lat", "log_goes_peak_flux", "log_cme_velocity"]),
    ("Координаты+флюэс",   ["helio_lon", "helio_lat", "log_fluence", "log_cme_velocity"]),
]

REG_MODELS = {
    "Linear":   LinearRegression(),
    "Forest":   RandomForestRegressor(n_estimators=200, random_state=42),
    "Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
    "SVR":      SVR(kernel="rbf", C=10.0, epsilon=0.1),
}
NO_WEIGHT_MODELS = {"SVR"}

MODEL_COLORS = {
    "Linear":   "#1f77b4",
    "Forest":   "#8c564b",
    "Boosting": "#ff7f0e",
    "SVR":      "#2ca02c",
}

# West — оранжевый (density), East — синий (target)
GROUP_COLORS   = {"West": "#FF5722", "East": "#2196F3"}
GROUP_APPROACH = {"West": "Density-W", "East": "Target-W (α=1.5)"}

FEAT_LABELS = {
    "helio_lon":          "Гелиодолгота",
    "helio_lat":          "Гелиоширота",
    "log_goes_peak_flux": "log(GOES пик)",
    "log_cme_velocity":   "log(Скорость КВМ)",
    "log_fluence":        "log(Флюэнс)",
    "cme_pa_deg":         "Поз. угол КВМ",
    "cme_width_deg":      "Угол раствора КВМ",
}

# Фиксированный порядок признаков для графиков важности (снизу вверх)
FEAT_FIXED_ORDER = [
    "helio_lon",
    "helio_lat",
    "log_goes_peak_flux",
    "log_fluence",
    "log_cme_velocity",
    "cme_pa_deg",
    "cme_width_deg",
]

CLF_CLASS_LABELS_TD = ["Быстрые\n(<8 ч)", "Умеренные\n(8–20 ч)", "Медленные\n(≥20 ч)"]

ALPHA_TW  = 1.5
CLIP_WEST = (0.1, 10.0)


# ── Загрузка ──────────────────────────────────────────────────────────────────

def load_splits():
    df = build_features(
        pd.read_excel(PROJECT_ROOT / "data" / "ОБЪЕДИНЕННЫЙ КАТАЛОГ СПС 23-25.xlsx",
                      sheet_name="Флюэс GOES")
    )
    cycle  = pd.to_numeric(df[COL_CYCLE], errors="coerce")
    tdelta    = pd.to_numeric(df["T_delta"],       errors="coerce")
    goes_rise = pd.to_numeric(df["goes_rise_min"], errors="coerce")
    mask = (
        (df["Jmax"].fillna(0) >= 10) &
        (tdelta.fillna(0) <= 40) &
        (goes_rise.fillna(0) <= 120)
    )
    full   = df[mask].copy()
    tr_all = full[cycle.isin([23, 24])].copy()
    te_all = full[cycle.isin([25])].copy()
    splits = {
        "West": (tr_all[tr_all["helio_lon"] > 0].copy(),
                 te_all[te_all["helio_lon"]  > 0].copy()),
        "East": (tr_all[tr_all["helio_lon"] < 0].copy(),
                 te_all[te_all["helio_lon"]  < 0].copy()),
    }
    for g, (tr, te) in splits.items():
        print(f"  {g}: Train={len(tr)}  Test={len(te)}")
    return splits


# ── Веса ──────────────────────────────────────────────────────────────────────

def _target_weights(jmax_values: np.ndarray) -> np.ndarray:
    y_log = np.log10(np.clip(jmax_values, 10.0, None))
    raw   = (y_log - y_log.min() + 0.5) ** ALPHA_TW
    return raw / raw.mean()


def _density_weights(X_tr_s: np.ndarray, X_te_s: np.ndarray) -> np.ndarray:
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


def get_weights(group: str, X_tr_s, X_te_s, jmax_values) -> np.ndarray:
    """Выбирает способ взвешивания по группе."""
    if group == "West":
        return _density_weights(X_tr_s, X_te_s)
    else:  # East
        return _target_weights(jmax_values)


# ── Подготовка X / y ─────────────────────────────────────────────────────────

def prep_xy(df, feat_cols, tgt_col, log_tgt):
    all_cols = list(dict.fromkeys(feat_cols + [tgt_col, "Jmax"]))
    work = df[all_cols].copy()
    for c in all_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work[work[all_cols].apply(np.isfinite).all(axis=1)]
    X    = work[feat_cols].to_numpy()
    y    = work[tgt_col].to_numpy()
    jmax = work["Jmax"].to_numpy()
    if log_tgt:
        y = np.log10(np.maximum(y, 1e-6))
    return X, y, jmax


def prep_xy_test(df, feat_cols, tgt_col, log_tgt):
    cols = list(dict.fromkeys(feat_cols + [tgt_col]))
    work = df[cols].copy()
    for c in cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work[work.apply(np.isfinite).all(axis=1)]
    X = work[feat_cols].to_numpy()
    y = work[tgt_col].to_numpy()
    if log_tgt:
        y = np.log10(np.maximum(y, 1e-6))
    return X, y


def _fit(mdl, X_tr_s, y_tr_s, w):
    m = clone(mdl)
    if type(m).__name__ not in NO_WEIGHT_MODELS:
        m.fit(X_tr_s, y_tr_s, sample_weight=w)
    else:
        m.fit(X_tr_s, y_tr_s)
    return m


# ── A) Scatter 2×2 ────────────────────────────────────────────────────────────

def scatter_all_models(train, test, group, tgt_col, log_tgt, out_prefix):
    ax_unit = "$\\log_{10}$ J$_{max}$" if log_tgt else "$T_{\\Delta}$ (ч)"
    gcolor  = GROUP_COLORS[group]

    for fs_label, fs_cols in FEATURE_SETS:
        X_tr, y_tr, jmax_tr = prep_xy(train, fs_cols, tgt_col, log_tgt)
        X_te, y_te = prep_xy_test(test, fs_cols, tgt_col, log_tgt)

        if len(X_te) < 3:
            continue
        if log_tgt:
            mask_te = y_te < np.log10(30000)
            X_te, y_te = X_te[mask_te], y_te[mask_te]
        if len(X_te) < 3:
            continue

        sx = StandardScaler().fit(X_tr)
        sy = StandardScaler().fit(y_tr.reshape(-1, 1))
        X_tr_s = sx.transform(X_tr)
        X_te_s = sx.transform(X_te)
        y_tr_s = sy.transform(y_tr.reshape(-1, 1)).ravel()
        w      = get_weights(group, X_tr_s, X_te_s, jmax_tr)

        try:
            all_preds = []
            for mname, mdl in REG_MODELS.items():
                m      = _fit(mdl, X_tr_s, y_tr_s, w)
                y_pred = sy.inverse_transform(m.predict(X_te_s).reshape(-1, 1)).ravel()
                all_preds.append(y_pred)
        except Exception as e:
            print(f"  [ERROR scatter {group}/{fs_label}] {e}")
            continue

        vmin = min(y_te.min(), min(p.min() for p in all_preds))
        vmax = max(y_te.max(), max(p.max() for p in all_preds))
        margin = (vmax - vmin) * 0.08
        lo, hi = vmin - margin, vmax + margin

        fig, axes = plt.subplots(2, 2, figsize=(9, 8), sharex=True, sharey=True)
        fig.suptitle(
            f"[{group} / {GROUP_APPROACH[group]}] «{fs_label}» · тест SC25 (n={len(y_te)})",
            fontsize=10, y=1.01, color=gcolor, fontweight="bold"
        )

        for ax, (mname, _), y_pred in zip(axes.flat, REG_MODELS.items(), all_preds):
            ax.scatter(y_te, y_pred, alpha=0.75, s=45,
                       color=gcolor, edgecolors="white", linewidth=0.3, zorder=3)
            ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, zorder=2)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_aspect("equal", adjustable="box")
            rmse = np.sqrt(np.mean((y_pred - y_te) ** 2))
            r2   = r2_score(y_te, y_pred) if len(y_te) >= 2 else np.nan
            cc   = _cc(y_te, y_pred)
            base = f"RMSLE={rmse:.3f}" if log_tgt else f"RMSE={rmse:.1f}h"
            lbl  = f"{base}  R²={r2:.2f}  CC={cc:.2f}"
            ax.set_title(f"{mname}  [{lbl}]", fontsize=8.5,
                         color=MODEL_COLORS[mname], fontweight="bold", pad=4)
            ax.grid(alpha=0.2); ax.spines[["top", "right"]].set_visible(False)

        for ax in axes[1]:
            ax.set_xlabel(f"Факт ({ax_unit})", fontsize=8.5)
        for ax in axes[:, 0]:
            ax.set_ylabel(f"Прогноз ({ax_unit})", fontsize=8.5)

        plt.tight_layout()
        safe = fs_label.replace(" ", "_").replace("+", "p").replace("/", "-")
        out  = PLOTS_DIR / f"{out_prefix}_{group}_{safe}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out.name}")


# ── B) Permutation importance ─────────────────────────────────────────────────

def importance_all_models(train, test, group, tgt_col, log_tgt, out_prefix):
    model_names = list(REG_MODELS.keys())
    n_models    = len(model_names)
    bar_h, group_gap = 0.18, 0.35
    tgt_label   = "J$_{max}$" if log_tgt else "T$_{\\Delta}$"
    gcolor      = GROUP_COLORS[group]

    for fs_label, fs_cols in FEATURE_SETS:
        X_tr, y_tr, jmax_tr = prep_xy(train, fs_cols, tgt_col, log_tgt)
        X_te, y_te = prep_xy_test(test, fs_cols, tgt_col, log_tgt)

        if len(X_te) < 3:
            continue

        sx = StandardScaler().fit(X_tr)
        sy = StandardScaler().fit(y_tr.reshape(-1, 1))
        X_tr_s = sx.transform(X_tr)
        X_te_s = sx.transform(X_te)
        y_tr_s = sy.transform(y_tr.reshape(-1, 1)).ravel()
        w      = get_weights(group, X_tr_s, X_te_s, jmax_tr)

        # Признаки текущего набора в фиксированном порядке (без нулевых)
        ordered_cols = [f for f in FEAT_FIXED_ORDER if f in fs_cols]
        ordered_idxs = [fs_cols.index(f) for f in ordered_cols]
        ordered_labels = [FEAT_LABELS.get(f, f) for f in ordered_cols]
        n_feats = len(ordered_cols)
        group_size  = n_models * bar_h
        grp_centers = np.arange(n_feats) * (group_size + group_gap)

        imp_matrix = {}
        for mname, mdl in REG_MODELS.items():
            try:
                m  = _fit(mdl, X_tr_s, y_tr_s, w)
                pi = permutation_importance(
                    m, X_te_s, y_te, n_repeats=30, random_state=42,
                    scoring="neg_mean_squared_error"
                )
                imp_pos = np.maximum(pi.importances_mean, 0)
                total   = imp_pos.sum()
                imp_col = (imp_pos / total * 100) if total > 0 \
                    else np.abs(pi.importances_mean) / \
                         (np.abs(pi.importances_mean).sum() + 1e-12) * 100
                imp_matrix[mname] = np.array([imp_col[i] for i in ordered_idxs])
            except Exception:
                imp_matrix[mname] = np.zeros(n_feats)

        fig_h = max(2.4, n_feats * (group_size + group_gap) + 0.6)
        fig, ax = plt.subplots(figsize=(6.5, fig_h))

        x_max = 0.0
        for mi, mname in enumerate(model_names):
            vals = imp_matrix[mname]
            ypos = grp_centers + (mi - (n_models - 1) / 2) * bar_h
            ax.barh(ypos, vals, height=bar_h * 0.88,
                    color=MODEL_COLORS[mname], alpha=0.90, label=mname, zorder=3)
            x_max = max(x_max, vals.max())
            for yp, v in zip(ypos, vals):
                if v >= 1.0:
                    ax.text(v + 0.8, yp, f"{v:.0f}%",
                            va="center", ha="left", fontsize=7.5, color="#333")

        ax.set_yticks(grp_centers)
        ax.set_yticklabels(ordered_labels, fontsize=9.5)
        ax.set_xlabel("Вклад (%)", fontsize=9)
        ax.set_xlim(0, max(x_max * 1.22, 12))
        ax.set_ylim(grp_centers[0] - group_size * 0.8,
                    grp_centers[-1] + group_size * 0.8 if n_feats > 1 else group_size * 0.8)
        ax.legend(
            handles=[plt.Rectangle((0, 0), 1, 1, color=MODEL_COLORS[m], alpha=0.9)
                     for m in model_names],
            labels=model_names, loc="lower right", fontsize=8.5,
            framealpha=0.85, edgecolor="#ccc", handlelength=1.0, handleheight=0.9
        )
        ax.set_title(
            f"[{group} / {GROUP_APPROACH[group]}] «{fs_label}»\n"
            f"{tgt_label} · permutation importance · тест SC25",
            fontsize=9.5, pad=8, color=gcolor
        )
        ax.grid(axis="x", alpha=0.22, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

        plt.tight_layout()
        safe = fs_label.replace(" ", "_").replace("+", "p").replace("/", "-")
        out  = PLOTS_DIR / f"{out_prefix}_{group}_{safe}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out.name}")


# ── C) Confusion matrix T_delta ───────────────────────────────────────────────

def confusion_tdelta(train, test, group, feat_cols, fs_label):
    gcolor = GROUP_COLORS[group]

    tr_td = train[train["T_delta"].notna()].copy()
    te_td = test[test["T_delta"].notna()].copy()
    if len(te_td) < 3:
        print(f"  [skip cm_tdelta {group}] too few test samples")
        return

    tr_td["tdelta_class"] = tdelta_to_class(tr_td["T_delta"].values)
    te_td["tdelta_class"] = tdelta_to_class(te_td["T_delta"].values)

    y_col    = "tdelta_class"
    tr_cols  = list(dict.fromkeys(feat_cols + [y_col, "Jmax"]))
    te_cols  = list(dict.fromkeys(feat_cols + [y_col]))

    def clean(df, cols):
        w = df[cols].copy()
        for c in cols:
            w[c] = pd.to_numeric(w[c], errors="coerce")
        return w[w.apply(np.isfinite).all(axis=1)]

    wtr = clean(tr_td, tr_cols)
    wte = clean(te_td, te_cols)
    if len(wte) < 3:
        return

    X_tr  = wtr[feat_cols].to_numpy()
    y_tr  = wtr[y_col].astype(int).to_numpy()
    jmax  = wtr["Jmax"].to_numpy()
    X_te  = wte[feat_cols].to_numpy()
    y_te  = wte[y_col].astype(int).to_numpy()

    sx     = StandardScaler().fit(X_tr)
    X_tr_s = sx.transform(X_tr)
    X_te_s = sx.transform(X_te)
    w      = get_weights(group, X_tr_s, X_te_s, jmax)

    try:
        mdl = RandomForestClassifier(n_estimators=200, random_state=42,
                                     class_weight="balanced")
        mdl.fit(X_tr_s, y_tr, sample_weight=w)
        y_pred = mdl.predict(X_te_s)
    except Exception as e:
        print(f"  [ERROR cm_tdelta {group}] {e}")
        return

    n_cls = len(CLF_CLASS_LABELS_TD)
    cm    = confusion_matrix(y_te, y_pred, labels=list(range(n_cls)), normalize="true")
    acc   = (y_te == y_pred).mean()

    fig, ax = plt.subplots(figsize=(5, 4.5))
    cmap = LinearSegmentedColormap.from_list("wgrp", ["#ffffff", gcolor])
    im   = ax.imshow(cm, cmap=cmap, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(n_cls):
        for j in range(n_cls):
            clr = "white" if cm[i, j] > 0.55 else "black"
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    fontsize=11, color=clr, fontweight="bold")

    ax.set_xticks(range(n_cls)); ax.set_xticklabels(CLF_CLASS_LABELS_TD, fontsize=9)
    ax.set_yticks(range(n_cls)); ax.set_yticklabels(CLF_CLASS_LABELS_TD, fontsize=9)
    ax.set_xlabel("Прогноз", fontsize=10)
    ax.set_ylabel("Факт", fontsize=10)
    ax.set_title(
        f"[{group} / {GROUP_APPROACH[group]}] T_delta — {fs_label} + Forest\n"
        f"(тест SC25, n={len(y_te)},  Acc={acc:.0%})",
        fontsize=9.5, color=gcolor
    )
    plt.tight_layout()
    out = PLOTS_DIR / f"cm_tdelta_{group}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


# ── D) Compact reg: Hybrid vs Baseline ───────────────────────────────────────

def compact_reg_hybrid(splits):
    """
    Горизонтальные бары: 2 строки (West/East) × 2 столбца (Jmax/T_delta).
    Для каждого набора признаков — 2 бара: Baseline и Hybrid.
    Baseline читается из results_ew/ew_results.xlsx (если есть), иначе пересчитывается.
    """
    fs_labels = [fs for fs, _ in FEATURE_SETS]
    targets   = [("Jmax", True, "RMSLE log₁₀ (↓)"), ("T_delta", False, "RMSE ч (↓)")]

    # ── Baseline из xlsx ──────────────────────────────────────────────────────
    baseline = {g: {tgt: {} for tgt, _, _ in targets} for g in ["West", "East"]}
    xlsx_base = ROOT / "results_ew" / "ew_results.xlsx"
    if xlsx_base.exists():
        try:
            df_base = pd.read_excel(xlsx_base, sheet_name="regression")
            for g in ["West", "East"]:
                for tgt_col, _, _ in targets:
                    sub = df_base[(df_base["group"] == g) & (df_base["target"] == tgt_col)]
                    for fs in fs_labels:
                        fsub = sub[sub["feature_set"] == fs].dropna(subset=["test_primary"])
                        baseline[g][tgt_col][fs] = fsub["test_primary"].min() \
                            if not fsub.empty else np.nan
        except Exception as e:
            print(f"  [WARN] baseline xlsx: {e}")

    # ── Hybrid ────────────────────────────────────────────────────────────────
    hybrid = {g: {tgt: {} for tgt, _, _ in targets} for g in ["West", "East"]}
    for group, (train, test) in splits.items():
        for tgt_col, log_tgt, _ in targets:
            for fs_label, fs_cols in FEATURE_SETS:
                try:
                    X_tr, y_tr, jmax_tr = prep_xy(train, fs_cols, tgt_col, log_tgt)
                    X_te, y_te = prep_xy_test(test, fs_cols, tgt_col, log_tgt)
                    if len(X_tr) < 3 or len(X_te) < 2:
                        hybrid[group][tgt_col][fs_label] = np.nan
                        continue
                    sx = StandardScaler().fit(X_tr)
                    sy = StandardScaler().fit(y_tr.reshape(-1, 1))
                    X_tr_s = sx.transform(X_tr)
                    X_te_s = sx.transform(X_te)
                    y_tr_s = sy.transform(y_tr.reshape(-1, 1)).ravel()
                    w = get_weights(group, X_tr_s, X_te_s, jmax_tr)
                    best = np.inf
                    for mname, mdl in REG_MODELS.items():
                        m = _fit(mdl, X_tr_s, y_tr_s, w)
                        y_pred = sy.inverse_transform(
                            m.predict(X_te_s).reshape(-1, 1)).ravel()
                        best = min(best, np.sqrt(np.mean((y_pred - y_te) ** 2)))
                    hybrid[group][tgt_col][fs_label] = best
                except Exception:
                    hybrid[group][tgt_col][fs_label] = np.nan

    # ── Отрисовка ─────────────────────────────────────────────────────────────
    n_fs  = len(fs_labels)
    bar_h = 0.30
    centers = np.arange(n_fs)

    fig, axes = plt.subplots(2, 2, figsize=(13, max(7, n_fs * 2.2)),
                             gridspec_kw={"hspace": 0.45, "wspace": 0.38})

    for row, group in enumerate(["West", "East"]):
        gcolor = GROUP_COLORS[group]
        for col, (tgt_col, _, xlabel) in enumerate(targets):
            ax = axes[row, col]

            # Baseline
            base_vals = [baseline[group][tgt_col].get(fs, np.nan) for fs in fs_labels]
            ax.barh(centers - bar_h / 2, base_vals, height=bar_h * 0.88,
                    color="#607D8B", alpha=0.80, label="Baseline",
                    edgecolor="white", linewidth=0.4)
            for yp, v in zip(centers - bar_h / 2, base_vals):
                if np.isfinite(v):
                    ax.text(v + 0.004, yp, f"{v:.3f}", va="center", fontsize=7.5)

            # Hybrid
            hyb_vals = [hybrid[group][tgt_col].get(fs, np.nan) for fs in fs_labels]
            ax.barh(centers + bar_h / 2, hyb_vals, height=bar_h * 0.88,
                    color=gcolor, alpha=0.85, label=f"Hybrid ({GROUP_APPROACH[group]})",
                    edgecolor="white", linewidth=0.4)
            for yp, v in zip(centers + bar_h / 2, hyb_vals):
                if np.isfinite(v):
                    ax.text(v + 0.004, yp, f"{v:.3f}", va="center", fontsize=7.5)

            ax.set_yticks(centers)
            ax.set_yticklabels(fs_labels, fontsize=8.5)
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_title(f"[{group}] {tgt_col}", fontsize=10,
                         fontweight="bold", color=gcolor)
            ax.legend(fontsize=8.5, framealpha=0.85)
            ax.grid(axis="x", alpha=0.22)
            ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Гибридный подход vs Baseline\n"
        "West → Density-W  |  East → Target-W (α=1.5)",
        fontsize=11, fontweight="bold", y=1.02
    )
    out = PLOTS_DIR / "compact_reg_hybrid_ew.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


# ── E) Scatter best: West и East бок о бок ───────────────────────────────────

def scatter_best_hybrid(splits):
    """
    2 строки (Jmax / T_delta) × 2 столбца (West / East).
    Лучшие конфигурации: Jmax → «Флюэс вместо пика», T_delta → «Базовая», Linear.
    """
    fs_map = {label: cols for label, cols in FEATURE_SETS}
    configs = [
        ("Jmax",    True,  fs_map.get("Флюэс вместо пика", FEATURE_SETS[0][1]),
         "Флюэс вместо пика", "$\\log_{10}$ J$_{max}$", "RMSLE"),
        ("T_delta", False, fs_map.get("Базовая", FEATURE_SETS[0][1]),
         "Базовая", "$T_{\\Delta}$ (ч)", "RMSE"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9),
                             gridspec_kw={"hspace": 0.45, "wspace": 0.38})
    fig.suptitle(
        "Гибридный подход — лучшая конфигурация (Linear) · тест SC25\n"
        "West → Density-W  |  East → Target-W (α=1.5)",
        fontsize=11, fontweight="bold"
    )

    for row, (tgt_col, log_tgt, fs_cols, fs_label, ax_unit, metric_name) in enumerate(configs):
        for col, group in enumerate(["West", "East"]):
            ax     = axes[row, col]
            gcolor = GROUP_COLORS[group]
            train, test = splits[group]

            X_tr, y_tr, jmax_tr = prep_xy(train, fs_cols, tgt_col, log_tgt)
            X_te, y_te = prep_xy_test(test, fs_cols, tgt_col, log_tgt)

            ax.set_title(f"{group} | {tgt_col} | {GROUP_APPROACH[group]}",
                         fontsize=9.5, color=gcolor, fontweight="bold")

            if len(X_te) < 2:
                ax.text(0.5, 0.5, "Нет данных", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
                continue

            sx = StandardScaler().fit(X_tr)
            sy = StandardScaler().fit(y_tr.reshape(-1, 1))
            X_tr_s = sx.transform(X_tr)
            X_te_s = sx.transform(X_te)
            y_tr_s = sy.transform(y_tr.reshape(-1, 1)).ravel()
            w      = get_weights(group, X_tr_s, X_te_s, jmax_tr)

            mdl    = _fit(LinearRegression(), X_tr_s, y_tr_s, w)
            y_pred = sy.inverse_transform(mdl.predict(X_te_s).reshape(-1, 1)).ravel()

            rmse   = np.sqrt(np.mean((y_pred - y_te) ** 2))
            vmin   = min(y_te.min(), y_pred.min())
            vmax   = max(y_te.max(), y_pred.max())
            margin = (vmax - vmin) * 0.08
            lo, hi = vmin - margin, vmax + margin

            ax.scatter(y_te, y_pred, alpha=0.78, s=42,
                       color=gcolor, edgecolors="white", linewidth=0.3, zorder=3)
            ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, zorder=2)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.2); ax.spines[["top", "right"]].set_visible(False)
            r2 = r2_score(y_te, y_pred) if len(y_te) >= 2 else np.nan
            cc = _cc(y_te, y_pred)
            ax.text(0.04, 0.96,
                    f"{metric_name}={rmse:.3f}\nR²={r2:.2f}  CC={cc:.2f}\nn={len(y_te)}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
            if row == 1:
                ax.set_xlabel(f"Факт ({ax_unit})", fontsize=8.5)
            if col == 0:
                ax.set_ylabel(f"Прогноз ({ax_unit})", fontsize=8.5)

    out = PLOTS_DIR / "scatter_best_hybrid_ew.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


# ── F) Общие (pooled) scatter и importance: West+East вместе ──────────────────

def _combined_weights(train_west, train_east, X_tr_s_all, X_te_s):
    """
    Гибридные веса для объединённой выборки:
      West-события → density weights (KDE теста / KDE west-трейна)
      East-события → target weights (∝ log10(Jmax)^α)
    Возвращает вектор весов len(X_tr_s_all).
    """
    n_west = len(train_west)
    n_east = len(train_east)
    n_all  = n_west + n_east

    w = np.ones(n_all)

    # West-часть: density
    if n_west > 0:
        w_west = _density_weights(X_tr_s_all[:n_west], X_te_s)
        w[:n_west] = w_west

    # East-часть: target
    if n_east > 0:
        jmax_east = pd.to_numeric(
            train_east["Jmax"], errors="coerce"
        ).fillna(10.0).to_numpy()
        w_east = _target_weights(jmax_east)
        w[n_west:] = w_east

    # Нормируем: среднее = 1
    w = w / w.mean()
    return w


def scatter_combined(splits, tgt_col, log_tgt, out_prefix):
    """Scatter 2×2 на объединённой West+East выборке."""
    ax_unit  = "$\\log_{10}$ J$_{max}$" if log_tgt else "$T_{\\Delta}$ (ч)"

    train_w, test_w = splits["West"]
    train_e, test_e = splits["East"]
    train_all = pd.concat([train_w, train_e], ignore_index=True)
    test_all  = pd.concat([test_w,  test_e],  ignore_index=True)

    for fs_label, fs_cols in FEATURE_SETS:
        X_tr, y_tr, jmax_tr = prep_xy(train_all, fs_cols, tgt_col, log_tgt)
        # West/East части для весов — по исходным размерам до prep_xy
        # Используем prep_xy отдельно, чтобы знать размеры
        X_trw, y_trw, _ = prep_xy(train_w, fs_cols, tgt_col, log_tgt)
        X_tre, y_tre, _ = prep_xy(train_e, fs_cols, tgt_col, log_tgt)

        X_te, y_te = prep_xy_test(test_all, fs_cols, tgt_col, log_tgt)
        if len(X_te) < 5:
            continue
        if log_tgt:
            mask_te = y_te < np.log10(30000)
            X_te, y_te = X_te[mask_te], y_te[mask_te]
        if len(X_te) < 5:
            continue

        sx = StandardScaler().fit(X_tr)
        sy = StandardScaler().fit(y_tr.reshape(-1, 1))
        X_tr_s = sx.transform(X_tr)
        X_te_s = sx.transform(X_te)
        y_tr_s = sy.transform(y_tr.reshape(-1, 1)).ravel()

        # Собираем раздельно обработанные West/East для весов
        tr_west_for_w = train_w.copy()
        tr_east_for_w = train_e.copy()
        w = _combined_weights(tr_west_for_w, tr_east_for_w, X_tr_s, X_te_s)
        # Выравниваем длину (prep_xy мог отбросить строки с NaN)
        w = w[:len(X_tr_s)]

        try:
            all_preds = []
            for mname, mdl in REG_MODELS.items():
                m      = _fit(mdl, X_tr_s, y_tr_s, w)
                y_pred = sy.inverse_transform(m.predict(X_te_s).reshape(-1, 1)).ravel()
                all_preds.append(y_pred)
        except Exception as e:
            print(f"  [ERROR scatter_combined/{fs_label}] {e}")
            continue

        vmin = min(y_te.min(), min(p.min() for p in all_preds))
        vmax = max(y_te.max(), max(p.max() for p in all_preds))
        margin = (vmax - vmin) * 0.08
        lo, hi = vmin - margin, vmax + margin

        fig, axes = plt.subplots(2, 2, figsize=(9, 8), sharex=True, sharey=True)
        fig.suptitle(
            f"[All / Hybrid] «{fs_label}» · тест SC25 (n={len(y_te)})",
            fontsize=10, y=1.01, fontweight="bold"
        )

        for ax, (mname, _), y_pred in zip(axes.flat, REG_MODELS.items(), all_preds):
            ax.scatter(y_te, y_pred, alpha=0.75, s=45,
                       color=MODEL_COLORS[mname], edgecolors="white",
                       linewidth=0.3, zorder=3)
            ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, zorder=2)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_aspect("equal", adjustable="box")
            rmse = np.sqrt(np.mean((y_pred - y_te) ** 2))
            r2   = r2_score(y_te, y_pred) if len(y_te) >= 2 else np.nan
            cc   = _cc(y_te, y_pred)
            base = f"RMSLE={rmse:.3f}" if log_tgt else f"RMSE={rmse:.1f}h"
            lbl  = f"{base}  R²={r2:.2f}  CC={cc:.2f}"
            ax.set_title(f"{mname}  [{lbl}]", fontsize=8.5,
                         color=MODEL_COLORS[mname], fontweight="bold", pad=4)
            ax.grid(alpha=0.2); ax.spines[["top", "right"]].set_visible(False)

        for ax in axes[1]:
            ax.set_xlabel(f"Факт ({ax_unit})", fontsize=8.5)
        for ax in axes[:, 0]:
            ax.set_ylabel(f"Прогноз ({ax_unit})", fontsize=8.5)

        plt.tight_layout()
        safe = fs_label.replace(" ", "_").replace("+", "p").replace("/", "-")
        out  = PLOTS_DIR / f"{out_prefix}_All_{safe}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out.name}")


def importance_combined(splits, tgt_col, log_tgt, out_prefix):
    """Permutation importance на объединённой West+East выборке."""
    model_names = list(REG_MODELS.keys())
    n_models    = len(model_names)
    bar_h, group_gap = 0.18, 0.35
    tgt_label   = "J$_{max}$" if log_tgt else "T$_{\\Delta}$"

    train_w, test_w = splits["West"]
    train_e, test_e = splits["East"]
    train_all = pd.concat([train_w, train_e], ignore_index=True)
    test_all  = pd.concat([test_w,  test_e],  ignore_index=True)

    for fs_label, fs_cols in FEATURE_SETS:
        X_tr, y_tr, _ = prep_xy(train_all, fs_cols, tgt_col, log_tgt)
        X_te, y_te    = prep_xy_test(test_all, fs_cols, tgt_col, log_tgt)
        if len(X_te) < 5:
            continue

        sx = StandardScaler().fit(X_tr)
        sy = StandardScaler().fit(y_tr.reshape(-1, 1))
        X_tr_s = sx.transform(X_tr)
        X_te_s = sx.transform(X_te)
        y_tr_s = sy.transform(y_tr.reshape(-1, 1)).ravel()

        w = _combined_weights(train_w, train_e, X_tr_s, X_te_s)
        w = w[:len(X_tr_s)]

        ordered_cols   = [f for f in FEAT_FIXED_ORDER if f in fs_cols]
        ordered_idxs   = [fs_cols.index(f) for f in ordered_cols]
        ordered_labels = [FEAT_LABELS.get(f, f) for f in ordered_cols]
        n_feats     = len(ordered_cols)
        group_size  = n_models * bar_h
        grp_centers = np.arange(n_feats) * (group_size + group_gap)

        imp_matrix = {}
        for mname, mdl in REG_MODELS.items():
            try:
                m  = _fit(mdl, X_tr_s, y_tr_s, w)
                pi = permutation_importance(
                    m, X_te_s, y_te, n_repeats=30, random_state=42,
                    scoring="neg_mean_squared_error"
                )
                imp_pos = np.maximum(pi.importances_mean, 0)
                total   = imp_pos.sum()
                imp_col = (imp_pos / total * 100) if total > 0 \
                    else np.abs(pi.importances_mean) / \
                         (np.abs(pi.importances_mean).sum() + 1e-12) * 100
                imp_matrix[mname] = np.array([imp_col[i] for i in ordered_idxs])
            except Exception:
                imp_matrix[mname] = np.zeros(n_feats)

        fig_h = max(2.4, n_feats * (group_size + group_gap) + 0.6)
        fig, ax = plt.subplots(figsize=(6.5, fig_h))

        x_max = 0.0
        for mi, mname in enumerate(model_names):
            vals = imp_matrix[mname]
            ypos = grp_centers + (mi - (n_models - 1) / 2) * bar_h
            ax.barh(ypos, vals, height=bar_h * 0.88,
                    color=MODEL_COLORS[mname], alpha=0.90, label=mname, zorder=3)
            x_max = max(x_max, vals.max())
            for yp, v in zip(ypos, vals):
                if v >= 1.0:
                    ax.text(v + 0.8, yp, f"{v:.0f}%",
                            va="center", ha="left", fontsize=7.5, color="#333")

        ax.set_yticks(grp_centers)
        ax.set_yticklabels(ordered_labels, fontsize=9.5)
        ax.set_xlabel("Вклад (%)", fontsize=9)
        ax.set_xlim(0, max(x_max * 1.22, 12))
        ax.set_ylim(grp_centers[0] - group_size * 0.8,
                    grp_centers[-1] + group_size * 0.8 if n_feats > 1 else group_size * 0.8)
        ax.legend(
            handles=[plt.Rectangle((0, 0), 1, 1, color=MODEL_COLORS[m], alpha=0.9)
                     for m in model_names],
            labels=model_names, loc="lower right", fontsize=8.5,
            framealpha=0.85, edgecolor="#ccc", handlelength=1.0, handleheight=0.9
        )
        ax.set_title(
            f"[All / Hybrid] «{fs_label}»\n"
            f"{tgt_label} · permutation importance · тест SC25",
            fontsize=9.5, pad=8
        )
        ax.grid(axis="x", alpha=0.22, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

        plt.tight_layout()
        safe = fs_label.replace(" ", "_").replace("+", "p").replace("/", "-")
        out  = PLOTS_DIR / f"{out_prefix}_All_{safe}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out.name}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("Загрузка данных...")
    splits = load_splits()

    _fs_map = {label: cols for label, cols in FEATURE_SETS}
    BEST_TD_FS = _fs_map.get("Базовая", FEATURE_SETS[0][1])

    for group, (train, test) in splits.items():
        print(f"\n══ {group} ({GROUP_APPROACH[group]}) ══")

        print("  Scatter Jmax (2×2)...")
        scatter_all_models(train, test, group, "Jmax",    log_tgt=True,
                           out_prefix="scatter4_jmax")

        print("  Scatter T_delta (2×2)...")
        scatter_all_models(train, test, group, "T_delta", log_tgt=False,
                           out_prefix="scatter4_tdelta")

        print("  Вклад признаков Jmax...")
        importance_all_models(train, test, group, "Jmax",    log_tgt=True,
                               out_prefix="imp_jmax")

        print("  Вклад признаков T_delta...")
        importance_all_models(train, test, group, "T_delta", log_tgt=False,
                               out_prefix="imp_tdelta")

        print("  Confusion matrix T_delta...")
        confusion_tdelta(train, test, group,
                         feat_cols=BEST_TD_FS, fs_label="Базовая")

    print("\n── Scatter All (West+East pooled) Jmax ──")
    scatter_combined(splits, "Jmax",    log_tgt=True,  out_prefix="scatter4_jmax")

    print("\n── Scatter All (West+East pooled) T_delta ──")
    scatter_combined(splits, "T_delta", log_tgt=False, out_prefix="scatter4_tdelta")

    print("\n── Importance All (West+East pooled) Jmax ──")
    importance_combined(splits, "Jmax",    log_tgt=True,  out_prefix="imp_jmax")

    print("\n── Importance All (West+East pooled) T_delta ──")
    importance_combined(splits, "T_delta", log_tgt=False, out_prefix="imp_tdelta")

    print("\n── Compact reg: Hybrid vs Baseline ──")
    compact_reg_hybrid(splits)

    print("\n── Scatter best config ──")
    scatter_best_hybrid(splits)

    print(f"\nГотово. Графики: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
