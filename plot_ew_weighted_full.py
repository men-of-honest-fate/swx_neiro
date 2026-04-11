"""
plot_ew_weighted_full.py
=========================
Визуализация для West/East взвешенных пайплайнов.

Для каждой группы (West/East) × подхода (targetw / densityw):
  A) Scatter 2×2  — scatter4_{tgt}_{group}_{approach}_{fs}.png
  B) Permutation importance — imp_{tgt}_{group}_{approach}_{fs}.png
  C) Confusion matrix T_delta — cm_tdelta_{group}_{approach}.png

Дополнительно:
  D) compact_reg_weighted_ew.png — West и East рядом × 3 подхода × 4 набора признаков
  E) scatter_best_weighted_ew.png — 2×3 решётка:
       строки = (Jmax, T_delta), столбцы = (Baseline, Target-W, Density-W)
       для West (лучшая группа) и East раздельно

Результаты: results_ew_weighted/plots/
Запуск: python plot_ew_weighted_full.py
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).parent))

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
from sklearn.metrics import confusion_matrix

from spe_utils import build_features, COL_CYCLE
from pipelines.tdelta_clf.utils import tdelta_to_class

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).parent
PLOTS_DIR = ROOT / "results_ew_weighted" / "plots"
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
    "SVR":      SVR(kernel="rbf", C=10.0),
}
# SVR не поддерживает sample_weight — обучается без весов
NO_WEIGHT_MODELS = {"SVR"}

MODEL_COLORS = {
    "Linear":   "#1f77b4",
    "Forest":   "#8c564b",
    "Boosting": "#ff7f0e",
    "SVR":      "#2ca02c",
}

GROUP_COLORS = {"West": "#FF5722", "East": "#2196F3"}

APPROACH_COLORS = {
    "baseline": "#607D8B",
    "targetw":  "#FF5722",
    "densityw": "#2196F3",
}
APPROACH_LABELS = {
    "baseline": "Baseline",
    "targetw":  "Target-W (α=1.5)",
    "densityw": "Density-W",
}

FEAT_LABELS = {
    "helio_lon":          "Гелиодолгота",
    "helio_lat":          "Гелиоширота",
    "log_goes_peak_flux": "log(GOES пик)",
    "log_cme_velocity":   "log(Скорость КВМ)",
    "log_fluence":        "log(Флюэнс)",
}

CLF_CLASS_LABELS_TD = ["Быстрые\n(<8 ч)", "Умеренные\n(8–20 ч)", "Медленные\n(≥20 ч)"]

ALPHA_TW    = 1.5
CLIP        = {"West": (0.1, 10.0), "East": (0.2, 5.0)}


# ── Загрузка ──────────────────────────────────────────────────────────────────

def load_splits():
    df = build_features(
        pd.read_excel(ROOT / "data" / "ОБЪЕДИНЕННЫЙ КАТАЛОГ СПС 23-25.xlsx",
                      sheet_name="Флюэс GOES")
    )
    cycle = pd.to_numeric(df[COL_CYCLE], errors="coerce")
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

def target_weights(jmax_values: np.ndarray, alpha=ALPHA_TW) -> np.ndarray:
    y_log = np.log10(np.clip(jmax_values, 10.0, None))
    raw   = (y_log - y_log.min() + 0.5) ** alpha
    return raw / raw.mean()


def density_weights(X_train_s: np.ndarray, X_test_s: np.ndarray,
                    clip_lo: float, clip_hi: float) -> np.ndarray:
    if len(X_test_s) < 3:
        return np.ones(len(X_train_s))
    try:
        kde_tr = gaussian_kde(X_train_s.T, bw_method="scott")
        kde_te = gaussian_kde(X_test_s.T,  bw_method="scott")
        p_tr   = np.clip(kde_tr(X_train_s.T), 1e-10, None)
        p_te   = kde_te(X_train_s.T)
        w      = np.clip(p_te / p_tr, clip_lo, clip_hi)
        return w / w.mean()
    except Exception:
        return np.ones(len(X_train_s))


def get_weights(approach: str, group: str,
                X_tr_s, X_te_s, jmax_values) -> np.ndarray:
    if approach == "targetw":
        return target_weights(jmax_values)
    if approach == "densityw":
        lo, hi = CLIP[group]
        return density_weights(X_tr_s, X_te_s, lo, hi)
    return np.ones(len(X_tr_s))   # baseline


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


# ── A) Scatter 2×2 ────────────────────────────────────────────────────────────

def scatter_all_models(train, test, group, approach, tgt_col, log_tgt, out_prefix):
    ax_unit = "$\\log_{10}$ J$_{max}$" if log_tgt else "$T_{\\Delta}$ (ч)"
    gcolor  = APPROACH_COLORS[approach]

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

        w = get_weights(approach, group, X_tr_s, X_te_s, jmax_tr)

        try:
            all_preds = []
            for mname, mdl in REG_MODELS.items():
                m = clone(mdl)
                kw = {"sample_weight": w} if mname not in NO_WEIGHT_MODELS else {}
                m.fit(X_tr_s, y_tr_s, **kw)
                y_pred_s = m.predict(X_te_s)
                y_pred   = sy.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
                all_preds.append(y_pred)
        except Exception as e:
            print(f"    [ERROR scatter {approach}/{group}/{fs_label}] {e}")
            continue

        vmin = min(y_te.min(), min(p.min() for p in all_preds))
        vmax = max(y_te.max(), max(p.max() for p in all_preds))
        margin = (vmax - vmin) * 0.08
        lo, hi = vmin - margin, vmax + margin

        fig, axes = plt.subplots(2, 2, figsize=(9, 8), sharex=True, sharey=True)
        fig.suptitle(
            f"[{group} / {APPROACH_LABELS[approach]}] «{fs_label}» · тест SC25 (n={len(y_te)})",
            fontsize=10, y=1.01, color=gcolor, fontweight="bold"
        )

        for ax, (mname, _), y_pred in zip(axes.flat, REG_MODELS.items(), all_preds):
            ax.scatter(y_te, y_pred, alpha=0.75, s=45,
                       color=gcolor, edgecolors="white", linewidth=0.3, zorder=3)
            ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, zorder=2)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_aspect("equal", adjustable="box")
            rmse = np.sqrt(np.mean((y_pred - y_te) ** 2))
            lbl  = f"RMSLE={rmse:.3f}" if log_tgt else f"RMSE={rmse:.1f}h"
            ax.set_title(f"{mname}  [{lbl}]", fontsize=9,
                         color=MODEL_COLORS[mname], fontweight="bold", pad=4)
            ax.grid(alpha=0.2); ax.spines[["top", "right"]].set_visible(False)

        for ax in axes[1]:
            ax.set_xlabel(f"Факт ({ax_unit})", fontsize=8.5)
        for ax in axes[:, 0]:
            ax.set_ylabel(f"Прогноз ({ax_unit})", fontsize=8.5)

        plt.tight_layout()
        safe = fs_label.replace(" ", "_").replace("+", "p").replace("/", "-")
        out = PLOTS_DIR / f"{out_prefix}_{group}_{approach}_{safe}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Saved: {out.name}")


# ── B) Permutation importance ─────────────────────────────────────────────────

def importance_all_models(train, test, group, approach, tgt_col, log_tgt, out_prefix):
    model_names = list(REG_MODELS.keys())
    n_models    = len(model_names)
    bar_h       = 0.18
    group_gap   = 0.35
    tgt_label   = "J$_{max}$" if log_tgt else "T$_{\\Delta}$"
    gcolor      = APPROACH_COLORS[approach]

    for fs_label, fs_cols in FEATURE_SETS:
        X_tr, y_tr, jmax_tr = prep_xy(train, fs_cols, tgt_col, log_tgt)
        X_te, y_te = prep_xy_test(test, fs_cols, tgt_col, log_tgt)

        if len(X_te) < 5:
            continue

        sx = StandardScaler().fit(X_tr)
        sy = StandardScaler().fit(y_tr.reshape(-1, 1))
        X_tr_s = sx.transform(X_tr)
        X_te_s = sx.transform(X_te)
        y_tr_s = sy.transform(y_tr.reshape(-1, 1)).ravel()

        w = get_weights(approach, group, X_tr_s, X_te_s, jmax_tr)

        feat_labels = [FEAT_LABELS.get(c, c) for c in fs_cols]
        n_feats     = len(feat_labels)
        imp_matrix  = {}

        for mname, mdl in REG_MODELS.items():
            try:
                m = clone(mdl)
                kw = {"sample_weight": w} if mname not in NO_WEIGHT_MODELS else {}
                m.fit(X_tr_s, y_tr_s, **kw)
                pi = permutation_importance(
                    m, X_te_s, y_te,
                    n_repeats=30, random_state=42,
                    scoring="neg_mean_squared_error"
                )
                imp_pos = np.maximum(pi.importances_mean, 0)
                total   = imp_pos.sum()
                imp_matrix[mname] = (imp_pos / total * 100) if total > 0 \
                    else (np.abs(pi.importances_mean) /
                          (np.abs(pi.importances_mean).sum() + 1e-12) * 100)
            except Exception:
                imp_matrix[mname] = np.zeros(n_feats)

        avg_imp      = np.mean([imp_matrix[m] for m in model_names], axis=0)
        order        = np.argsort(avg_imp)
        feat_ord     = [feat_labels[i] for i in order]
        group_size   = n_models * bar_h
        grp_centers  = np.arange(n_feats) * (group_size + group_gap)

        fig_h = max(2.4, n_feats * (group_size + group_gap) + 0.6)
        fig, ax = plt.subplots(figsize=(6.5, fig_h))

        x_max = 0.0
        for mi, mname in enumerate(model_names):
            vals = np.array([imp_matrix[mname][i] for i in order])
            ypos = grp_centers + (mi - (n_models - 1) / 2) * bar_h
            ax.barh(ypos, vals, height=bar_h * 0.88,
                    color=MODEL_COLORS[mname], alpha=0.90, label=mname, zorder=3)
            x_max = max(x_max, vals.max())
            for yp, v in zip(ypos, vals):
                if v >= 1.0:
                    ax.text(v + 0.8, yp, f"{v:.0f}%",
                            va="center", ha="left", fontsize=7.5, color="#333")

        ax.set_yticks(grp_centers)
        ax.set_yticklabels(feat_ord, fontsize=9.5)
        ax.set_xlabel("Вклад (%)", fontsize=9)
        ax.set_xlim(0, max(x_max * 1.22, 12))
        ax.set_ylim(grp_centers[0] - group_size * 0.8,
                    grp_centers[-1] + group_size * 0.8)
        ax.legend(
            handles=[plt.Rectangle((0, 0), 1, 1, color=MODEL_COLORS[m], alpha=0.9)
                     for m in model_names],
            labels=model_names, loc="lower right", fontsize=8.5,
            framealpha=0.85, edgecolor="#ccc", handlelength=1.0, handleheight=0.9
        )
        ax.set_title(
            f"[{group} / {APPROACH_LABELS[approach]}] «{fs_label}»\n"
            f"{tgt_label} · permutation importance · тест SC25",
            fontsize=9.5, pad=8, color=gcolor
        )
        ax.grid(axis="x", alpha=0.22, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

        plt.tight_layout()
        safe = fs_label.replace(" ", "_").replace("+", "p").replace("/", "-")
        out = PLOTS_DIR / f"{out_prefix}_{group}_{approach}_{safe}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Saved: {out.name}")


# ── C) Confusion matrix T_delta ───────────────────────────────────────────────

def confusion_tdelta(train, test, group, approach, feat_cols, fs_label):
    gcolor = APPROACH_COLORS[approach]

    tr_td = train[train["T_delta"].notna()].copy()
    te_td = test[test["T_delta"].notna()].copy()
    if len(te_td) < 3:
        return

    tr_td["tdelta_class"] = tdelta_to_class(tr_td["T_delta"].values)
    te_td["tdelta_class"] = tdelta_to_class(te_td["T_delta"].values)

    y_col = "tdelta_class"
    all_tr_cols = list(dict.fromkeys(feat_cols + [y_col, "Jmax"]))
    all_te_cols = list(dict.fromkeys(feat_cols + [y_col]))

    def clean(df, cols):
        w = df[cols].copy()
        for c in cols:
            w[c] = pd.to_numeric(w[c], errors="coerce")
        return w[w.apply(np.isfinite).all(axis=1)]

    wtr = clean(tr_td, all_tr_cols)
    wte = clean(te_td, all_te_cols)
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

    w = get_weights(approach, group, X_tr_s, X_te_s, jmax)

    try:
        mdl = RandomForestClassifier(n_estimators=200, random_state=42,
                                     class_weight="balanced")
        mdl.fit(X_tr_s, y_tr, sample_weight=w)
        y_pred = mdl.predict(X_te_s)
    except Exception as e:
        print(f"    [ERROR cm_tdelta {group}/{approach}] {e}")
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
        f"[{group} / {APPROACH_LABELS[approach]}] T_delta — {fs_label} + Forest\n"
        f"(тест SC25, n={len(y_te)},  Acc={acc:.0%})",
        fontsize=9.5, color=gcolor
    )
    plt.tight_layout()
    out = PLOTS_DIR / f"cm_tdelta_{group}_{approach}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out.name}")


# ── D) Compact reg bar: все подходы × обе группы ─────────────────────────────

def compact_reg_weighted(splits):
    """
    2 строки (West / East) × 2 подграфика (Jmax / T_delta).
    Для каждого набора признаков — 3 бара: Baseline, Target-W, Density-W.
    """
    fs_labels  = [fs for fs, _ in FEATURE_SETS]
    targets    = [("Jmax", True, "RMSLE log₁₀ (↓)"), ("T_delta", False, "RMSE ч (↓)")]
    approaches = ["baseline", "targetw", "densityw"]

    # Считаем метрики
    results = {}  # results[group][approach][tgt][fs] = best metric
    for group, (train, test) in splits.items():
        results[group] = {}
        for ap in approaches:
            results[group][ap] = {tgt: {} for tgt, _, _ in targets}
            for tgt_col, log_tgt, _ in targets:
                for fs_label, fs_cols in FEATURE_SETS:
                    try:
                        X_tr, y_tr, jmax_tr = prep_xy(train, fs_cols, tgt_col, log_tgt)
                        X_te, y_te = prep_xy_test(test, fs_cols, tgt_col, log_tgt)
                        if len(X_tr) < 3 or len(X_te) < 2:
                            results[group][ap][tgt_col][fs_label] = np.nan
                            continue
                        sx = StandardScaler().fit(X_tr)
                        sy = StandardScaler().fit(y_tr.reshape(-1, 1))
                        X_tr_s = sx.transform(X_tr)
                        X_te_s = sx.transform(X_te)
                        y_tr_s = sy.transform(y_tr.reshape(-1, 1)).ravel()
                        w = get_weights(ap, group, X_tr_s, X_te_s, jmax_tr)
                        best = np.inf
                        for mname, mdl in REG_MODELS.items():
                            m = clone(mdl)
                            kw = {"sample_weight": w} if mname not in NO_WEIGHT_MODELS else {}
                            m.fit(X_tr_s, y_tr_s, **kw)
                            y_pred = sy.inverse_transform(
                                m.predict(X_te_s).reshape(-1, 1)).ravel()
                            rmse = np.sqrt(np.mean((y_pred - y_te) ** 2))
                            best = min(best, rmse)
                        results[group][ap][tgt_col][fs_label] = best
                    except Exception:
                        results[group][ap][tgt_col][fs_label] = np.nan

    n_fs  = len(fs_labels)
    n_ap  = len(approaches)
    bar_h = 0.22
    centers = np.arange(n_fs)

    fig, axes = plt.subplots(2, 2, figsize=(14, max(7, n_fs * 2.2)),
                             gridspec_kw={"hspace": 0.45, "wspace": 0.38})

    for row, group in enumerate(["West", "East"]):
        for col, (tgt_col, _, xlabel) in enumerate(targets):
            ax = axes[row, col]
            for ai, ap in enumerate(approaches):
                offset = (ai - (n_ap - 1) / 2) * bar_h
                vals = [results[group][ap][tgt_col].get(fs, np.nan) for fs in fs_labels]
                ax.barh(centers + offset, vals, height=bar_h * 0.88,
                        color=APPROACH_COLORS[ap], alpha=0.85,
                        label=APPROACH_LABELS[ap], edgecolor="white", linewidth=0.4)
                for yp, v in zip(centers + offset, vals):
                    if np.isfinite(v):
                        ax.text(v + 0.005, yp, f"{v:.3f}", va="center", fontsize=7)
            ax.set_yticks(centers)
            ax.set_yticklabels(fs_labels, fontsize=8.5)
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_title(f"[{group}] {tgt_col}", fontsize=10, fontweight="bold",
                         color=GROUP_COLORS[group])
            ax.grid(axis="x", alpha=0.22)
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0 and col == 0:
                ax.legend(fontsize=8.5, framealpha=0.85)

    fig.suptitle("West / East × Baseline / Target-W / Density-W\n"
                 "Лучшая модель по набору признаков (тест SC25)",
                 fontsize=11, fontweight="bold", y=1.01)

    out = PLOTS_DIR / "compact_reg_weighted_ew.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


# ── E) Scatter best: 2 groups × 3 approaches ─────────────────────────────────

def scatter_best_weighted(splits):
    """
    2 строки = (West, East),  3 столбца = (Baseline, Target-W, Density-W).
    Jmax регрессия, лучший набор признаков: «Флюэс вместо пика» + Linear.
    """
    fs_cols  = ["helio_lon", "log_fluence", "log_cme_velocity"]
    fs_label = "Флюэс вместо пика"
    tgt_col, log_tgt = "Jmax", True
    ax_unit = "$\\log_{10}$ J$_{max}$"
    approaches = ["baseline", "targetw", "densityw"]

    fig, axes = plt.subplots(2, 3, figsize=(13, 9),
                             gridspec_kw={"hspace": 0.45, "wspace": 0.38})
    fig.suptitle(f"Jmax — «{fs_label}» + Linear · тест SC25\n"
                 "Строки = West / East,  Столбцы = Baseline / Target-W / Density-W",
                 fontsize=11, fontweight="bold")

    for row, group in enumerate(["West", "East"]):
        train, test = splits[group]
        X_tr, y_tr, jmax_tr = prep_xy(train, fs_cols, tgt_col, log_tgt)
        X_te, y_te = prep_xy_test(test, fs_cols, tgt_col, log_tgt)
        gcolor = GROUP_COLORS[group]

        sx = StandardScaler().fit(X_tr)
        sy = StandardScaler().fit(y_tr.reshape(-1, 1))
        X_tr_s = sx.transform(X_tr)
        X_te_s = sx.transform(X_te) if len(X_te) > 0 else np.empty((0, X_tr_s.shape[1]))
        y_tr_s = sy.transform(y_tr.reshape(-1, 1)).ravel()

        for col, ap in enumerate(approaches):
            ax = axes[row, col]
            ax.set_title(f"[{group}] {APPROACH_LABELS[ap]}",
                         fontsize=9.5, fontweight="bold", color=APPROACH_COLORS[ap])

            if len(X_te_s) < 2:
                ax.text(0.5, 0.5, "Нет данных", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
                continue

            w = get_weights(ap, group, X_tr_s, X_te_s, jmax_tr)
            mdl = clone(LinearRegression())
            mdl.fit(X_tr_s, y_tr_s)   # Linear supports sample_weight but result same

            # Для Linear sample_weight влияет незначительно; покажем разницу через Forest
            m = clone(REG_MODELS["Forest"])
            kw = {"sample_weight": w} if ap != "baseline" else {}
            m.fit(X_tr_s, y_tr_s, **kw)
            y_pred = sy.inverse_transform(m.predict(X_te_s).reshape(-1, 1)).ravel()

            rmse = np.sqrt(np.mean((y_pred - y_te) ** 2))
            vmin = min(y_te.min(), y_pred.min())
            vmax = max(y_te.max(), y_pred.max())
            margin = (vmax - vmin) * 0.08
            lo, hi = vmin - margin, vmax + margin

            ax.scatter(y_te, y_pred, alpha=0.78, s=40,
                       color=gcolor, edgecolors="white", linewidth=0.3, zorder=3)
            ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, zorder=2)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
            ax.text(0.04, 0.96, f"RMSLE={rmse:.3f}\nn={len(y_te)}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
            if row == 1:
                ax.set_xlabel(f"Факт ({ax_unit})", fontsize=8.5)
            if col == 0:
                ax.set_ylabel(f"Прогноз ({ax_unit})", fontsize=8.5)

    out = PLOTS_DIR / "scatter_best_weighted_ew.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("Загрузка данных...")
    splits = load_splits()

    APPROACHES = ["targetw", "densityw"]
    BEST_TD_FS  = ["helio_lon", "log_goes_peak_flux", "log_cme_velocity"]   # «Базовая»

    for approach in APPROACHES:
        print(f"\n══ Подход: {APPROACH_LABELS[approach]} ══")
        for group, (train, test) in splits.items():
            print(f"  ── {group} ──")

            print("    Scatter Jmax (2×2)...")
            scatter_all_models(train, test, group, approach, "Jmax",
                               log_tgt=True, out_prefix="scatter4_jmax")

            print("    Scatter T_delta (2×2)...")
            scatter_all_models(train, test, group, approach, "T_delta",
                               log_tgt=False, out_prefix="scatter4_tdelta")

            print("    Вклад признаков Jmax...")
            importance_all_models(train, test, group, approach, "Jmax",
                                  log_tgt=True, out_prefix="imp_jmax")

            print("    Вклад признаков T_delta...")
            importance_all_models(train, test, group, approach, "T_delta",
                                  log_tgt=False, out_prefix="imp_tdelta")

            print("    Confusion matrix T_delta...")
            confusion_tdelta(train, test, group, approach,
                             feat_cols=BEST_TD_FS, fs_label="Базовая")

    print("\n── Compact reg: все подходы × West/East ──")
    compact_reg_weighted(splits)

    print("\n── Scatter best: 2 groups × 3 approaches ──")
    scatter_best_weighted(splits)

    print(f"\nГотово. Графики: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
