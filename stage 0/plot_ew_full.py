"""
plot_ew_full.py
===============
Генерация графиков для West vs East сравнения (аналог plot_simple_full.py).

Для каждой группы (West/East):
  A) Scatter 2×2 — все 4 модели × все наборы признаков (Jmax и T_delta)
  B) Permutation importance — горизонтальные бары × все наборы признаков
  C) Confusion matrix — для T_delta классификации (лучший набор признаков)

Дополнительно:
  D) Compact reg bar chart — West vs East бок о бок (RMSLE Jmax | RMSE T_delta)

Результаты: results_ew/plots/

Запуск: python plot_ew_full.py
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
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
PLOTS_DIR = ROOT / "results_ew" / "plots"
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

MODEL_COLORS = {
    "Linear":   "#1f77b4",
    "Forest":   "#8c564b",
    "Boosting": "#ff7f0e",
    "SVR":      "#2ca02c",
}

GROUP_COLORS = {"West": "#FF5722", "East": "#2196F3"}

FEAT_LABELS = {
    "helio_lon":          "Гелиодолгота",
    "helio_lat":          "Гелиоширота",
    "log_goes_peak_flux": "log(GOES пик)",
    "log_cme_velocity":   "log(Скорость КВМ)",
    "log_fluence":        "log(Флюэнс)",
}

CLF_CLASS_LABELS_TD = ["Быстрые\n(< 8 ч)", "Умеренные\n(8–20 ч)", "Медленные\n(≥ 20 ч)"]


# ── Загрузка и разбивка данных ────────────────────────────────────────────────

def load_splits():
    df = build_features(
        pd.read_excel(PROJECT_ROOT / "data" / "ОБЪЕДИНЕННЫЙ КАТАЛОГ СПС 23-25.xlsx",
                      sheet_name="Флюэс GOES")
    )
    cycle     = pd.to_numeric(df[COL_CYCLE], errors="coerce")
    tdelta    = pd.to_numeric(df["T_delta"],       errors="coerce")
    goes_rise = pd.to_numeric(df["goes_rise_min"], errors="coerce")
    mask = (
        (df["Jmax"].fillna(0) >= 10) &
        (tdelta.fillna(0) <= 40) &
        (goes_rise.fillna(0) <= 120)
    )
    df_full = df[mask].copy()

    train_all = df_full[cycle.isin([23, 24])].copy()
    test_all  = df_full[cycle.isin([25])].copy()

    splits = {
        "West": (train_all[train_all["helio_lon"] > 0].copy(),
                 test_all[test_all["helio_lon"]   > 0].copy()),
        "East": (train_all[train_all["helio_lon"] < 0].copy(),
                 test_all[test_all["helio_lon"]   < 0].copy()),
    }
    for g, (tr, te) in splits.items():
        print(f"  {g}: Train={len(tr)}  Test={len(te)}")
    return splits


# ── Утилиты ───────────────────────────────────────────────────────────────────

def prep_xy(df, feat_cols, tgt_col, log_tgt=False):
    work = df[feat_cols + [tgt_col]].copy()
    for c in feat_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work[tgt_col] = pd.to_numeric(work[tgt_col], errors="coerce")
    mask = work[feat_cols].apply(np.isfinite).all(axis=1) & work[tgt_col].notna()
    work = work[mask]
    X = work[feat_cols].to_numpy()
    y = work[tgt_col].to_numpy()
    if log_tgt:
        y = np.log10(np.maximum(y, 1e-12))
    return X, y


def fit_predict(model, X_tr, y_tr, X_te, sx):
    m = clone(model)
    m.fit(sx.transform(X_tr), y_tr)
    return m, m.predict(sx.transform(X_te))


# ── A) Scatter 2×2 ────────────────────────────────────────────────────────────

def scatter_all_models(train, test, group, tgt_col, log_tgt, out_prefix):
    if log_tgt:
        ax_unit = "$\\log_{10}$ J$_{max}$"
    else:
        ax_unit = "$T_{\\Delta}$ (ч)"

    gcolor = GROUP_COLORS[group]

    for fs_label, fs_cols in FEATURE_SETS:
        X_tr, y_tr = prep_xy(train, fs_cols, tgt_col, log_tgt)
        X_te, y_te = prep_xy(test,  fs_cols, tgt_col, log_tgt)

        if len(X_te) < 3:
            print(f"  [skip {group}/{fs_label}] too few test samples ({len(X_te)})")
            continue

        if log_tgt:
            mask_te = y_te < np.log10(30000)
            X_te, y_te = X_te[mask_te], y_te[mask_te]

        if len(X_te) < 3:
            print(f"  [skip {group}/{fs_label}] too few test samples after filter")
            continue

        sx = StandardScaler().fit(X_tr)

        try:
            all_preds = [fit_predict(mdl, X_tr, y_tr, X_te, sx)[1]
                         for mdl in REG_MODELS.values()]
        except Exception as e:
            print(f"  [ERROR {group}/{fs_label}] {e}")
            continue

        vmin = min(y_te.min(), min(p.min() for p in all_preds))
        vmax = max(y_te.max(), max(p.max() for p in all_preds))
        margin = (vmax - vmin) * 0.08
        lo, hi = vmin - margin, vmax + margin

        fig, axes = plt.subplots(2, 2, figsize=(9, 8), sharex=True, sharey=True)
        fig.suptitle(
            f"{group} — «{fs_label}» · тест SC25 (n={len(y_te)})",
            fontsize=10, y=1.01, color=gcolor, fontweight="bold"
        )

        for ax, (mname, mdl), y_pred in zip(axes.flat, REG_MODELS.items(), all_preds):
            ax.scatter(y_te, y_pred, alpha=0.75, s=45,
                       color=gcolor, edgecolors="white", linewidth=0.3, zorder=3)
            ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, zorder=2)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_aspect("equal", adjustable="box")
            rmse = np.sqrt(np.mean((y_pred - y_te) ** 2))
            r2   = r2_score(y_te, y_pred) if len(y_te) >= 2 else np.nan
            cc   = _cc(y_te, y_pred)
            base = f"RMSLE={rmse:.3f}" if log_tgt else f"RMSE={rmse:.1f}h"
            metric_str = f"{base}  R²={r2:.2f}  CC={cc:.2f}"
            ax.set_title(f"{mname}  [{metric_str}]", fontsize=8.5,
                         color=MODEL_COLORS[mname], fontweight="bold", pad=4)
            ax.grid(alpha=0.2, zorder=0)
            ax.spines[["top", "right"]].set_visible(False)

        for ax in axes[1]:
            ax.set_xlabel(f"Факт ({ax_unit})", fontsize=8.5)
        for ax in axes[:, 0]:
            ax.set_ylabel(f"Прогноз ({ax_unit})", fontsize=8.5)

        plt.tight_layout()
        safe = fs_label.replace(" ", "_").replace("+", "p").replace("/", "-")
        out = PLOTS_DIR / f"{out_prefix}_{group}_{safe}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out}")


# ── B) Permutation importance ─────────────────────────────────────────────────

def importance_all_models(train, test, group, tgt_col, log_tgt, out_prefix):
    model_names = list(REG_MODELS.keys())
    n_models    = len(model_names)
    bar_h       = 0.18
    group_gap   = 0.35
    tgt_label   = "J$_{max}$" if log_tgt else "T$_{\\Delta}$"
    gcolor      = GROUP_COLORS[group]

    for fs_label, fs_cols in FEATURE_SETS:
        X_tr, y_tr = prep_xy(train, fs_cols, tgt_col, log_tgt)
        X_te, y_te = prep_xy(test,  fs_cols, tgt_col, log_tgt)

        if len(X_te) < 5:
            print(f"  [skip imp {group}/{fs_label}] too few test samples ({len(X_te)})")
            continue

        sx = StandardScaler().fit(X_tr)
        X_tr_s = sx.transform(X_tr)
        X_te_s = sx.transform(X_te)

        feat_labels = [FEAT_LABELS.get(c, c) for c in fs_cols]
        n_feats     = len(feat_labels)

        imp_matrix = {}
        for mname, mdl in REG_MODELS.items():
            try:
                m = clone(mdl)
                m.fit(X_tr_s, y_tr)
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
            except Exception as e:
                imp_matrix[mname] = np.zeros(n_feats)

        avg_imp = np.mean([imp_matrix[m] for m in model_names], axis=0)
        order   = np.argsort(avg_imp)
        feat_ord = [feat_labels[i] for i in order]

        group_size    = n_models * bar_h
        group_centers = np.arange(n_feats) * (group_size + group_gap)

        fig_h = max(2.4, n_feats * (group_size + group_gap) + 0.6)
        fig, ax = plt.subplots(figsize=(6.5, fig_h))

        x_max = 0.0
        for mi, mname in enumerate(model_names):
            vals = np.array([imp_matrix[mname][i] for i in order])
            y_positions = group_centers + (mi - (n_models - 1) / 2) * bar_h
            ax.barh(
                y_positions, vals,
                height=bar_h * 0.88,
                color=MODEL_COLORS[mname], alpha=0.90,
                label=mname, zorder=3
            )
            x_max = max(x_max, vals.max())
            for yp, v in zip(y_positions, vals):
                if v >= 1.0:
                    ax.text(v + 0.8, yp, f"{v:.0f}%",
                            va="center", ha="left", fontsize=7.5, color="#333")

        ax.set_yticks(group_centers)
        ax.set_yticklabels(feat_ord, fontsize=9.5)
        ax.set_xlabel("Вклад (%)", fontsize=9)
        ax.set_xlim(0, max(x_max * 1.22, 12))
        ax.set_ylim(group_centers[0] - group_size * 0.8,
                    group_centers[-1] + group_size * 0.8)

        ax.legend(
            handles=[plt.Rectangle((0, 0), 1, 1, color=MODEL_COLORS[m], alpha=0.9)
                     for m in model_names],
            labels=model_names,
            loc="lower right", fontsize=8.5,
            framealpha=0.85, edgecolor="#ccc",
            handlelength=1.0, handleheight=0.9
        )
        ax.set_title(
            f"[{group}] «{fs_label}» — вклад признаков\n"
            f"{tgt_label} · permutation importance · тест SC25",
            fontsize=9.5, pad=8, color=gcolor
        )
        ax.grid(axis="x", alpha=0.22, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

        plt.tight_layout()
        safe = fs_label.replace(" ", "_").replace("+", "p").replace("/", "-")
        out = PLOTS_DIR / f"{out_prefix}_{group}_{safe}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out}")


# ── C) Confusion matrix для T_delta классификации ────────────────────────────

def confusion_tdelta(train, test, group, feat_cols, fs_label):
    gcolor = GROUP_COLORS[group]

    train_td = train[train["T_delta"].notna()].copy()
    test_td  = test[test["T_delta"].notna()].copy()

    if len(test_td) < 3:
        print(f"  [skip cm_tdelta {group}] too few test samples ({len(test_td)})")
        return

    train_td["tdelta_class"] = tdelta_to_class(train_td["T_delta"].values)
    test_td["tdelta_class"]  = tdelta_to_class(test_td["T_delta"].values)

    y_col = "tdelta_class"
    work_tr = train_td[feat_cols + [y_col]].copy()
    work_te = test_td[feat_cols  + [y_col]].copy()
    for c in feat_cols:
        work_tr[c] = pd.to_numeric(work_tr[c], errors="coerce")
        work_te[c] = pd.to_numeric(work_te[c], errors="coerce")

    mask_tr = work_tr[feat_cols].apply(np.isfinite).all(axis=1) & work_tr[y_col].notna()
    mask_te = work_te[feat_cols].apply(np.isfinite).all(axis=1) & work_te[y_col].notna()
    work_tr = work_tr[mask_tr]
    work_te = work_te[mask_te]

    if len(work_te) < 3:
        print(f"  [skip cm_tdelta {group}] too few test samples after filter")
        return

    X_tr = work_tr[feat_cols].to_numpy()
    y_tr = work_tr[y_col].astype(int).to_numpy()
    X_te = work_te[feat_cols].to_numpy()
    y_te = work_te[y_col].astype(int).to_numpy()

    sx = StandardScaler()
    X_tr = sx.fit_transform(X_tr)
    X_te = sx.transform(X_te)

    try:
        mdl = RandomForestClassifier(n_estimators=200, random_state=42,
                                     class_weight="balanced")
        mdl.fit(X_tr, y_tr)
        y_pred = mdl.predict(X_te)
    except Exception as e:
        print(f"  [ERROR cm_tdelta {group}] {e}")
        return

    n_cls = len(CLF_CLASS_LABELS_TD)
    cm = confusion_matrix(y_te, y_pred, labels=list(range(n_cls)), normalize="true")
    acc = (y_te == y_pred).mean()

    fig, ax = plt.subplots(figsize=(5, 4.5))
    cmap = LinearSegmentedColormap.from_list("wgrp", ["#ffffff", gcolor])
    im = ax.imshow(cm, cmap=cmap, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(n_cls):
        for j in range(n_cls):
            clr = "white" if cm[i, j] > 0.55 else "black"
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    fontsize=11, color=clr, fontweight="bold")

    ax.set_xticks(range(n_cls))
    ax.set_xticklabels(CLF_CLASS_LABELS_TD, fontsize=9)
    ax.set_yticks(range(n_cls))
    ax.set_yticklabels(CLF_CLASS_LABELS_TD, fontsize=9)
    ax.set_xlabel("Прогноз", fontsize=10)
    ax.set_ylabel("Факт", fontsize=10)
    ax.set_title(
        f"[{group}] T_delta — {fs_label} + Forest\n"
        f"(тест SC25, n={len(y_te)},  Acc = {acc:.0%})",
        fontsize=10, color=gcolor
    )

    plt.tight_layout()
    out = PLOTS_DIR / f"cm_tdelta_{group}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── D) Compact West vs East reg bar chart ────────────────────────────────────

def compact_reg_ew(splits):
    """
    Горизонтальные сгруппированные бары: West vs East.
    2 подграфика: RMSLE Jmax | RMSE T_delta.
    Строки = наборы признаков, группы = West/East.
    """
    fs_labels = [fs for fs, _ in FEATURE_SETS]
    targets = [
        ("Jmax",    True,  "RMSLE log₁₀ (↓)"),
        ("T_delta", False, "RMSE, ч (↓)"),
    ]

    # Пересчитываем метрики на лету
    results = {g: {"Jmax": {}, "T_delta": {}} for g in ["West", "East"]}

    for group, (train, test) in splits.items():
        for tgt_col, log_tgt, _ in targets:
            for fs_label, fs_cols in FEATURE_SETS:
                X_tr, y_tr = prep_xy(train, fs_cols, tgt_col, log_tgt)
                X_te, y_te = prep_xy(test,  fs_cols, tgt_col, log_tgt)
                if len(X_tr) < 3 or len(X_te) < 2:
                    results[group][tgt_col][fs_label] = np.nan
                    continue
                sx = StandardScaler().fit(X_tr)
                best = np.inf
                for mdl in REG_MODELS.values():
                    try:
                        _, y_pred = fit_predict(mdl, X_tr, y_tr, X_te, sx)
                        rmse = np.sqrt(np.mean((y_pred - y_te) ** 2))
                        best = min(best, rmse)
                    except Exception:
                        pass
                results[group][tgt_col][fs_label] = best if best < np.inf else np.nan

    n_fs = len(fs_labels)
    bar_h = 0.35
    group_centers = np.arange(n_fs)

    fig, axes = plt.subplots(1, 2, figsize=(12, max(3.5, n_fs * 1.1)))
    fig.suptitle("Регрессия: West vs East — лучшая модель по набору признаков",
                 fontsize=11, y=1.02)

    for ax, (tgt_col, _, xlabel) in zip(axes, targets):
        for gi, group in enumerate(["West", "East"]):
            offset = (gi - 0.5) * bar_h
            vals = [results[group][tgt_col].get(fs, np.nan) for fs in fs_labels]
            ax.barh(group_centers + offset, vals, height=bar_h * 0.88,
                    color=GROUP_COLORS[group], alpha=0.85, label=group,
                    edgecolor="white", linewidth=0.4)
            for yp, v in zip(group_centers + offset, vals):
                if np.isfinite(v):
                    ax.text(v + 0.003, yp, f"{v:.2f}", va="center", fontsize=7.5)

        ax.set_yticks(group_centers)
        ax.set_yticklabels(fs_labels, fontsize=9)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.grid(axis="x", alpha=0.22, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=9, framealpha=0.85)

    plt.tight_layout()
    out = PLOTS_DIR / "compact_reg_ew.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── E) Side-by-side scatter comparison (best config per group) ────────────────

def scatter_best_ew(splits):
    """
    Один рисунок, 2 столбца (West | East) × 2 строки (Jmax | T_delta).
    Лучшая конфигурация: Jmax → «Флюэс вместо пика» + Linear,
                          T_delta → «Базовая» + Linear.
    """
    configs = [
        ("Jmax",    True,  ["helio_lon", "log_fluence", "log_cme_velocity"],
         "Флюэс вместо пика", "$\\log_{10}$ J$_{max}$", "RMSLE"),
        ("T_delta", False, ["helio_lon", "log_goes_peak_flux", "log_cme_velocity"],
         "Базовая", "$T_{\\Delta}$ (ч)", "RMSE"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    fig.suptitle("West vs East — лучшая регрессионная конфигурация (Linear)\nтест SC25",
                 fontsize=11)

    for row, (tgt_col, log_tgt, fs_cols, fs_label, ax_unit, metric_name) in enumerate(configs):
        for col, group in enumerate(["West", "East"]):
            ax = axes[row, col]
            train, test = splits[group]
            gcolor = GROUP_COLORS[group]

            X_tr, y_tr = prep_xy(train, fs_cols, tgt_col, log_tgt)
            X_te, y_te = prep_xy(test,  fs_cols, tgt_col, log_tgt)

            ax.set_title(f"{group} | {tgt_col}", fontsize=10,
                         color=gcolor, fontweight="bold")

            if len(X_te) < 2:
                ax.text(0.5, 0.5, "Нет данных", ha="center", va="center",
                        transform=ax.transAxes, fontsize=11, color="gray")
                continue

            sx = StandardScaler().fit(X_tr)
            mdl = LinearRegression()
            mdl.fit(sx.transform(X_tr), y_tr)
            y_pred = mdl.predict(sx.transform(X_te))

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
            ax.grid(alpha=0.2, zorder=0)
            ax.spines[["top", "right"]].set_visible(False)
            ax.text(0.04, 0.96, f"{metric_name}={rmse:.3f}\nn={len(y_te)}",
                    transform=ax.transAxes, va="top", ha="left",
                    fontsize=8.5, bbox=dict(boxstyle="round,pad=0.2",
                                            facecolor="white", alpha=0.8))
            if row == 1:
                ax.set_xlabel(f"Факт ({ax_unit})", fontsize=8.5)
            if col == 0:
                ax.set_ylabel(f"Прогноз ({ax_unit})", fontsize=8.5)

    plt.tight_layout()
    out = PLOTS_DIR / "scatter_best_ew.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("Загрузка данных и разбивка East/West...")
    splits = load_splits()

    for group, (train, test) in splits.items():
        print(f"\n══ {group} ══")

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

        print("  Confusion matrix T_delta (Базовая)...")
        confusion_tdelta(train, test, group,
                         feat_cols=["helio_lon", "log_goes_peak_flux", "log_cme_velocity"],
                         fs_label="Базовая")

    print("\n── Компактный регрессионный бар-чарт West vs East ──")
    compact_reg_ew(splits)

    print("\n── Scatter лучшей конфигурации West vs East ──")
    scatter_best_ew(splits)

    print("\nГотово. Графики сохранены в:", PLOTS_DIR)


if __name__ == "__main__":
    main()
