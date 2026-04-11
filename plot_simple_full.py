"""
Генерация графиков для REPORT_SIMPLE.md:
  A) Scatter-графики: для каждого набора признаков — 2×2 подграфика (все 4 модели)
     — по 4 таких фигуры для J_max и 4 для T_delta
  B) Графики вклада (permutation importance): для каждого набора признаков — 2×2 подграфика
     — по 4 таких фигуры для J_max и 4 для T_delta

Запуск: python plot_simple_full.py
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

from spe_utils import build_features, COL_CYCLE

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).parent
PLOTS_DIR = ROOT / "results_simple" / "plots"
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

FEAT_LABELS = {
    "helio_lon":          "Гелиодолгота",
    "helio_lat":          "Гелиоширота",
    "log_goes_peak_flux": "log(GOES пик)",
    "log_cme_velocity":   "log(Скорость КВМ)",
    "log_fluence":        "log(Флюэнс)",
}


# ── Загрузка данных ───────────────────────────────────────────────────────────

def load():
    df = build_features(
        pd.read_excel(ROOT / "data" / "ОБЪЕДИНЕННЫЙ КАТАЛОГ СПС 23-25.xlsx",
                      sheet_name="Флюэс GOES")
    )
    cycle = pd.to_numeric(df[COL_CYCLE], errors="coerce")
    train = df[cycle.isin([23, 24]) & (df["Jmax"].fillna(0) >= 10)].copy()
    test  = df[cycle.isin([25])     & (df["Jmax"].fillna(0) >= 10)].copy()
    return train, test


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
    from sklearn.base import clone
    m = clone(model)
    m.fit(sx.transform(X_tr), y_tr)
    return m, m.predict(sx.transform(X_te))


# ── A) Scatter-фигуры ─────────────────────────────────────────────────────────

def scatter_all_models(train, test, tgt_col, log_tgt, out_prefix):
    """
    Для каждого feature set → 1 фигура, 2×2 подграфика (по одному на модель).
    """
    if log_tgt:
        ax_unit  = f"$\\log_{{10}}$ J_max"
        metric_f = lambda yt, yp: f"RMSLE={np.sqrt(np.mean((yp-yt)**2)):.3f}"
    else:
        ax_unit  = "T_delta (ч)"
        metric_f = lambda yt, yp: f"RMSE={np.sqrt(np.mean((yp-yt)**2)):.1f}h"

    for fs_label, fs_cols in FEATURE_SETS:
        X_tr, y_tr = prep_xy(train, fs_cols, tgt_col, log_tgt)
        X_te, y_te = prep_xy(test,  fs_cols, tgt_col, log_tgt)

        # Убираем выброс с максимальной интенсивностью (Jmax > 30000 pfu)
        if log_tgt:
            mask_te = y_te < np.log10(30000)
            X_te, y_te = X_te[mask_te], y_te[mask_te]

        sx = StandardScaler().fit(X_tr)

        # Диапазон осей — только по тестовым данным (выброс уже убран)
        all_preds_tmp = [fit_predict(mdl, X_tr, y_tr, X_te, sx)[1]
                         for mdl in REG_MODELS.values()]
        vmin = min(y_te.min(), min(p.min() for p in all_preds_tmp))
        vmax = max(y_te.max(), max(p.max() for p in all_preds_tmp))
        margin = (vmax - vmin) * 0.08

        fig, axes = plt.subplots(2, 2, figsize=(9, 8),
                                 sharex=True, sharey=True)

        lo, hi = vmin - margin, vmax + margin
        for ax, (mname, mdl), y_pred in zip(axes.flat, REG_MODELS.items(), all_preds_tmp):
            ax.scatter(y_te, y_pred, alpha=0.75, s=45,
                       color=MODEL_COLORS[mname], edgecolors="white", linewidth=0.3, zorder=3)
            ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, zorder=2)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_aspect("equal", adjustable="box")
            # Подпись модели — легенда вместо заголовка
            ax.text(0.97, 0.05, mname, transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=10,
                    fontweight="bold", color=MODEL_COLORS[mname])
            ax.grid(alpha=0.25)

        fig.text(0.5, 0.02, f"Факт ({ax_unit})", ha="center", fontsize=10)
        fig.text(0.02, 0.5, f"Прогноз ({ax_unit})", va="center",
                 rotation="vertical", fontsize=10)

        plt.tight_layout(rect=[0.04, 0.05, 1, 1])
        safe = fs_label.replace(" ", "_").replace("+", "p").replace("/", "-")
        out = PLOTS_DIR / f"{out_prefix}_{safe}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out}")


# ── B) Importance-фигуры (компактный сгруппированный bar chart) ───────────────

def importance_all_models(train, test, tgt_col, log_tgt, out_prefix):
    """
    Для каждого feature set → 1 компактная фигура.
    Сгруппированный горизонтальный bar chart: 4 модели × N признаков.
    """
    from sklearn.base import clone

    model_names = list(REG_MODELS.keys())
    n_models    = len(model_names)
    bar_h       = 0.18          # высота одного бара
    group_gap   = 0.35          # отступ между группами признаков

    tgt_label = "J_max" if log_tgt else "T_delta"

    for fs_label, fs_cols in FEATURE_SETS:
        X_tr, y_tr = prep_xy(train, fs_cols, tgt_col, log_tgt)
        X_te, y_te = prep_xy(test,  fs_cols, tgt_col, log_tgt)

        if len(X_te) < 5:
            print(f"  [skip {fs_label}] too few test samples")
            continue

        sx = StandardScaler().fit(X_tr)
        X_tr_s = sx.transform(X_tr)
        X_te_s = sx.transform(X_te)

        feat_labels = [FEAT_LABELS.get(c, c) for c in fs_cols]
        n_feats     = len(feat_labels)

        # Собираем importance для каждой модели
        imp_matrix = {}   # mname → array[n_feats] в %
        for mname, mdl in REG_MODELS.items():
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
                else (np.abs(pi.importances_mean) / (np.abs(pi.importances_mean).sum() + 1e-12) * 100)

        # Сортируем признаки по среднему вкладу (desc → снизу вверх на barh)
        avg_imp = np.mean([imp_matrix[m] for m in model_names], axis=0)
        order   = np.argsort(avg_imp)          # от меньшего к большему → вверх
        feat_ord = [feat_labels[i] for i in order]

        # Позиции: каждый признак — группа из n_models баров
        group_size = n_models * bar_h
        group_centers = np.arange(n_feats) * (group_size + group_gap)

        fig_h = max(2.4, n_feats * (group_size + group_gap) + 0.6)
        fig, ax = plt.subplots(figsize=(6.5, fig_h))

        x_max = 0.0
        for mi, mname in enumerate(model_names):
            vals = np.array([imp_matrix[mname][i] for i in order])
            y_positions = group_centers + (mi - (n_models - 1) / 2) * bar_h
            bars = ax.barh(
                y_positions, vals,
                height=bar_h * 0.88,
                color=MODEL_COLORS[mname], alpha=0.90,
                label=mname, zorder=3
            )
            x_max = max(x_max, vals.max())
            # Подписи значений
            for yp, v in zip(y_positions, vals):
                if v >= 1.0:
                    ax.text(v + 0.8, yp, f"{v:.0f}%",
                            va="center", ha="left", fontsize=7.5, color="#333")

        # Y-тики по центрам групп
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
            f"Вклад признаков — «{fs_label}»\n"
            f"{tgt_label} · permutation importance · тест SC25",
            fontsize=9.5, pad=8
        )
        ax.grid(axis="x", alpha=0.22, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

        plt.tight_layout()
        safe = fs_label.replace(" ", "_").replace("+", "p").replace("/", "-")
        out = PLOTS_DIR / f"{out_prefix}_{safe}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out}")


# ── C) Compact scatter (2×2, общие оси, компактный стиль) ────────────────────

def scatter_compact(train, test, tgt_col, log_tgt, out_prefix):
    """
    Для каждого feature set → 1 фигура, 2×2 подграфика с общими осями.
    Каждый подграфик — одна модель; общий диапазон осей, метрики внутри.
    """
    if log_tgt:
        ax_unit  = "$\\log_{10}$ J$_{max}$"
        rmse_lbl = "RMSLE"
    else:
        ax_unit  = "$T_{\\Delta}$ (ч)"
        rmse_lbl = "RMSE"

    tgt_label = "J$_{max}$" if log_tgt else "$T_{\\Delta}$"
    rmse_fn   = lambda yt, yp: np.sqrt(np.mean((yp - yt) ** 2))

    for fs_label, fs_cols in FEATURE_SETS:
        X_tr, y_tr = prep_xy(train, fs_cols, tgt_col, log_tgt)
        X_te, y_te = prep_xy(test,  fs_cols, tgt_col, log_tgt)
        if len(X_te) < 5:
            continue

        # Убираем выброс с максимальной интенсивностью (Jmax > 30000 pfu)
        if log_tgt:
            mask_te = y_te < np.log10(30000)
            X_te, y_te = X_te[mask_te], y_te[mask_te]

        sx = StandardScaler().fit(X_tr)

        # Общий диапазон по всем моделям
        all_preds = [fit_predict(mdl, X_tr, y_tr, X_te, sx)[1]
                     for mdl in REG_MODELS.values()]
        vmin = min(y_te.min(), min(p.min() for p in all_preds))
        vmax = max(y_te.max(), max(p.max() for p in all_preds))
        margin = (vmax - vmin) * 0.07
        lo, hi = vmin - margin, vmax + margin

        fig, axes = plt.subplots(2, 2, figsize=(6.5, 6.0),
                                 sharex=True, sharey=True)
        fig.suptitle(
            f"{tgt_label} — «{fs_label}» · тест SC25 (n={len(y_te)})",
            fontsize=10, y=1.01
        )

        for ax, (mname, mdl), y_pred in zip(axes.flat, REG_MODELS.items(), all_preds):
            ax.scatter(y_te, y_pred, s=32, alpha=0.78,
                       color=MODEL_COLORS[mname],
                       edgecolors="white", linewidth=0.25, zorder=3)
            ax.plot([lo, hi], [lo, hi], color="#666", lw=0.9, ls="--", zorder=2)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(mname, fontsize=9, color=MODEL_COLORS[mname],
                         fontweight="bold", pad=4)
            ax.grid(alpha=0.2, zorder=0)
            ax.spines[["top", "right"]].set_visible(False)

        # Общие подписи осей
        for ax in axes[1]:
            ax.set_xlabel(f"Факт ({ax_unit})", fontsize=8.5)
        for ax in axes[:, 0]:
            ax.set_ylabel(f"Прогноз ({ax_unit})", fontsize=8.5)

        plt.tight_layout()
        safe = fs_label.replace(" ", "_").replace("+", "p").replace("/", "-")
        out = PLOTS_DIR / f"{out_prefix}_{safe}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out}")


# ── D) Compact regression bar chart (Jmax + T_delta на одном) ─────────────────

def reg_compact():
    """
    Один рисунок, 2 подграфика: RMSLE J_max | RMSE T_delta.
    Горизонтальные сгруппированные бары (модели × наборы признаков).
    """
    xlsx = ROOT / "results_simple" / "simple_results.xlsx"
    df = pd.read_excel(xlsx, sheet_name="regression")

    fs_labels  = list(dict.fromkeys(df["feature_set"]))
    model_list = list(dict.fromkeys(df["model"]))
    n_fs = len(fs_labels)
    n_m  = len(model_list)
    bar_h = 0.8 / n_m

    targets = [
        ("Jmax",    "RMSLE (тест SC25)"),
        ("T_delta", "RMSE, ч (тест SC25)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, max(3.0, n_fs * 1.5)))
    fig.suptitle("Регрессия: J$_{max}$ и T$_{\\Delta}$", fontsize=11, y=1.02)

    for ax, (tgt, xlabel) in zip(axes, targets):
        sub = df[df["target"] == tgt]
        group_centers = np.arange(n_fs)

        for mi, model in enumerate(model_list):
            offset = (mi - (n_m - 1) / 2) * bar_h
            vals = []
            for fs in fs_labels:
                row = sub[(sub["feature_set"] == fs) & (sub["model"] == model)]
                vals.append(row["test_primary"].values[0] if not row.empty else np.nan)

            y_pos = group_centers + offset
            color = MODEL_COLORS.get(model, "#888")
            ax.barh(y_pos, vals, height=bar_h * 0.88,
                    color=color, alpha=0.88,
                    edgecolor="white", linewidth=0.4)

            for yp, v in zip(y_pos, vals):
                if np.isfinite(v):
                    ax.text(v + 0.005, yp, f"{v:.2f}", va="center",
                            fontsize=7.5)

        ax.set_yticks(group_centers)
        ax.set_yticklabels(fs_labels, fontsize=9)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.grid(axis="x", alpha=0.22, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fig.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, color=MODEL_COLORS[m], alpha=0.88)
                 for m in model_list],
        labels=model_list,
        loc="lower center", ncol=n_m, fontsize=9,
        framealpha=0.85, edgecolor="#ccc",
        handlelength=1.0, handleheight=0.9,
        bbox_to_anchor=(0.5, -0.06)
    )
    out = PLOTS_DIR / "compact_reg.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── E) Compact classification bar charts ──────────────────────────────────────

CLF_COLOR = {
    "LogReg":   "#1f77b4",   # как Linear
    "Forest":   "#8c564b",
    "Boosting": "#ff7f0e",
    "SVC":      "#2ca02c",   # как SVR
}

def clf_compact(pipeline_name, title_label, out_prefix):
    """
    Читает results xlsx → 2 горизонтальных subplot: Log Loss | AUC.
    Строки = feature sets, сгруппированные по моделям.
    """
    xlsx = ROOT / "results_simple" / "simple_results.xlsx"
    df = pd.read_excel(xlsx, sheet_name=pipeline_name)

    fs_labels = list(dict.fromkeys(df["feature_set"]))   # сохраняем порядок
    model_list = list(dict.fromkeys(df["model"]))

    metrics = [
        ("test_primary", "Log Loss (тест SC25)", True),   # lower is better
        ("test_auc",     "AUC macro (тест SC25)", False),  # higher is better
    ]

    n_fs  = len(fs_labels)
    n_m   = len(model_list)
    bar_h = 0.8 / n_m

    fig, axes = plt.subplots(1, 2, figsize=(11, max(3.0, n_fs * 1.5)))
    fig.suptitle(f"Классификация {title_label}", fontsize=11, y=1.02)

    for ax, (col, ylabel, lower_better) in zip(axes, metrics):
        group_centers = np.arange(n_fs)

        for mi, model in enumerate(model_list):
            offset = (mi - (n_m - 1) / 2) * bar_h
            vals = []
            for fs in fs_labels:
                row = df[(df["feature_set"] == fs) & (df["model"] == model)]
                vals.append(row[col].values[0] if not row.empty else np.nan)

            y_pos = group_centers + offset
            color = CLF_COLOR.get(model, "#888")
            ax.barh(y_pos, vals, height=bar_h * 0.88,
                    color=color, alpha=0.88,
                    edgecolor="white", linewidth=0.4)

            for yp, v in zip(y_pos, vals):
                if np.isfinite(v):
                    ax.text(v + 0.005, yp, f"{v:.3f}", va="center",
                            fontsize=7.5)

        ax.set_yticks(group_centers)
        ax.set_yticklabels(fs_labels, fontsize=9)
        ax.set_xlabel(ylabel, fontsize=9)
        ax.grid(axis="x", alpha=0.22, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

        # Best annotation
        best_vals = []
        for fs in fs_labels:
            sub = df[df["feature_set"] == fs]
            if lower_better:
                best_vals.append(sub[col].min())
            else:
                best_vals.append(sub[col].max())

    # Единая легенда
    axes[0].legend(
        handles=[plt.Rectangle((0, 0), 1, 1, color=CLF_COLOR[m], alpha=0.88)
                 for m in model_list],
        labels=model_list,
        loc="lower right", fontsize=8.5,
        framealpha=0.85, edgecolor="#ccc",
        handlelength=1.0, handleheight=0.9
    )

    plt.tight_layout()
    out = PLOTS_DIR / f"{out_prefix}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("Загрузка данных...")
    train, test = load()

    print("\n── Scatter: J_max регрессия (4 набора × 4 модели) ──")
    scatter_all_models(train, test, "Jmax",    log_tgt=True,  out_prefix="scatter4_jmax")

    print("\n── Scatter: T_delta регрессия (4 набора × 4 модели) ──")
    scatter_all_models(train, test, "T_delta", log_tgt=False, out_prefix="scatter4_tdelta")

    print("\n── Вклад признаков: J_max (4 набора × 4 модели) ──")
    importance_all_models(train, test, "Jmax",    log_tgt=True,  out_prefix="imp_jmax")

    print("\n── Вклад признаков: T_delta (4 набора × 4 модели) ──")
    importance_all_models(train, test, "T_delta", log_tgt=False, out_prefix="imp_tdelta")

    print("\n── Compact scatter: J_max ──")
    scatter_compact(train, test, "Jmax",    log_tgt=True,  out_prefix="compact_jmax")

    print("\n── Compact scatter: T_delta ──")
    scatter_compact(train, test, "T_delta", log_tgt=False, out_prefix="compact_tdelta")

    print("\n── Compact reg: J_max + T_delta ──")
    reg_compact()

    print("\n── Compact clf: J_max S-class ──")
    clf_compact("jmax_clf",  "J$_{max}$ (S-класс)", "compact_clf_jmax")

    print("\n── Compact clf: T_delta ──")
    clf_compact("tdelta_clf", "T$_{\\Delta}$ (Быстрые/Умеренные/Медленные)", "compact_clf_tdelta")

    print("\nГотово.")


if __name__ == "__main__":
    main()
