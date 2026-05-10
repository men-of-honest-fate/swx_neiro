"""
Генерация scatter-графиков для REPORT_SIMPLE.md.
Лучшие конфигурации:
  - J_max регрессия:    «Флюэс вместо пика» + Linear
  - T_delta регрессия:  «Базовая»           + Linear
  - J_max классификация: «Координаты+флюэс» + Forest  (confusion matrix)
  - T_delta классификация: «Базовая»         + Forest  (confusion matrix)

Запуск: python plot_scatter_simple.py
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from spe_utils import build_features, COL_CYCLE
from pipelines.classification.utils import jmax_to_class
from pipelines.tdelta_clf.utils import tdelta_to_class, CLASS_SHORT as TD_CLASS_SHORT

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).parent
PLOTS_DIR = ROOT / "results_simple" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Загрузка данных ───────────────────────────────────────────────────────────

def load():
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
    train = df[cycle.isin([23, 24]) & mask].copy()
    test  = df[cycle.isin([25])     & mask].copy()
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
        y = np.log10(y)
    return X, y


# ── Scatter: регрессия ────────────────────────────────────────────────────────

def scatter_regression(train, test, feat_cols, tgt_col, log_tgt,
                       fs_label, out_path):
    X_tr, y_tr = prep_xy(train, feat_cols, tgt_col, log_tgt)
    X_te, y_te = prep_xy(test,  feat_cols, tgt_col, log_tgt)

    sx = StandardScaler()
    X_tr_s = sx.fit_transform(X_tr)
    X_te_s = sx.transform(X_te)

    mdl = LinearRegression()
    mdl.fit(X_tr_s, y_tr)
    y_pred = mdl.predict(X_te_s)

    rho, _ = spearmanr(y_te, y_pred)
    r2 = 1 - np.sum((y_te - y_pred)**2) / np.sum((y_te - np.mean(y_te))**2)
    cc = (float(np.corrcoef(y_te, y_pred)[0, 1])
          if len(y_te) >= 2 and np.std(y_pred) > 0 and np.std(y_te) > 0
          else np.nan)

    if log_tgt:
        rmse_metric = np.sqrt(np.mean((y_pred - y_te)**2))
        metric_label = (f"RMSLE = {rmse_metric:.3f}\n"
                        f"$R^2_{{\\log}}$ = {r2:.2f},  CC = {cc:.2f},  $\\rho_s$ = {rho:.2f}")
        ax_label = "$\\log_{{10}}$ J_max"
        title = f"J_max регрессия — {fs_label} + Linear\n(тест SC25, n={len(y_te)})"
    else:
        rmse_metric = np.sqrt(np.mean((y_pred - y_te)**2))
        metric_label = (f"RMSE = {rmse_metric:.1f} ч\n"
                        f"$R^2$ = {r2:.2f},  CC = {cc:.2f},  $\\rho_s$ = {rho:.2f}")
        ax_label = "T_delta (ч)"
        title = f"T_delta регрессия — {fs_label} + Linear\n(тест SC25, n={len(y_te)})"

    vmin = min(y_te.min(), y_pred.min())
    vmax = max(y_te.max(), y_pred.max())
    margin = (vmax - vmin) * 0.07

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(y_te, y_pred, alpha=0.75, edgecolors="white",
               linewidth=0.4, s=55, color="#2c7bb6", zorder=3)
    ax.plot([vmin - margin, vmax + margin],
            [vmin - margin, vmax + margin],
            "k--", lw=1.2, zorder=2, label="идеал")

    ax.set_xlim(vmin - margin, vmax + margin)
    ax.set_ylim(vmin - margin, vmax + margin)
    ax.set_xlabel(f"Факт  ({ax_label})", fontsize=11)
    ax.set_ylabel(f"Прогноз  ({ax_label})", fontsize=11)
    ax.set_title(title, fontsize=10)
    ax.text(0.04, 0.96, metric_label,
            transform=ax.transAxes, va="top", ha="left",
            fontsize=9, bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="white", alpha=0.8))
    ax.grid(alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ── Confusion matrix: классификация ──────────────────────────────────────────

CLF_CLASS_LABELS_JMAX = ["S1–S2\n(10–100)", "S3\n(100–1k)", "S4–S5\n(≥1k)"]
CLF_CLASS_LABELS_TD   = ["Быстрые\n(< 8 ч)", "Умеренные\n(8–20 ч)", "Медленные\n(≥ 20 ч)"]


def confusion_clf(train, test, feat_cols, y_col, class_fn, class_labels,
                  fs_label, task_label, out_path):
    train = train.copy()
    test  = test.copy()
    train[y_col] = class_fn(train[y_col].values)
    test[y_col]  = class_fn(test[y_col].values)

    work_tr = train[feat_cols + [y_col]].copy()
    work_te = test[feat_cols  + [y_col]].copy()
    for c in feat_cols:
        work_tr[c] = pd.to_numeric(work_tr[c], errors="coerce")
        work_te[c] = pd.to_numeric(work_te[c], errors="coerce")
    mask_tr = work_tr[feat_cols].apply(np.isfinite).all(axis=1) & work_tr[y_col].notna()
    mask_te = work_te[feat_cols].apply(np.isfinite).all(axis=1) & work_te[y_col].notna()
    work_tr = work_tr[mask_tr]
    work_te = work_te[mask_te]

    X_tr = work_tr[feat_cols].to_numpy()
    y_tr = work_tr[y_col].astype(int).to_numpy()
    X_te = work_te[feat_cols].to_numpy()
    y_te = work_te[y_col].astype(int).to_numpy()

    sx = StandardScaler()
    X_tr = sx.fit_transform(X_tr)
    X_te = sx.transform(X_te)

    mdl = RandomForestClassifier(n_estimators=200, random_state=42,
                                 class_weight="balanced")
    mdl.fit(X_tr, y_tr)
    y_pred = mdl.predict(X_te)

    n_cls = len(class_labels)
    cm = confusion_matrix(y_te, y_pred, labels=list(range(n_cls)), normalize="true")
    acc = (y_te == y_pred).mean()

    fig, ax = plt.subplots(figsize=(5, 4.5))
    cmap = LinearSegmentedColormap.from_list("wblue", ["#ffffff", "#2c7bb6"])
    im = ax.imshow(cm, cmap=cmap, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(n_cls):
        for j in range(n_cls):
            clr = "white" if cm[i, j] > 0.55 else "black"
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    fontsize=11, color=clr, fontweight="bold")

    ax.set_xticks(range(n_cls))
    ax.set_xticklabels(class_labels, fontsize=9)
    ax.set_yticks(range(n_cls))
    ax.set_yticklabels(class_labels, fontsize=9)
    ax.set_xlabel("Прогноз", fontsize=10)
    ax.set_ylabel("Факт", fontsize=10)
    ax.set_title(
        f"{task_label} — {fs_label} + Forest\n"
        f"(тест SC25, n={len(y_te)},  Acc = {acc:.0%})",
        fontsize=10
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("Загрузка данных...")
    train, test = load()

    print("\nRegression scatter plots...")
    scatter_regression(
        train, test,
        feat_cols=["helio_lon", "log_fluence", "log_cme_velocity"],
        tgt_col="Jmax", log_tgt=True,
        fs_label="Флюэс вместо пика",
        out_path=PLOTS_DIR / "scatter_jmax_reg.png",
    )
    scatter_regression(
        train, test,
        feat_cols=["helio_lon", "log_goes_peak_flux", "log_cme_velocity"],
        tgt_col="T_delta", log_tgt=False,
        fs_label="Базовая",
        out_path=PLOTS_DIR / "scatter_tdelta_reg.png",
    )

    print("\nConfusion matrix plots...")
    # J_max clf: Координаты+флюэс + Forest
    train_jmax = train[train["Jmax"].notna()].copy()
    test_jmax  = test[test["Jmax"].notna()].copy()
    confusion_clf(
        train_jmax, test_jmax,
        feat_cols=["helio_lon", "helio_lat", "log_fluence", "log_cme_velocity"],
        y_col="Jmax",
        class_fn=jmax_to_class,
        class_labels=CLF_CLASS_LABELS_JMAX,
        fs_label="Координаты + флюэс",
        task_label="J_max S-класс",
        out_path=PLOTS_DIR / "cm_jmax_clf.png",
    )

    # T_delta clf: Базовая + Forest
    train_td = train[train["T_delta"].notna()].copy()
    test_td  = test[test["T_delta"].notna()].copy()

    def _td_class_fn(arr):
        return tdelta_to_class(arr)

    confusion_clf(
        train_td, test_td,
        feat_cols=["helio_lon", "log_goes_peak_flux", "log_cme_velocity"],
        y_col="T_delta",
        class_fn=_td_class_fn,
        class_labels=CLF_CLASS_LABELS_TD,
        fs_label="Базовая",
        task_label="T_delta классификация",
        out_path=PLOTS_DIR / "cm_tdelta_clf.png",
    )

    print("\nГотово.")


if __name__ == "__main__":
    main()
