"""
compare_weights_summary.py
==========================
Сравнение трёх подходов:
  - Baseline  (results_simple/simple_results.xlsx)
  - Target weighting  (results_weighted/target_weighted_results.xlsx)
  - Density weighting (results_weighted/density_weighted_results.xlsx)

Для каждого набора признаков берётся лучшая модель по test_primary.
Delta = weighted - baseline (отрицательная delta = улучшение для RMSLE/RMSE).

Выходные файлы:
  results_weighted/plots/comparison_jmax.png
  results_weighted/plots/comparison_tdelta.png
  results_weighted/plots/delta_heatmap.png

Запуск: python compare_weights_summary.py
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).parent
PLOTS_DIR = ROOT / "results_weighted" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

APPROACH_COLORS = {
    "Baseline":       "#607D8B",
    "Target weight":  "#FF5722",
    "Density weight": "#2196F3",
}

FEATURE_SETS = [
    "Базовая",
    "Флюэс вместо пика",
    "Обе координаты",
    "Координаты+флюэс",
]


# ── Загрузка результатов ──────────────────────────────────────────────────────

def load_best(xlsx_path, sheet, metric_col="test_primary"):
    """Читает xlsx, возвращает лучшую метрику per feature_set."""
    if not Path(xlsx_path).exists():
        return {}
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    result = {}
    for fs in FEATURE_SETS:
        sub = df[df["feature_set"] == fs].dropna(subset=[metric_col])
        if sub.empty:
            result[fs] = np.nan
        else:
            result[fs] = sub[metric_col].min()
    return result


def load_all():
    base_jmax    = load_best(ROOT / "results_simple"  / "simple_results.xlsx",        "regression",   )
    base_tdelta  = load_best(ROOT / "results_simple"  / "simple_results.xlsx",        "regression",   )

    # Базовый: фильтруем по target внутри regression sheet
    def load_base_target(tgt):
        p = ROOT / "results_simple" / "simple_results.xlsx"
        if not p.exists():
            return {}
        df = pd.read_excel(p, sheet_name="regression")
        df = df[df["target"] == tgt]
        result = {}
        for fs in FEATURE_SETS:
            sub = df[df["feature_set"] == fs].dropna(subset=["test_primary"])
            result[fs] = sub["test_primary"].min() if not sub.empty else np.nan
        return result

    data = {
        "Jmax": {
            "Baseline":       load_base_target("Jmax"),
            "Target weight":  load_best(ROOT / "results_weighted" / "target_weighted_results.xlsx",  "jmax_reg"),
            "Density weight": load_best(ROOT / "results_weighted" / "density_weighted_results.xlsx", "jmax_reg"),
        },
        "T_delta": {
            "Baseline":       load_base_target("T_delta"),
            "Target weight":  load_best(ROOT / "results_weighted" / "target_weighted_results.xlsx",  "tdelta_reg"),
            "Density weight": load_best(ROOT / "results_weighted" / "density_weighted_results.xlsx", "tdelta_reg"),
        },
    }
    return data


# ── Bar chart ─────────────────────────────────────────────────────────────────

def plot_comparison(data_dict, target_label, metric_label, out_path):
    approaches = list(APPROACH_COLORS.keys())
    n_fs = len(FEATURE_SETS)
    n_ap = len(approaches)
    bar_h = 0.22
    group_centers = np.arange(n_fs)

    fig, ax = plt.subplots(figsize=(9, max(3.5, n_fs * 1.3)))

    for ai, ap in enumerate(approaches):
        offset = (ai - (n_ap - 1) / 2) * bar_h
        vals = [data_dict[ap].get(fs, np.nan) for fs in FEATURE_SETS]
        y_pos = group_centers + offset
        bars = ax.barh(y_pos, vals, height=bar_h * 0.88,
                       color=APPROACH_COLORS[ap], alpha=0.85,
                       label=ap, edgecolor="white", linewidth=0.4)
        for yp, v in zip(y_pos, vals):
            if np.isfinite(v):
                ax.text(v + 0.003, yp, f"{v:.3f}", va="center", fontsize=7.5)

    ax.set_yticks(group_centers)
    ax.set_yticklabels(FEATURE_SETS, fontsize=9)
    ax.set_xlabel(f"{metric_label} (тест SC25, ↓ лучше)", fontsize=9)
    ax.set_title(f"{target_label}: Baseline vs Target weight vs Density weight",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.85)
    ax.grid(axis="x", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ── Delta heatmap ─────────────────────────────────────────────────────────────

def plot_delta_heatmap(data, out_path):
    """
    Тепловая карта: строки = feature sets × targets,
    столбцы = Target weight delta, Density weight delta.
    Delta = weighted - baseline (< 0 = улучшение).
    """
    row_labels = []
    col_labels = ["Target weight Δ", "Density weight Δ"]
    matrix = []

    for tgt, metric in [("Jmax", "RMSLE"), ("T_delta", "RMSE ч")]:
        for fs in FEATURE_SETS:
            base = data[tgt]["Baseline"].get(fs, np.nan)
            tw   = data[tgt]["Target weight"].get(fs, np.nan)
            dw   = data[tgt]["Density weight"].get(fs, np.nan)
            matrix.append([tw - base, dw - base])
            row_labels.append(f"{tgt} / {fs}")

    mat = np.array(matrix)
    vabs = np.nanmax(np.abs(mat))
    vabs = max(vabs, 0.01)

    fig, ax = plt.subplots(figsize=(6, max(4, len(row_labels) * 0.55)))
    cmap = plt.cm.RdYlGn_r   # красный = ухудшение, зелёный = улучшение (reversed: neg=green)
    # Flip: delta < 0 = улучшение → хотим зелёный для отрицательного
    im = ax.imshow(mat, cmap="RdYlGn", vmin=-vabs, vmax=vabs, aspect="auto")
    plt.colorbar(im, ax=ax, label="Δ (weighted − baseline)")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title("Δ = weighted − baseline  (зелёный = улучшение, красный = ухудшение)",
                 fontsize=9, pad=8)

    for i in range(len(row_labels)):
        for j in range(2):
            v = mat[i, j]
            if np.isfinite(v):
                clr = "black" if abs(v) < vabs * 0.6 else "white"
                sign = "+" if v > 0 else ""
                ax.text(j, i, f"{sign}{v:.3f}", ha="center", va="center",
                        fontsize=7.5, color=clr, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ── Текстовая сводка ──────────────────────────────────────────────────────────

def print_summary(data):
    for tgt, metric in [("Jmax", "RMSLE log₁₀"), ("T_delta", "RMSE ч")]:
        print(f"\n── {tgt} ({metric}) — тест SC25 ──")
        print(f"  {'Набор признаков':<26}  {'Baseline':>9}  {'Tgt-W':>9}  {'Δ':>7}  "
              f"{'Den-W':>9}  {'Δ':>7}")
        print(f"  {'-'*26}  {'-'*9}  {'-'*9}  {'-'*7}  {'-'*9}  {'-'*7}")
        for fs in FEATURE_SETS:
            base = data[tgt]["Baseline"].get(fs, np.nan)
            tw   = data[tgt]["Target weight"].get(fs, np.nan)
            dw   = data[tgt]["Density weight"].get(fs, np.nan)
            d_tw = tw - base if np.isfinite(tw) and np.isfinite(base) else np.nan
            d_dw = dw - base if np.isfinite(dw) and np.isfinite(base) else np.nan

            def fmt(v):
                return f"{v:9.3f}" if np.isfinite(v) else "        —"
            def fmtd(v):
                if not np.isfinite(v): return "      —"
                return f"{v:+7.3f}"

            print(f"  {fs:<26}  {fmt(base)}  {fmt(tw)}  {fmtd(d_tw)}  {fmt(dw)}  {fmtd(d_dw)}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # Проверяем наличие файлов
    missing = []
    for p in [
        ROOT / "results_simple"  / "simple_results.xlsx",
        ROOT / "results_weighted" / "target_weighted_results.xlsx",
        ROOT / "results_weighted" / "density_weighted_results.xlsx",
    ]:
        if not p.exists():
            missing.append(str(p))
    if missing:
        print("Отсутствуют файлы результатов:")
        for m in missing:
            print(f"  {m}")
        print("Запустите сначала compare_simple.py, compare_weighted_target.py,"
              " compare_weighted_density.py")
        return

    print("Загрузка результатов...")
    data = load_all()

    print("\nПостроение графиков...")
    plot_comparison(data["Jmax"],    "J_max",   "RMSLE log₁₀",
                    PLOTS_DIR / "comparison_jmax.png")
    plot_comparison(data["T_delta"], "T_delta", "RMSE (часы)",
                    PLOTS_DIR / "comparison_tdelta.png")
    plot_delta_heatmap(data, PLOTS_DIR / "delta_heatmap.png")

    print_summary(data)
    print("\nГотово.")


if __name__ == "__main__":
    main()
