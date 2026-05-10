"""
compare_ew_weights_summary.py
==============================
Сравнение для West / East:
  - Baseline EW   (results_ew/ew_results.xlsx)
  - Target weight (results_ew_weighted/target_weighted_ew_results.xlsx)
  - Density weight(results_ew_weighted/density_weighted_ew_results.xlsx)

Выходные файлы: results_ew_weighted/plots/
  comparison_{group}_{target}.png — bar chart
  delta_heatmap_ew.png            — тепловая карта дельт

Запуск: python compare_ew_weights_summary.py
"""

import sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).parent
PLOTS_DIR = ROOT / "results_ew_weighted" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

GROUPS   = ["West", "East"]
TARGETS  = [("Jmax", "RMSLE log₁₀"), ("T_delta", "RMSE ч")]
FEATURE_SETS = ["Базовая", "Флюэс вместо пика", "Обе координаты", "Координаты+флюэс"]

APPROACH_COLORS = {
    "Baseline":       "#607D8B",
    "Target weight":  "#FF5722",
    "Density weight": "#2196F3",
}
GROUP_COLORS = {"West": "#FF5722", "East": "#2196F3"}


# ── Загрузка ──────────────────────────────────────────────────────────────────

def best_per_fs(df, group, target):
    """Лучшая test_primary per feature_set для данной группы и таргета."""
    sub = df[(df["group"] == group) & (df["target"] == target)] \
        if "group" in df.columns else df[df["target"] == target]
    result = {}
    for fs in FEATURE_SETS:
        fsub = sub[sub["feature_set"] == fs].dropna(subset=["test_primary"])
        result[fs] = fsub["test_primary"].min() if not fsub.empty else np.nan
    return result


def load_all():
    paths = {
        "Baseline":       ROOT / "results_ew"         / "ew_results.xlsx",
        "Target weight":  ROOT / "results_ew_weighted" / "target_weighted_ew_results.xlsx",
        "Density weight": ROOT / "results_ew_weighted" / "density_weighted_ew_results.xlsx",
    }
    missing = [str(v) for k, v in paths.items() if not v.exists()]
    if missing:
        print("Отсутствуют файлы:")
        for m in missing:
            print(f"  {m}")
        return None

    data = {}  # data[group][target][approach] = {fs: value}

    for group in GROUPS:
        data[group] = {}
        for tgt, _ in TARGETS:
            data[group][tgt] = {}
            for ap, path in paths.items():
                try:
                    if ap == "Baseline":
                        df = pd.read_excel(path, sheet_name="regression")
                        df = df[df["pipeline"] == "regression"]
                    else:
                        sheet = f"{group}_{tgt}"
                        df = pd.read_excel(path, sheet_name=sheet)
                        # Добавляем колонки group/target если их нет (для совместимости)
                        if "group" not in df.columns:
                            df["group"] = group
                        if "target" not in df.columns:
                            df["target"] = tgt
                    data[group][tgt][ap] = best_per_fs(df, group, tgt)
                except Exception as e:
                    print(f"  [WARN] {ap} / {group} / {tgt}: {e}")
                    data[group][tgt][ap] = {fs: np.nan for fs in FEATURE_SETS}
    return data


# ── Bar chart ─────────────────────────────────────────────────────────────────

def plot_bar(ap_data, group, target, metric_label, out_path):
    approaches = list(APPROACH_COLORS.keys())
    n_fs = len(FEATURE_SETS)
    n_ap = len(approaches)
    bar_h = 0.22
    centers = np.arange(n_fs)

    fig, ax = plt.subplots(figsize=(9, max(3.5, n_fs * 1.3)))
    for ai, ap in enumerate(approaches):
        offset = (ai - (n_ap - 1) / 2) * bar_h
        vals = [ap_data[ap].get(fs, np.nan) for fs in FEATURE_SETS]
        ax.barh(centers + offset, vals, height=bar_h * 0.88,
                color=APPROACH_COLORS[ap], alpha=0.85,
                label=ap, edgecolor="white", linewidth=0.4)
        for yp, v in zip(centers + offset, vals):
            if np.isfinite(v):
                ax.text(v + 0.003, yp, f"{v:.3f}", va="center", fontsize=7.5)

    ax.set_yticks(centers)
    ax.set_yticklabels(FEATURE_SETS, fontsize=9)
    ax.set_xlabel(f"{metric_label} (тест SC25, ↓ лучше)", fontsize=9)
    ax.set_title(f"[{group}] {target}: Baseline vs Target-W vs Density-W",
                 fontsize=10, fontweight="bold", color=GROUP_COLORS[group])
    ax.legend(fontsize=9, framealpha=0.85)
    ax.grid(axis="x", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ── Delta heatmap ─────────────────────────────────────────────────────────────

def plot_delta_heatmap(data, out_path):
    row_labels, matrix = [], []

    for group in GROUPS:
        for tgt, _ in TARGETS:
            for fs in FEATURE_SETS:
                base = data[group][tgt]["Baseline"].get(fs, np.nan)
                tw   = data[group][tgt]["Target weight"].get(fs, np.nan)
                dw   = data[group][tgt]["Density weight"].get(fs, np.nan)
                matrix.append([tw - base, dw - base])
                row_labels.append(f"{group} / {tgt} / {fs[:12]}")

    mat  = np.array(matrix)
    vabs = max(np.nanmax(np.abs(mat)), 0.01)

    fig, ax = plt.subplots(figsize=(6.5, max(5, len(row_labels) * 0.45)))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=-vabs, vmax=vabs, aspect="auto")
    plt.colorbar(im, ax=ax, label="Δ (weighted − baseline)")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Target weight Δ", "Density weight Δ"], fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7.5)
    ax.set_title("Δ = weighted − baseline\nзелёный = улучшение, красный = ухудшение",
                 fontsize=9, pad=8)

    for i in range(len(row_labels)):
        for j in range(2):
            v = mat[i, j]
            if np.isfinite(v):
                clr = "black" if abs(v) < vabs * 0.6 else "white"
                ax.text(j, i, f"{'+' if v > 0 else ''}{v:.3f}",
                        ha="center", va="center", fontsize=7, color=clr, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ── Текстовая сводка ──────────────────────────────────────────────────────────

def print_summary(data):
    ap_list = ["Baseline", "Target weight", "Density weight"]
    for group in GROUPS:
        for tgt, metric in TARGETS:
            print(f"\n── [{group}] {tgt} ({metric}) ──")
            print(f"  {'Набор признаков':<26}  {'Base':>7}  {'Tgt-W':>7}  {'Δ':>6}  "
                  f"{'Den-W':>7}  {'Δ':>6}")
            print(f"  {'-'*26}  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*7}  {'-'*6}")
            for fs in FEATURE_SETS:
                vals = {ap: data[group][tgt][ap].get(fs, np.nan) for ap in ap_list}
                base = vals["Baseline"]
                tw   = vals["Target weight"]
                dw   = vals["Density weight"]
                d_tw = tw - base if np.isfinite(tw) and np.isfinite(base) else np.nan
                d_dw = dw - base if np.isfinite(dw) and np.isfinite(base) else np.nan
                f = lambda v: f"{v:7.3f}" if np.isfinite(v) else "      —"
                fd = lambda v: f"{v:+6.3f}" if np.isfinite(v) else "     —"
                print(f"  {fs:<26}  {f(base)}  {f(tw)}  {fd(d_tw)}  {f(dw)}  {fd(d_dw)}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    data = load_all()
    if data is None:
        return

    print("\nПостроение графиков...")
    for group in GROUPS:
        for tgt, metric in TARGETS:
            plot_bar(data[group][tgt], group, tgt, metric,
                     PLOTS_DIR / f"comparison_{group}_{tgt}.png")

    plot_delta_heatmap(data, PLOTS_DIR / "delta_heatmap_ew.png")
    print_summary(data)
    print("\nГотово.")


if __name__ == "__main__":
    main()
