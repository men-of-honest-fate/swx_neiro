"""
plot_distribution_shift.py
==========================
Сравнение распределений входных и выходных параметров
между обучающей (SC23+SC24) и тестовой (SC25) выборками.

Строка 1 = Train SC23+SC24, строка 2 = Test SC25 — каждая выборка на своих осях.
KS-статистика указана между строками в заголовке колонки.

Выходные файлы:
  plots/dist_shift_inputs.png   — входные признаки (2 строки × 5 колонок)
  plots/dist_shift_targets.png  — целевые переменные (2 строки × 2 колонки)
  plots/dist_shift_all.png      — всё вместе (2 строки × 7 колонок)

Запуск: python plot_distribution_shift.py
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ks_2samp, gaussian_kde

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spe_utils import build_features, COL_CYCLE

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).parent
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_COLOR = "#FF5722"
TEST_COLOR  = "#2196F3"
ALPHA_BAR   = 0.70
ALPHA_KDE   = 0.95

# ── Параметры ─────────────────────────────────────────────────────────────────

# (col, xlabel, log_scale, panel_title, xlim_override)
PARAMS = [
    ("helio_lon",          "Гелиодолгота (°)",            False, "Гелиодолгота",       (-90, 90)),
    ("helio_lat",          "Гелиоширота (°)",             False, "Гелиоширота",         None),
    ("log_goes_peak_flux", "log₁₀ GOES пик (Вт/м²)",     False, "log(GOES пик)",       None),
    ("log_fluence",        "log₁₀ флюэнс (Дж/м²)",       False, "log(Флюэнс)",         None),
    ("log_cme_velocity",   "log₁₀ скорость КВМ (км/с)",  False, "log(Скорость КВМ)",   None),
    ("Jmax",               "log₁₀ J_max (pfu)",           True,  "J_max",               None),
    ("T_delta",            "T_delta (часы)",              False, "T_delta",             None),
]

INPUT_PARAMS  = PARAMS[:5]
TARGET_PARAMS = PARAMS[5:]


# ── Загрузка ──────────────────────────────────────────────────────────────────

def load():
    df = build_features(
        pd.read_excel(ROOT / "data" / "ОБЪЕДИНЕННЫЙ КАТАЛОГ СПС 23-25.xlsx",
                      sheet_name="Флюэс GOES")
    )
    cycle      = pd.to_numeric(df[COL_CYCLE], errors="coerce")
    tdelta     = pd.to_numeric(df["T_delta"],       errors="coerce")
    goes_rise  = pd.to_numeric(df["goes_rise_min"], errors="coerce")
    mask = (
        (df["Jmax"].fillna(0) >= 10) &
        (tdelta.fillna(0) <= 40) &
        (goes_rise.fillna(0) <= 120)
    )
    train  = df[cycle.isin([23, 24]) & mask].copy()
    test   = df[cycle.isin([25])     & mask].copy()
    print(f"Train SC23+SC24: {len(train)}  |  Test SC25: {len(test)}")
    return train, test


# ── Одна панель (одна выборка) ────────────────────────────────────────────────

def draw_panel(ax, vals_raw, color, label, xlabel, log_scale, xlim_override, bins_ref=None):
    """
    Рисует гистограмму + KDE для одной группы.
    bins_ref — общие бины (чтобы train/test были совместимы по ширине).
    Возвращает бины.
    """
    vals = np.array(vals_raw, dtype=float)
    vals = vals[np.isfinite(vals)]
    if log_scale:
        vals = np.log10(np.maximum(vals, 1e-30))
    if xlim_override is not None:
        lo_x, hi_x = xlim_override
        if log_scale:
            lo_x = np.log10(lo_x) if lo_x > 0 else lo_x
            hi_x = np.log10(hi_x) if hi_x > 0 else hi_x
        vals = vals[(vals >= lo_x) & (vals <= hi_x)]

    if len(vals) < 3:
        ax.text(0.5, 0.5, "Нет данных", ha="center", va="center",
                transform=ax.transAxes, color="gray", fontsize=10)
        ax.set_xlabel(xlabel, fontsize=8.5)
        return bins_ref

    # Бины
    if bins_ref is None:
        vmin, vmax = vals.min(), vals.max()
        span = max(vmax - vmin, 1e-6)
        n_bins = min(20, max(8, int(len(vals) / 4)))
        bins_ref = np.linspace(vmin - span * 0.03, vmax + span * 0.03, n_bins + 1)

    ax.hist(vals, bins=bins_ref,
            color=color, alpha=ALPHA_BAR,
            edgecolor="white", linewidth=0.5)

    # KDE (масштабирована под количество событий)
    x_lo, x_hi = bins_ref[0], bins_ref[-1]
    x_grid = np.linspace(x_lo, x_hi, 300)
    if len(vals) >= 5 and vals.std() > 0:
        bin_width = bins_ref[1] - bins_ref[0]
        kde = gaussian_kde(vals, bw_method="scott")
        ax.plot(x_grid, kde(x_grid) * len(vals) * bin_width,
                color=color, lw=2.2, alpha=ALPHA_KDE)

    # Медиана
    med = np.median(vals)
    ax.axvline(med, color=color, lw=1.8, ls="--", alpha=0.9,
               label=f"Медиана = {med:.2f}")

    ax.set_xlim(x_lo, x_hi)
    ax.set_xlabel(xlabel, fontsize=8.5)
    ax.set_ylabel("Количество событий", fontsize=8)
    ax.legend(fontsize=7.5, framealpha=0.7)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    return bins_ref


# ── Построение фигуры (2 строки × N колонок) ─────────────────────────────────

def build_figure(param_list, train_df, test_df, fig_title, out_path):
    n = len(param_list)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 8.5),
                             gridspec_kw={"hspace": 0.55, "wspace": 0.38})
    if n == 1:
        axes = axes.reshape(2, 1)

    # Метки строк
    fig.text(0.005, 0.73, "Train\nSC23+SC24", va="center", ha="left",
             fontsize=10, fontweight="bold", color=TRAIN_COLOR,
             rotation=90)
    fig.text(0.005, 0.27, "Test\nSC25", va="center", ha="left",
             fontsize=10, fontweight="bold", color=TEST_COLOR,
             rotation=90)

    for col_idx, (col, xlabel, log_scale, title, xlim_ov) in enumerate(param_list):
        ax_tr = axes[0, col_idx]
        ax_te = axes[1, col_idx]

        tr_raw = pd.to_numeric(train_df.get(col), errors="coerce").dropna().values
        te_raw = pd.to_numeric(test_df.get(col),  errors="coerce").dropna().values

        # Общие бины: строим по объединённым данным
        tr_proc = np.log10(np.maximum(tr_raw, 1e-30)) if log_scale else tr_raw.copy()
        te_proc = np.log10(np.maximum(te_raw, 1e-30)) if log_scale else te_raw.copy()
        tr_proc = tr_proc[np.isfinite(tr_proc)]
        te_proc = te_proc[np.isfinite(te_proc)]
        if xlim_ov is not None:
            lo_x, hi_x = xlim_ov
            tr_proc = tr_proc[(tr_proc >= lo_x) & (tr_proc <= hi_x)]
            te_proc = te_proc[(te_proc >= lo_x) & (te_proc <= hi_x)]

        all_v = np.concatenate([tr_proc, te_proc])
        if len(all_v) >= 6:
            vmin, vmax = all_v.min(), all_v.max()
            span = max(vmax - vmin, 1e-6)
            n_bins = min(20, max(8, int(len(all_v) / 5)))
            shared_bins = np.linspace(vmin - span * 0.03,
                                      vmax + span * 0.03, n_bins + 1)
        else:
            shared_bins = None

        draw_panel(ax_tr, tr_raw, TRAIN_COLOR,
                   f"Train (n={len(tr_proc)})", xlabel,
                   log_scale, xlim_ov, bins_ref=shared_bins)
        draw_panel(ax_te, te_raw, TEST_COLOR,
                   f"Test (n={len(te_proc)})",  xlabel,
                   log_scale, xlim_ov, bins_ref=shared_bins)

        # KS в заголовке колонки (над верхней панелью)
        if len(tr_proc) >= 3 and len(te_proc) >= 3:
            ks, p = ks_2samp(tr_proc, te_proc)
            flag  = "" if p > 0.05 else (" ⚠" if p > 0.01 else " ✗")
            ks_str = f"KS={ks:.2f}  p={p:.3f}{flag}"
        else:
            ks_str = ""

        ax_tr.set_title(f"{title}\n{ks_str}", fontsize=9.5, fontweight="bold",
                        color="#222", pad=6)
        ax_te.set_title("")

    fig.suptitle(fig_title, fontsize=12, fontweight="bold", y=1.01)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("Загрузка данных...")
    train, test = load()

    print("\nВходные признаки...")
    build_figure(
        INPUT_PARAMS, train, test,
        "Входные признаки: Train SC23+SC24 (сверху) vs Test SC25 (снизу)",
        PLOTS_DIR / "dist_shift_inputs.png"
    )

    print("Целевые переменные...")
    build_figure(
        TARGET_PARAMS, train, test,
        "Целевые переменные: Train SC23+SC24 (сверху) vs Test SC25 (снизу)",
        PLOTS_DIR / "dist_shift_targets.png"
    )

    print("Общий дашборд...")
    build_figure(
        PARAMS, train, test,
        "Распределение параметров: Train SC23+SC24 (сверху) vs Test SC25 (снизу)",
        PLOTS_DIR / "dist_shift_all.png"
    )

    # Сводная таблица
    print("\n── KS-статистика ──")
    print(f"  {'Параметр':<22}  {'KS':>6}  {'p-value':>9}  Вывод")
    print(f"  {'-'*22}  {'-'*6}  {'-'*9}  {'-'*10}")
    for col, xlabel, log_scale, title, xlim_ov in PARAMS:
        tr = pd.to_numeric(train.get(col), errors="coerce").dropna().values
        te = pd.to_numeric(test.get(col),  errors="coerce").dropna().values
        if log_scale:
            tr = np.log10(np.maximum(tr, 1e-30))
            te = np.log10(np.maximum(te, 1e-30))
        if xlim_ov:
            tr = tr[(tr >= xlim_ov[0]) & (tr <= xlim_ov[1])]
            te = te[(te >= xlim_ov[0]) & (te <= xlim_ov[1])]
        tr, te = tr[np.isfinite(tr)], te[np.isfinite(te)]
        if len(tr) < 3 or len(te) < 3:
            print(f"  {title:<22}  {'—':>6}  {'—':>9}  нет данных")
            continue
        ks, p = ks_2samp(tr, te)
        flag = "OK" if p > 0.05 else ("ВНИМАНИЕ" if p > 0.01 else "СДВИГ!")
        print(f"  {title:<22}  {ks:>6.3f}  {p:>9.4f}  {flag}")

    print("\nГотово.")


if __name__ == "__main__":
    main()
