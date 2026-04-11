"""
xray_rise_plots.py
==================
Строит два графика по данным из листа «Флюэс GOES»
файла data/ОБЪЕДИНЕННЫЙ КАТАЛОГ СПС 23-25.xlsx:

1. Соотношение между длительностью нарастания рентгеновского излучения Солнца
   (GOES_Rise_Min) и временем от начала солнечной вспышки до начала роста
   потока протонов (T_delta_flare_onset, часы).

2. Гистограмма количества СПС по длительности фазы нарастания
   рентгеновского излучения (шаг 30 мин).
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Настройки
# ---------------------------------------------------------------------------
DATA_PATH  = "data/ОБЪЕДИНЕННЫЙ КАТАЛОГ СПС 23-25.xlsx"
SHEET_NAME = "Флюэс GOES"
OUTPUT_DIR = "plots"
BIN_STEP   = 30   # минуты

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.unicode_minus": False,
    "figure.dpi":        150,
})

CYCLE_COLORS = {23: "#2196F3", 24: "#FF5722", 25: "#4CAF50"}


# ---------------------------------------------------------------------------
# Загрузка данных
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)

    # Нужные колонки
    df = df[["Цикл", "GOES_Rise_Min", "T_delta_flare_onset",
             "Фаза нарастания (часы)", "Максимальная интенсивность"]].copy()

    df = df.rename(columns={
        "Цикл":                       "Cycle",
        "GOES_Rise_Min":              "xray_rise_min",
        "T_delta_flare_onset":        "flare_to_spe_h",
        "Фаза нарастания (часы)":     "spe_rise_h",
        "Максимальная интенсивность": "jmax",
    })

    df["Cycle"]        = pd.to_numeric(df["Cycle"],        errors="coerce")
    df["xray_rise_min"]= pd.to_numeric(df["xray_rise_min"],errors="coerce")
    df["flare_to_spe_h"]= pd.to_numeric(df["flare_to_spe_h"], errors="coerce")
    df["jmax"]         = pd.to_numeric(df["jmax"],         errors="coerce")

    print(f"Всего строк в листе: {len(df)}")
    return df


# ---------------------------------------------------------------------------
# График 1 — Scatter: длительность нарастания рентгена vs задержка вспышка→СПС
# ---------------------------------------------------------------------------
def plot_scatter(df: pd.DataFrame, save_path: str):
    sub = df.dropna(subset=["xray_rise_min", "flare_to_spe_h", "jmax"])
    # Фильтры:
    #   1) Jmax >= 10 pfu
    #   2) Длительность нарастания рентгена < 2 ч (< 120 мин)
    #   3) Задержка вспышка → СПС <= 10 ч
    sub = sub[
        (sub["jmax"] >= 10) &
        (sub["xray_rise_min"] > 0) &
        (sub["xray_rise_min"] < 120) &
        (sub["flare_to_spe_h"] > 0) &
        (sub["flare_to_spe_h"] <= 10)
    ].copy()

    x = sub["xray_rise_min"].values / 60  # длительность нарастания рентгена, ч
    y = sub["flare_to_spe_h"].values      # задержка вспышка → начало СПС, ч

    xlim_max = x.max() * 1.05
    ylim_max = y.max() * 1.05

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(x, y, color="#2196F3", alpha=0.7, s=45, edgecolors="none")

    ax.plot([0, min(xlim_max, ylim_max)], [0, min(xlim_max, ylim_max)],
            color="crimson", linewidth=1.8, linestyle="--",
            label="y = x  (задержка = длительность нарастания)")

    ax.set_xlim(0, xlim_max)
    ax.set_ylim(0, ylim_max)

    ax.set_xlabel(
        "Длительность нарастания рентгеновского излучения (GOES), ч",
        fontsize=11,
    )
    ax.set_ylabel(
        "Время от начала вспышки до начала роста потока протонов, ч",
        fontsize=11,
    )
    ax.set_title(
        "Соотношение длительности нарастания рентгеновского\n"
        "излучения и задержки вспышка → начало СПС",
        fontsize=12,
    )
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Сохранён: {save_path}")


# ---------------------------------------------------------------------------
# График 1b — Scatter крупным планом: первые 10 часов
# ---------------------------------------------------------------------------
def plot_scatter_zoom(df: pd.DataFrame, save_path: str, zoom_h: float = 10.0):
    sub = df.dropna(subset=["xray_rise_min", "flare_to_spe_h", "jmax"])
    sub = sub[
        (sub["jmax"] >= 10) &
        (sub["xray_rise_min"] > 0) &
        (sub["xray_rise_min"] < 120) &
        (sub["flare_to_spe_h"] > 0) &
        (sub["flare_to_spe_h"] <= 10)
    ].copy()

    x = sub["xray_rise_min"].values / 60
    y = sub["flare_to_spe_h"].values

    xlim_max = x.max() * 1.05

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(x, y, color="#2196F3", alpha=0.7, s=45, edgecolors="none")

    ax.plot([0, min(xlim_max, zoom_h)], [0, min(xlim_max, zoom_h)],
            color="crimson", linewidth=1.8, linestyle="--",
            label="y = x  (задержка = длительность нарастания)")

    ax.set_xlim(0, xlim_max)
    ax.set_ylim(0, zoom_h)

    ax.set_xlabel(
        "Длительность нарастания рентгеновского излучения (GOES), ч",
        fontsize=11,
    )
    ax.set_ylabel(
        "Время от начала вспышки до начала роста потока протонов, ч",
        fontsize=11,
    )
    ax.set_title(
        f"Соотношение длительности нарастания рентгеновского\n"
        f"излучения и задержки вспышка → начало СПС  (Y: 0–{zoom_h:.0f} ч)",
        fontsize=12,
    )
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Сохранён: {save_path}")


# ---------------------------------------------------------------------------
# График 2 — Гистограмма: количество СПС по длительности нарастания рентгена
# ---------------------------------------------------------------------------
def plot_histogram(df: pd.DataFrame, save_path: str):
    sub = df.dropna(subset=["xray_rise_min"])
    rise = sub["xray_rise_min"].values   # минуты

    max_val = np.ceil(rise.max() / BIN_STEP) * BIN_STEP
    bins = np.arange(0, max_val + BIN_STEP, BIN_STEP)

    fig, ax = plt.subplots(figsize=(11, 6))

    counts, edges, patches = ax.hist(
        rise, bins=bins,
        color="#2196F3", edgecolor="white", linewidth=0.6,
        alpha=0.85,
    )

    # Подпись значений над столбцами
    for count, patch in zip(counts, patches):
        if count > 0:
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                count + 0.3,
                str(int(count)),
                ha="center", va="bottom", fontsize=8,
            )

    ax.xaxis.set_major_locator(ticker.MultipleLocator(60))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(BIN_STEP))

    ax.set_xlabel(
        "Длительность нарастания рентгеновского излучения (GOES), мин",
        fontsize=11,
    )
    ax.set_ylabel("Количество СПС", fontsize=11)
    ax.set_title(
        "Распределение СПС по длительности фазы нарастания\n"
        "рентгеновского излучения (шаг 30 мин)",
        fontsize=12,
    )
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    median_val = np.median(rise)
    mean_val   = np.mean(rise)
    ax.axvline(median_val, color="crimson",    linestyle="--", linewidth=1.8,
               label=f"Медиана = {median_val:.1f} мин")
    ax.axvline(mean_val,   color="darkorange", linestyle="-.", linewidth=1.8,
               label=f"Среднее = {mean_val:.1f} мин")
    ax.legend(fontsize=9)

    ax.text(
        0.98, 0.97, f"N = {len(rise)}",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=9, color="gray",
    )

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Сохранён: {save_path}")


# ---------------------------------------------------------------------------
# График 3 — Гистограмма T_delta (фаза нарастания потока протонов)
# ---------------------------------------------------------------------------
TDELTA_MAX_H = 40.0   # события с T_delta > 40 ч обрезаются
TDELTA_BIN_H = 0.5    # шаг бина — 30 мин в часах

def plot_tdelta_histogram(save_path: str):
    df_cat = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)
    tdelta = pd.to_numeric(df_cat["Фаза нарастания (часы)"], errors="coerce")
    jmax   = pd.to_numeric(df_cat["Максимальная интенсивность"], errors="coerce")

    mask   = tdelta.notna() & (jmax >= 10) & (tdelta <= TDELTA_MAX_H)
    values = tdelta[mask].values

    bins = np.arange(0, TDELTA_MAX_H + TDELTA_BIN_H, TDELTA_BIN_H)

    fig, ax = plt.subplots(figsize=(11, 6))

    counts, edges, patches = ax.hist(
        values, bins=bins,
        color="#2196F3", edgecolor="white", linewidth=0.6,
        alpha=0.85,
    )

    for count, patch in zip(counts, patches):
        if count > 0:
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                count + 0.15,
                str(int(count)),
                ha="center", va="bottom", fontsize=8,
            )

    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(TDELTA_BIN_H))

    ax.set_xlim(0, TDELTA_MAX_H)
    ax.set_xlabel("Длительность фазы нарастания потока протонов, T_delta (ч)", fontsize=11)
    ax.set_ylabel("Количество СПС", fontsize=11)
    ax.set_title(
        f"Распределение СПС по длительности фазы нарастания\n"
        f"(шаг 30 мин, Jmax ≥ 10 pfu, T_delta ≤ {TDELTA_MAX_H:.0f} ч)",
        fontsize=12,
    )
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    median_val = np.median(values)
    mean_val   = np.mean(values)
    ax.axvline(median_val, color="crimson",    linestyle="--", linewidth=1.8,
               label=f"Медиана = {median_val:.1f} ч")
    ax.axvline(mean_val,   color="darkorange", linestyle="-.", linewidth=1.8,
               label=f"Среднее = {mean_val:.1f} ч")
    ax.legend(fontsize=9)

    ax.text(
        0.98, 0.97, f"N = {len(values)}",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=9, color="gray",
    )

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Сохранён: {save_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_data()

    plot_scatter(
        df,
        os.path.join(OUTPUT_DIR, "xray_rise_vs_flare_delay.png"),
    )
    plot_scatter_zoom(
        df,
        os.path.join(OUTPUT_DIR, "xray_rise_vs_flare_delay_zoom10h.png"),
    )
    plot_histogram(
        df,
        os.path.join(OUTPUT_DIR, "spe_rise_duration_histogram.png"),
    )
    plot_tdelta_histogram(
        os.path.join(OUTPUT_DIR, "t_delta_spe_histogram.png"),
    )

    print("\nГотово. Графики сохранены в папку:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
