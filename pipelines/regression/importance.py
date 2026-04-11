"""
Анализ вклада признаков — регрессия.

Для каждого набора признаков × модели: вычисляются важности тремя методами
(builtin, SHAP, mutual_info).  Основная визуализация: per-model heatmap
(признаки × наборы признаков) — отдельный файл для каждой (модели, target).

Запуск: python importance.py
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spe_utils import (
    build_features, fit_and_score, compute_importances,
    COL_CYCLE, MODEL_COLORS,
)

warnings.filterwarnings("ignore")

ROOT       = Path(__file__).parent.parent.parent
PIPELINE   = Path(__file__).parent
INPUT_XLSX = ROOT / "data" / "ОБЪЕДИНЕННЫЙ КАТАЛОГ СПС 23-25.xlsx"
SHEET      = "Флюэс GOES"
PLOTS_DIR  = PIPELINE / "plots" / "importance"
OUT_XLSX   = PIPELINE / "results" / "importance_results.xlsx"

FEATURE_SETS = [
    ("Базовая",           ["helio_lon", "log_goes_peak_flux", "log_cme_velocity"]),
    ("Обе координаты",    ["helio_lon", "helio_lat", "log_goes_peak_flux", "log_cme_velocity"]),
    ("Без КВМ",           ["helio_lon", "log_goes_peak_flux"]),
    ("T_delta_flare",     ["t_delta_flare", "log_goes_peak_flux", "log_cme_velocity"]),
    ("Все базовые",       ["helio_lon", "helio_lat", "t_delta_flare",
                           "log_goes_peak_flux", "log_cme_velocity"]),
    ("Флюэс вместо пика", ["helio_lon", "log_fluence", "log_cme_velocity"]),
    ("Флюэс + пик",       ["helio_lon", "log_fluence", "log_goes_peak_flux", "log_cme_velocity"]),
    ("КВМ расширенный",   ["helio_lon", "log_goes_peak_flux", "log_cme_velocity",
                           "cme_width_deg", "cme_pa_deg"]),
    ("Вспышка расшир.",   ["helio_lon", "log_fluence", "log_goes_peak_flux",
                           "log_flare_dur_min", "log_cme_velocity"]),
    ("Kitchen sink",      ["helio_lon", "helio_lat", "t_delta_flare",
                           "log_fluence", "log_goes_peak_flux", "log_flare_dur_min",
                           "log_cme_velocity", "cme_width_deg", "cme_pa_deg"]),
    ("Пик+нараст.",       ["helio_lon", "log_goes_peak_flux",
                           "log_goes_rise_min", "log_cme_velocity"]),
    ("GOES полный",       ["helio_lon", "helio_lat", "log_goes_peak_flux",
                           "log_fluence", "log_cme_velocity"]),
    ("GOES KS",           ["helio_lon", "helio_lat", "log_goes_peak_flux",
                           "log_fluence", "log_flare_dur_min",
                           "log_cme_velocity", "cme_width_deg", "cme_pa_deg", "t_delta_flare"]),
]

TARGETS = [
    dict(col="Jmax",    log=True,  label="Jmax (pfu)"),
    dict(col="T_delta", log=False, label="T_delta (часы)"),
]

MODEL_ORDER = ["Linear", "Ridge", "Huber", "Forest", "ExtraTrees", "Boosting", "SVR", "GPR_RBF"]

# Для дерево-моделей предпочитаем SHAP, для линейных — builtin
_SHAP_MODELS = {"Forest", "ExtraTrees", "Boosting"}

FEAT_LABELS = {
    "helio_lon":          "Гелио-долгота",
    "helio_lat":          "Гелио-широта",
    "log_goes_peak_flux": "log(GOES пик, W/m²)",
    "log_cme_velocity":   "log(Скорость КВМ)",
    "t_delta_flare":      "Задержка вспышка-СПС",
    "log_fluence":        "log(Флюэнс)",
    "log_flare_dur_min":  "log(Длит. вспышки)",
    "cme_width_deg":      "Угол КВМ",
    "cme_pa_deg":         "Позиц. угол КВМ",
    "log_goes_rise_min":  "log(Нарастание вспышки)",
}


def _feat_label(f):
    return FEAT_LABELS.get(f, f)


# ── Сбор важностей ────────────────────────────────────────────────────────────

def collect_importances(train_df, test_df):
    """
    Возвращает DataFrame:
      target, target_label, feature_set, feature, feat_label, model, method, importance
    """
    records = []
    for tgt in TARGETS:
        for fs_label, fs_cols in FEATURE_SETS:
            print(f"  [{tgt['col']}] {fs_label:<25}", end=" ... ", flush=True)
            try:
                result = fit_and_score(train_df, test_df, fs_cols,
                                       tgt["col"], tgt["log"])
                imps = compute_importances(result, fs_cols)
                for (model, method), arr in imps.items():
                    for fi, col in enumerate(fs_cols):
                        records.append({
                            "target":      tgt["col"],
                            "target_label": tgt["label"],
                            "feature_set": fs_label,
                            "feature":     col,
                            "feat_label":  _feat_label(col),
                            "model":       model,
                            "method":      method,
                            "importance":  float(arr[fi]) if fi < len(arr) else 0.0,
                        })
                print("OK")
            except Exception as e:
                print(f"ERROR: {e}")
    return pd.DataFrame(records)


# ── Утилита: тепловая карта (переиспользуется) ────────────────────────────────

def _heatmap(mat: pd.DataFrame, title: str, out_path: Path, cmap_name="YlOrRd"):
    fig, ax = plt.subplots(figsize=(max(12, mat.shape[1] * 0.9),
                                    max(5, mat.shape[0] * 0.5)))
    data   = mat.values.astype(float)
    masked = np.ma.masked_invalid(data)
    cmap   = plt.cm.get_cmap(cmap_name).copy()
    cmap.set_bad("lightgray")
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0)

    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(mat.index, fontsize=9)

    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            v = data[r, c]
            if np.isfinite(v) and v > 0:
                color = "white" if v > 65 else "black"
                ax.text(c, r, f"{v:.0f}%", ha="center", va="center",
                        fontsize=7, color=color)

    plt.colorbar(im, ax=ax, label="Важность, %")
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def _build_mat(sub: pd.DataFrame) -> pd.DataFrame:
    """Строит матрицу (feat_label × feature_set), усредняя по моделям."""
    fs_order = [fs for fs, _ in FEATURE_SETS]
    agg = sub.groupby(["feat_label", "feature_set"])["importance"].mean().reset_index()
    feat_total = agg.groupby("feat_label")["importance"].sum().sort_values(ascending=False)
    feat_order = feat_total.index.tolist()
    mat = pd.DataFrame(np.nan, index=feat_order, columns=fs_order)
    for _, row in agg.iterrows():
        if row["feat_label"] in mat.index and row["feature_set"] in mat.columns:
            mat.loc[row["feat_label"], row["feature_set"]] = row["importance"]
    return mat


# ── Глобальная тепловая карта (среднее по моделям) ───────────────────────────

def plot_global_heatmap(df: pd.DataFrame, method: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    sub = df[df["method"] == method]
    for tgt_col in df["target"].unique():
        tsub = sub[sub["target"] == tgt_col]
        mean_imp = (
            tsub.groupby(["feat_label", "feature_set"])["importance"]
            .mean().reset_index()
        )
        mat = _build_mat(mean_imp.rename(columns={"importance": "importance"}))
        # _build_mat expects rows with feat_label / feature_set / importance
        mat2 = _build_mat(tsub)
        target_label = tsub["target_label"].iloc[0] if len(tsub) > 0 else tgt_col
        _heatmap(
            mat2,
            f"Важность признаков [{method}] — {target_label}  (среднее по моделям)\n"
            f"Серое = признак не входит в набор",
            out_dir / f"global_{method}_{tgt_col.lower()}.png",
        )


# ── Per-model тепловые карты (SHAP для деревьев, builtin для остальных) ───────

def plot_per_model_heatmaps(df: pd.DataFrame, out_dir: Path):
    """
    Для каждой (модель, target): тепловая карта (признаки × наборы признаков).
    Выбирает SHAP для дерево-моделей, builtin для линейных.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for tgt_col in df["target"].unique():
        tsub = df[df["target"] == tgt_col]
        target_label = tsub["target_label"].iloc[0]

        for model_name in MODEL_ORDER:
            msub = tsub[tsub["model"] == model_name]
            if msub.empty:
                continue

            # Выбор метода: SHAP если дерево и доступен с ненулевыми значениями
            shap_sub = msub[msub["method"] == "shap"]
            if model_name in _SHAP_MODELS and shap_sub["importance"].sum() > 0:
                method, sub = "shap", shap_sub
            else:
                method = "builtin"
                sub = msub[msub["method"] == "builtin"]

            mat = _build_mat(sub)
            safe_name = model_name.lower().replace("/", "_")
            _heatmap(
                mat,
                f"{model_name} [{method}] — {target_label}\n"
                f"Важность признаков × набор признаков. Серое = не в наборе",
                out_dir / f"model_{safe_name}_{tgt_col.lower()}.png",
                cmap_name="YlOrRd",
            )


# ── Рейтинг признаков по каждому набору (averaged over models) ────────────────

def plot_per_set_ranking(df: pd.DataFrame, out_dir: Path):
    """
    Для каждого (feature_set, target): горизонтальный бар-чарт признаков,
    важность = среднее по всем моделям внутри набора.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for tgt_col in df["target"].unique():
        tsub = df[df["target"] == tgt_col]
        target_label = tsub["target_label"].iloc[0]
        for fs_label, fs_cols in FEATURE_SETS:
            fsub = tsub[tsub["feature_set"] == fs_label]
            if fsub.empty:
                continue
            ranking = fsub.groupby("feat_label")["importance"].mean().sort_values()
            # Отображаем только признаки из этого набора (ненулевые)
            ranking = ranking[ranking > 0]
            if ranking.empty:
                continue

            fig, ax = plt.subplots(figsize=(7, max(2.5, len(ranking) * 0.55)))
            cmap = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(ranking)))
            bars = ax.barh(ranking.index, ranking.values, color=cmap,
                           edgecolor="white", linewidth=0.5)
            for bar, v in zip(bars, ranking.values):
                ax.text(v + 0.4, bar.get_y() + bar.get_height() / 2,
                        f"{v:.1f}%", va="center", fontsize=9)
            ax.set_xlabel("Средняя важность (%, averaged over models)")
            ax.set_title(f"{fs_label}  —  {target_label}", fontsize=10)
            ax.set_xlim(0, max(ranking.values) * 1.2)
            ax.grid(axis="x", alpha=0.3)
            plt.tight_layout()

            safe_fs = fs_label.replace("/", "-").replace(" ", "_").replace("+", "p")
            path = out_dir / f"rank_{tgt_col.lower()}_{safe_fs}.png"
            plt.savefig(path, dpi=130, bbox_inches="tight")
            plt.close()

    print(f"  Saved per-set rankings -> {out_dir}/")


# ── Глобальный сводный рейтинг признаков ─────────────────────────────────────

def plot_feature_ranking(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Для дерево-моделей используем SHAP, для остальных builtin
    def best_row(grp):
        model = grp["model"].iloc[0]
        method = "shap" if model in _SHAP_MODELS else "builtin"
        sub = grp[grp["method"] == method]
        return sub["importance"].mean() if not sub.empty else grp["importance"].mean()

    for tgt_col in df["target"].unique():
        tsub = df[df["target"] == tgt_col]
        target_label = tsub["target_label"].iloc[0]

        # Берём builtin (нейтральный для рейтинга по всем моделям)
        sub = tsub[tsub["method"] == "builtin"]
        ranking = sub.groupby("feat_label")["importance"].mean().sort_values()

        fig, ax = plt.subplots(figsize=(7, max(3, len(ranking) * 0.4)))
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(ranking)))
        bars = ax.barh(ranking.index, ranking.values, color=colors, edgecolor="white")
        for bar, v in zip(bars, ranking.values):
            ax.text(v + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{v:.1f}%", va="center", fontsize=8)
        ax.set_xlabel("Средняя важность (builtin, %)")
        ax.set_title(f"Глобальный рейтинг признаков — {target_label}", fontsize=10)
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()

        path = out_dir / f"ranking_{tgt_col.lower()}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print(f"Загрузка: {INPUT_XLSX}")
    df_raw = build_features(pd.read_excel(INPUT_XLSX, sheet_name=SHEET))
    cycle = pd.to_numeric(df_raw[COL_CYCLE], errors="coerce")
    train_df = df_raw[cycle.isin([23, 24])].copy()
    test_df  = df_raw[cycle.isin([25])].copy()
    train_df = train_df[train_df["Jmax"].fillna(0) >= 10].copy()
    test_df  = test_df[test_df["Jmax"].fillna(0) >= 10].copy()
    print(f"Train SC23+SC24: {len(train_df)}  |  Test SC25: {len(test_df)}\n")

    print("Вычисление важностей признаков...")
    df_imp = collect_importances(train_df, test_df)

    print("\nГлобальные тепловые карты (среднее по моделям)...")
    for method in ["builtin", "shap", "mutual_info"]:
        plot_global_heatmap(df_imp, method, PLOTS_DIR / "global")

    print("\nРейтинг признаков...")
    plot_feature_ranking(df_imp, PLOTS_DIR / "global")

    print("\nPer-model тепловые карты...")
    plot_per_model_heatmaps(df_imp, PLOTS_DIR / "per_model")

    # Чистим старые per_set файлы (заменены per_set_ranking + per_model)
    import shutil
    old_per_set = PLOTS_DIR / "per_set"
    if old_per_set.exists():
        shutil.rmtree(old_per_set)

    print("\nPer-set рейтинги признаков (по каждому набору, среднее по моделям)...")
    plot_per_set_ranking(df_imp, PLOTS_DIR / "per_set_ranking")

    OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        df_imp.to_excel(writer, sheet_name="raw", index=False)
        for method in ["builtin", "shap", "mutual_info"]:
            for tgt_col in df_imp["target"].unique():
                sub = df_imp[(df_imp["method"] == method) & (df_imp["target"] == tgt_col)]
                piv = sub.pivot_table(
                    index=["feature_set", "feature"], columns="model",
                    values="importance", aggfunc="mean"
                )
                name = f"{method[:4]}_{tgt_col[:5]}"
                if not piv.empty:
                    piv.to_excel(writer, sheet_name=name)

    print(f"\nГотово. Таблица: {OUT_XLSX}")
    print(f"Графики: {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
