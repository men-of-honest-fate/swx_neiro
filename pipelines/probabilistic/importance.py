"""
Анализ вклада признаков — вероятностный пайплайн.

Для дерево-моделей (QuantBoosting, ConformalRF) вычисляется SHAP,
для линейных (QuantLinear, BayesRidge) — |coef_| (builtin).
GPR_RBF пропускается.

Выход:
  plots/importance/global/   — глобальные тепловые карты + рейтинг
  plots/importance/per_model/ — per-model тепловые карты (признаки × наборы)
  results/importance_results.xlsx

Запуск: python importance.py
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import (
    load_data, make_prob_models, prob_fit_and_score,
    QuantileLinear, QuantileBoosting, GaussianWrapper, ConformalRF,
    MODEL_COLORS,
)
from spe_utils import prepare_xy
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PIPELINE  = Path(__file__).parent
PLOTS_DIR = PIPELINE / "plots" / "importance"
OUT_XLSX  = PIPELINE / "results" / "importance_results.xlsx"

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

MODEL_ORDER = ["QuantLinear", "QuantBoosting", "BayesRidge", "ConformalRF"]

_SHAP_MODELS = {"QuantBoosting", "ConformalRF"}

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


# ── Извлечение важностей ──────────────────────────────────────────────────────

def _builtin_importance(model, n_feats):
    if isinstance(model, QuantileLinear):
        fi = np.abs(np.asarray(model.q_mid_.coef_, dtype=float))
    elif isinstance(model, QuantileBoosting):
        fi = np.asarray(model.q_mid_.feature_importances_, dtype=float)
    elif isinstance(model, GaussianWrapper):
        inner = model.model_
        if hasattr(inner, "coef_"):
            fi = np.abs(np.asarray(inner.coef_, dtype=float))
        else:
            return np.zeros(n_feats)
    elif isinstance(model, ConformalRF):
        fi = np.asarray(model.rf_.feature_importances_, dtype=float)
    else:
        return np.zeros(n_feats)
    fi = fi[:n_feats]
    fi = np.nan_to_num(fi)
    s = fi.sum()
    return fi / s * 100 if s > 0 else fi


def _shap_importance(model, X_tr, model_name, n_feats):
    """SHAP для QuantBoosting (q_mid_) и ConformalRF (rf_)."""
    try:
        import shap
        if model_name == "QuantBoosting":
            sv = shap.TreeExplainer(model.q_mid_).shap_values(X_tr)
        elif model_name == "ConformalRF":
            sv = shap.TreeExplainer(model.rf_).shap_values(X_tr)
        else:
            return None
        if isinstance(sv, list):
            sv = sv[0]
        si = np.mean(np.abs(sv), axis=0)[:n_feats]
        si = np.nan_to_num(si)
        s = si.sum()
        return si / s * 100 if s > 0 else si
    except Exception:
        return None


def _get_importance(model, model_name, X_tr, n_feats):
    """Возвращает (importance_array, method_str)."""
    if model_name in _SHAP_MODELS:
        si = _shap_importance(model, X_tr, model_name, n_feats)
        if si is not None:
            return si, "shap"
    return _builtin_importance(model, n_feats), "builtin"


# ── Сбор данных ───────────────────────────────────────────────────────────────

def collect_importances(train_df, test_df):
    records = []
    models = make_prob_models()

    for tgt in TARGETS:
        for fs_label, fs_cols in FEATURE_SETS:
            print(f"  [{tgt['col']}] {fs_label:<25}", end=" ... ", flush=True)
            try:
                result = prob_fit_and_score(
                    train_df, test_df, fs_cols, tgt["col"], tgt["log"],
                    models={k: v for k, v in models.items()},
                )
                # Вычисляем X_tr (scaled) для SHAP
                X_tr_raw, _, _, _ = prepare_xy(train_df, fs_cols, tgt["col"])
                sx = StandardScaler()
                X_tr = sx.fit_transform(X_tr_raw)

                n_feats = len(fs_cols)
                for model_name, fitted_mdl in result["fitted"].items():
                    if model_name == "GPR_RBF":
                        continue
                    imp, method = _get_importance(fitted_mdl, model_name, X_tr, n_feats)
                    for fi, col in enumerate(fs_cols):
                        records.append({
                            "target":       tgt["col"],
                            "target_label": tgt["label"],
                            "feature_set":  fs_label,
                            "feature":      col,
                            "feat_label":   _feat_label(col),
                            "model":        model_name,
                            "method":       method,
                            "importance":   float(imp[fi]) if fi < len(imp) else 0.0,
                        })
                print("OK")
            except Exception as e:
                print(f"ERROR: {e}")
    return pd.DataFrame(records)


# ── Утилита: тепловая карта ───────────────────────────────────────────────────

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


# ── Глобальная тепловая карта (среднее по моделям) ───────────────────────────

def plot_global_heatmap(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for tgt_col in df["target"].unique():
        tsub = df[df["target"] == tgt_col]
        target_label = tsub["target_label"].iloc[0]
        _heatmap(
            _build_mat(tsub),
            f"Важность признаков — Вероятностный [{target_label}]  (среднее по моделям)\n"
            f"Серое = признак не входит в набор",
            out_dir / f"global_heatmap_prob_{tgt_col.lower()}.png",
        )


# ── Per-model тепловые карты ──────────────────────────────────────────────────

def plot_per_model_heatmaps(df: pd.DataFrame, out_dir: Path):
    """Для каждой (модель, target): тепловая карта (признаки × наборы признаков)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for tgt_col in df["target"].unique():
        tsub = df[df["target"] == tgt_col]
        target_label = tsub["target_label"].iloc[0]

        for model_name in MODEL_ORDER:
            msub = tsub[tsub["model"] == model_name]
            if msub.empty:
                continue
            # Одна запись метода на модель (уже выбрано при сборе)
            method = msub["method"].iloc[0]
            mat = _build_mat(msub)
            safe_name = model_name.lower().replace("/", "_")
            _heatmap(
                mat,
                f"{model_name} [{method}] — {target_label}\n"
                f"Важность признаков × набор признаков.  Серое = не в наборе",
                out_dir / f"model_{safe_name}_{tgt_col.lower()}.png",
                cmap_name="YlOrRd",
            )


# ── Рейтинг признаков по каждому набору (averaged over models) ────────────────

def plot_per_set_ranking(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for tgt_col in df["target"].unique():
        tsub = df[df["target"] == tgt_col]
        target_label = tsub["target_label"].iloc[0]
        for fs_label, fs_cols in FEATURE_SETS:
            fsub = tsub[tsub["feature_set"] == fs_label]
            if fsub.empty:
                continue
            ranking = fsub.groupby("feat_label")["importance"].mean().sort_values()
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


# ── Глобальный рейтинг признаков ──────────────────────────────────────────────

def plot_feature_ranking(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for tgt_col in df["target"].unique():
        tsub = df[df["target"] == tgt_col]
        ranking = tsub.groupby("feat_label")["importance"].mean().sort_values()
        target_label = tsub["target_label"].iloc[0]

        fig, ax = plt.subplots(figsize=(7, max(3, len(ranking) * 0.45)))
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(ranking)))
        bars = ax.barh(ranking.index, ranking.values, color=colors, edgecolor="white")
        for bar, v in zip(bars, ranking.values):
            ax.text(v + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{v:.1f}%", va="center", fontsize=8)
        ax.set_xlabel("Средняя важность, %")
        ax.set_title(f"Рейтинг признаков — Вероятностный [{target_label}]", fontsize=10)
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()

        path = out_dir / f"ranking_prob_{tgt_col.lower()}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("Загрузка данных...")
    train_df, test_df = load_data()
    print()

    print("Вычисление важностей признаков (вероятностный пайплайн)...")
    df_imp = collect_importances(train_df, test_df)

    print("\nГлобальные тепловые карты...")
    plot_global_heatmap(df_imp, PLOTS_DIR / "global")

    print("Per-model тепловые карты...")
    plot_per_model_heatmaps(df_imp, PLOTS_DIR / "per_model")

    print("Рейтинги признаков...")
    plot_feature_ranking(df_imp, PLOTS_DIR / "global")

    print("Per-set рейтинги признаков...")
    plot_per_set_ranking(df_imp, PLOTS_DIR / "per_set_ranking")

    OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        df_imp.to_excel(writer, sheet_name="raw", index=False)
        for tgt_col in df_imp["target"].unique():
            sub = df_imp[df_imp["target"] == tgt_col]
            piv = sub.pivot_table(
                index=["feature_set", "feature"], columns="model",
                values="importance", aggfunc="mean"
            )
            if not piv.empty:
                piv.to_excel(writer, sheet_name=f"pivot_{tgt_col[:6]}")

    print(f"\nГотово. Таблица: {OUT_XLSX}")
    print(f"Графики: {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
