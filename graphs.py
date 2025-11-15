import warnings
import openpyxl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

INPUT_XLSX = "predictions.xlsx"
OUT_DIR = "plots"

MODEL_COLORS = {
    "Boosting": "#1f77b4",
    "Forest":   "#8c564b",
    "Linear":   "#17becf",
}
ALPHA = 0.9
MS = 35  # размер точек


def _safe_numeric(s: pd.Series):
    return pd.to_numeric(s, errors="coerce")


def _xy_true_line_and_pred_points(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    color: str,
    out_path: Path,
    xlabel: str,
    ylabel: str,
    log_scale: bool = False
):
    """
    По X — экспериментальные значения.
    По Y — линия истины: те же экспериментальные значения (y_true).
           точки: предсказанные (y_pred).
    Точки сортируются по X (y_true), соединены линией.
    log_scale: если True, логарифмические оси и пределы для Jmax.
    """
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if y_true.size == 0:
        print(f"[WARN] пустой набор точек для графика: {title}")
        return

    order = np.argsort(y_true)
    x_sorted = y_true[order]
    y_true_sorted = y_true[order]
    y_pred_sorted = y_pred[order]

    plt.figure(figsize=(7.5, 5.2))

    # Линия экспериментальных значений (ломаная): X = y_true, Y = y_true
    plt.plot(x_sorted, y_true_sorted, color="#444444", linewidth=2.0, label="Экспериментальные (линия)")
    # Линия предсказаний
    plt.plot(x_sorted, y_pred_sorted, color=color, linewidth=2.0, linestyle="--", alpha=0.8, label="Предсказанные (линия)")
    # Точки предсказаний
    plt.scatter(x_sorted, y_pred_sorted, s=MS, c=color, alpha=ALPHA, edgecolor="k", linewidth=0.3, label="Предсказанные (точки)")

    plt.grid(alpha=0.3, which="both")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # Для Jmax: логарифм, пределы 10–2000
    if log_scale:
        plt.xscale("log")
        plt.yscale("log")
        plt.xlim(10, 2000)
        plt.ylim(10, 2000)
        ticks = [10, 20, 50, 100, 200, 500, 1000, 2000]
        plt.xticks(ticks, [str(t) for t in ticks])
        plt.yticks(ticks, [str(t) for t in ticks])

    plt.legend(loc="best", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"График сохранён: {out_path}")


def main():
    warnings.filterwarnings("ignore")

    df = pd.read_excel(INPUT_XLSX)

    for col in ["Jmax_parsed", "T_delta_SPE"]:
        if col in df.columns:
            df[col] = _safe_numeric(df[col])

    if "split" in df.columns:
        df_test = df[df["split"] == "test_sc25"].copy()
        if df_test.empty:
            df_test = df.copy()
    else:
        df_test = df.copy()

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = [
        dict(
            name="Jmax_parsed",
            true_col="Jmax_parsed",
            pred_cols={
                "Boosting": "Boosting_Jpred",
                "Forest":   "Forest_Jpred",
                "Linear":   "Linear_Jpred",
            },
            xlabel="Экспериментальные Jmax_parsed (pfu)",
            ylabel="Предсказанные Jmax_parsed (pfu)",
            basename="jmax"
        ),
        dict(
            name="T_delta_SPE",
            true_col="T_delta_SPE",
            pred_cols={
                "Boosting": "Boosting_Delta_T_max",
                "Forest":   "Forest_Delta_T_max",
                "Linear":   "Linear_Delta_T_max",
            },
            xlabel="Экспериментальные T_delta_SPE (часы)",
            ylabel="Предсказанные T_delta_SPE (часы)",
            basename="tdelta"
        ),
    ]

    for tgt in targets:
        tname = tgt["name"]
        y_true_col = tgt["true_col"]
        if y_true_col not in df_test.columns:
            print(f"[SKIP] Нет столбца истины '{y_true_col}' для цели {tname}")
            continue

        y_true = _safe_numeric(df_test[y_true_col]).to_numpy()

        for model_name, pred_col in tgt["pred_cols"].items():
            if pred_col not in df_test.columns:
                print(f"[SKIP] Нет предсказаний '{pred_col}' для модели {model_name} и цели {tname}")
                continue

            y_pred = _safe_numeric(df_test[pred_col]).to_numpy()
            color = MODEL_COLORS.get(model_name, "#333333")
            title = f"{tname} — {model_name}"
            out_path = out_dir / f"{tgt['basename']}_{model_name.lower()}.png"

            log_scale = (tgt["basename"] == "jmax")  # только для Jmax
            _xy_true_line_and_pred_points(
                y_true=y_true,
                y_pred=y_pred,
                title=title,
                color=color,
                out_path=out_path,
                xlabel=tgt["xlabel"],
                ylabel=tgt["ylabel"],
                log_scale=log_scale
            )

    print("Готово.")


if __name__ == "__main__":
    main()
