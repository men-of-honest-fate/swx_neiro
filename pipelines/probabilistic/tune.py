"""
Тюнинг гиперпараметров вероятностных моделей.
Оценивает Winkler score через cross-cycle CV.

Запуск:  python tune.py
"""

import sys
import warnings
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.base import clone

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils import (
    load_data, prob_fit_and_score, COVERAGE,
    QuantileLinear, QuantileBoosting, GaussianWrapper, ConformalRF,
    BayesianRidge,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).parent.parent.parent
PIPELINE = Path(__file__).parent

# ── Конфигурации для тюнинга ──────────────────────────────────────────────────

CONFIGS = {
    "baseline": ["helio_lon", "log_flare_power", "log_cme_velocity"],
    "best":     ["t_delta_flare", "log_flare_power", "log_cme_velocity"],
}

TARGETS = [
    dict(col="Jmax",    log=True,  label="Jmax"),
    dict(col="T_delta", log=False, label="T_delta"),
]

# ── Сетки гиперпараметров ─────────────────────────────────────────────────────

PARAM_GRIDS = {
    "QuantBoosting": {
        "n_estimators":  [100, 200, 400],
        "max_depth":     [2, 3, 4],
        "learning_rate": [0.03, 0.05, 0.10],
    },
    "ConformalRF": {
        "n_estimators": [100, 200, 400],
        "calib_frac":   [0.15, 0.20, 0.30],
    },
    "GPR_RBF": {
        "n_restarts_optimizer": [3, 5, 10],
        "length_scale":         [0.5, 1.0, 2.0],
    },
}


def _make_gpr(n_restarts, length_scale):
    kernel = ConstantKernel(1.0) * RBF(length_scale=length_scale) + WhiteKernel(0.1)
    return GaussianWrapper(
        GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=n_restarts,
            normalize_y=True,
            random_state=42,
        )
    )


def grid_search_winkler(train_df, test_df, feature_cols, target_col, log_target,
                        model_name: str, param_grid: dict):
    """Перебор по сетке, минимизируем CV Winkler score."""
    keys   = list(param_grid.keys())
    values = list(param_grid.values())
    best_winkler = np.inf
    best_params  = {}
    results_rows = []

    total = 1
    for v in values:
        total *= len(v)

    print(f"    {model_name}: {total} комбинаций")

    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))

        # Создаём модель с нужными параметрами
        if model_name == "QuantBoosting":
            mdl = QuantileBoosting(**params)
        elif model_name == "ConformalRF":
            mdl = ConformalRF(**params)
        elif model_name == "GPR_RBF":
            mdl = _make_gpr(params["n_restarts_optimizer"], params["length_scale"])
        else:
            raise ValueError(f"Unknown model: {model_name}")

        try:
            result = prob_fit_and_score(
                train_df, test_df,
                feature_cols=feature_cols,
                target_col=target_col,
                log_target=log_target,
                models={model_name: mdl},
            )
            winkler = result["cv_metrics"][model_name]["winkler"]
            cov     = result["cv_metrics"][model_name]["coverage"]
        except Exception as e:
            winkler = np.nan
            cov     = np.nan

        results_rows.append({**params, "cv_winkler": winkler, "cv_coverage": cov})

        if winkler < best_winkler:
            best_winkler = winkler
            best_params  = params

    df_grid = pd.DataFrame(results_rows).sort_values("cv_winkler")
    return best_params, best_winkler, df_grid


# ── Вывод ─────────────────────────────────────────────────────────────────────

def print_grid_results(df_grid: pd.DataFrame, model_name: str, top_n: int = 5):
    print(f"\n      Топ-{top_n} для {model_name}:")
    cols = [c for c in df_grid.columns if c not in ("cv_coverage",)]
    print("      " + df_grid.head(top_n)[cols].to_string(index=False))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    train_df, test_df = load_data()

    all_rows = []

    for cfg_name, feature_cols in CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"  Конфигурация: {cfg_name}  features={feature_cols}")
        print(f"{'='*60}")

        for tgt in TARGETS:
            col, log, label = tgt["col"], tgt["log"], tgt["label"]
            print(f"\n  >> Цель: {label}")

            # Базовые метрики до тюнинга
            from utils import make_prob_models
            base_result = prob_fit_and_score(
                train_df, test_df, feature_cols, col, log,
                models=make_prob_models(),
            )
            print("    Базовые CV Winkler:")
            for name, m in base_result["cv_metrics"].items():
                print(f"      {name:<18} Winkler={m['winkler']:.4f}  Coverage={m['coverage']:.0%}")

            # Тюнинг
            for model_name, param_grid in PARAM_GRIDS.items():
                best_p, best_wink, df_grid = grid_search_winkler(
                    train_df, test_df, feature_cols, col, log,
                    model_name, param_grid
                )
                print_grid_results(df_grid, model_name)
                print(f"      Лучшие параметры: {best_p}  Winkler={best_wink:.4f}")

                all_rows.append(dict(
                    config=cfg_name, target=label, model=model_name,
                    best_winkler=best_wink, **best_p
                ))

    # Сохраняем сводку
    df_out = pd.DataFrame(all_rows)
    out_path = PIPELINE / "results" / "tune_results.xlsx"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_excel(out_path, index=False)
    print(f"\nРезультаты тюнинга: '{out_path}'")

    print("\n" + "="*60)
    print("  Сводка лучших параметров:")
    print("="*60)
    for _, row in df_out.iterrows():
        params = {k: v for k, v in row.items()
                  if k not in ("config", "target", "model", "best_winkler")}
        print(f"  [{row['config']}] {row['target']} / {row['model']}: "
              f"Winkler={row['best_winkler']:.4f}  params={params}")


if __name__ == "__main__":
    main()
