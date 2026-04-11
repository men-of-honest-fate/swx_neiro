"""
Вероятностный/интервальный прогноз параметров СПС.
Аналог spe_utils.py — все модели возвращают (y_lo, y_mid, y_hi).

Метрики оцениваются в «рабочем» пространстве модели:
  Jmax   → log10(pfu)
  T_delta → часы
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import QuantileRegressor, BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from spe_utils import (
    build_features, prepare_xy, make_cycle_cv_splits,
    COL_CYCLE, KF_STRAT,
)

# ── Константы ─────────────────────────────────────────────────────────────────

CATALOG_PATH = Path(__file__).parent.parent.parent / "data" / "ОБЪЕДИНЕННЫЙ КАТАЛОГ СПС 23-25.xlsx"
SHEET        = "Флюэс GOES"
TRAIN_CYCLES = {23, 24}
TEST_CYCLE   = 25
COVERAGE     = 0.80         # 80% prediction interval
_ALPHA       = 1 - COVERAGE  # 0.20

_GPR_KERNEL = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)

MODEL_COLORS = {
    "QuantLinear":   "#1f77b4",
    "QuantBoosting": "#ff7f0e",
    "BayesRidge":    "#2ca02c",
    "GPR_RBF":       "#d62728",
    "ConformalRF":   "#9467bd",
}

# ── Модели-обёртки ────────────────────────────────────────────────────────────


class QuantileLinear(BaseEstimator, RegressorMixin):
    """Три QuantileRegressor (Q_lo, Q_mid, Q_hi) в одной обёртке."""

    def __init__(self, coverage: float = COVERAGE):
        self.coverage = coverage

    def fit(self, X, y):
        lo, hi = _alpha2quantiles(self.coverage)
        self.q_lo_  = QuantileRegressor(quantile=lo,  alpha=0.0, solver="highs").fit(X, y)
        self.q_mid_ = QuantileRegressor(quantile=0.5, alpha=0.0, solver="highs").fit(X, y)
        self.q_hi_  = QuantileRegressor(quantile=hi,  alpha=0.0, solver="highs").fit(X, y)
        return self

    def predict(self, X):
        return self.q_mid_.predict(X)

    def predict_interval(self, X):
        return self.q_lo_.predict(X), self.q_mid_.predict(X), self.q_hi_.predict(X)


class QuantileBoosting(BaseEstimator, RegressorMixin):
    """GradientBoostingRegressor с quantile loss для трёх квантилей."""

    def __init__(self, coverage=COVERAGE, n_estimators=200,
                 max_depth=3, learning_rate=0.05):
        self.coverage      = coverage
        self.n_estimators  = n_estimators
        self.max_depth     = max_depth
        self.learning_rate = learning_rate

    def fit(self, X, y):
        lo, hi = _alpha2quantiles(self.coverage)
        kw = dict(n_estimators=self.n_estimators, max_depth=self.max_depth,
                  learning_rate=self.learning_rate, random_state=42)
        self.q_lo_  = GradientBoostingRegressor(loss="quantile", alpha=lo,  **kw).fit(X, y)
        self.q_mid_ = GradientBoostingRegressor(loss="quantile", alpha=0.5, **kw).fit(X, y)
        self.q_hi_  = GradientBoostingRegressor(loss="quantile", alpha=hi,  **kw).fit(X, y)
        return self

    def predict(self, X):
        return self.q_mid_.predict(X)

    def predict_interval(self, X):
        return self.q_lo_.predict(X), self.q_mid_.predict(X), self.q_hi_.predict(X)


class GaussianWrapper(BaseEstimator, RegressorMixin):
    """Обёртка для BayesianRidge / GPR: predict(return_std=True) → интервал."""

    def __init__(self, base_model, coverage=COVERAGE):
        self.base_model = base_model
        self.coverage   = coverage

    def fit(self, X, y):
        self.model_ = clone(self.base_model)
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_std(self, X):
        _, sigma = self.model_.predict(X, return_std=True)
        return sigma

    def predict_interval(self, X):
        z = stats.norm.ppf(0.5 + self.coverage / 2)
        mu, sigma = self.model_.predict(X, return_std=True)
        return mu - z * sigma, mu, mu + z * sigma


class ConformalRF(BaseEstimator, RegressorMixin):
    """
    RandomForest + split-conformal prediction intervals.
    При fit() оставляет calib_frac данных для калибровки радиуса.
    """

    def __init__(self, coverage=COVERAGE, n_estimators=200, calib_frac=0.20):
        self.coverage     = coverage
        self.n_estimators = n_estimators
        self.calib_frac   = calib_frac

    def fit(self, X, y):
        rng = np.random.default_rng(42)
        n_cal = max(5, int(len(X) * self.calib_frac))
        idx = rng.permutation(len(X))
        cal_idx, fit_idx = idx[:n_cal], idx[n_cal:]

        self.rf_ = RandomForestRegressor(
            n_estimators=self.n_estimators, random_state=42
        ).fit(X[fit_idx], y[fit_idx])

        resid = np.abs(y[cal_idx] - self.rf_.predict(X[cal_idx]))
        n = len(resid)
        q_idx = int(np.ceil((n + 1) * self.coverage)) - 1
        self.radius_ = float(np.sort(resid)[min(q_idx, n - 1)])
        return self

    def predict(self, X):
        return self.rf_.predict(X)

    def predict_interval(self, X):
        mu = self.rf_.predict(X)
        return mu - self.radius_, mu, mu + self.radius_


def _alpha2quantiles(coverage):
    a = (1 - coverage) / 2
    return a, 1 - a


def make_prob_models():
    return {
        "QuantLinear":   QuantileLinear(),
        "QuantBoosting": QuantileBoosting(),
        "BayesRidge":    GaussianWrapper(BayesianRidge()),
        "GPR_RBF":       GaussianWrapper(
            GaussianProcessRegressor(
                kernel=clone(_GPR_KERNEL),
                n_restarts_optimizer=3,
                normalize_y=True,
                random_state=42,
            )
        ),
        "ConformalRF":   ConformalRF(),
    }


# ── Метрики ───────────────────────────────────────────────────────────────────


def picp(y_true, y_lo, y_hi):
    """Prediction Interval Coverage Probability (цель = 0.80)."""
    return float(np.mean((y_true >= y_lo) & (y_true <= y_hi)))


def miw(y_lo, y_hi):
    """Mean Interval Width."""
    return float(np.mean(y_hi - y_lo))


def winkler_score(y_true, y_lo, y_hi, alpha=_ALPHA):
    """Winkler score — совмещает ширину и штраф за выход за границы. Меньше = лучше."""
    width = y_hi - y_lo
    pen   = (np.where(y_true < y_lo, 2 / alpha * (y_lo - y_true), 0.0) +
             np.where(y_true > y_hi, 2 / alpha * (y_true - y_hi), 0.0))
    return float(np.mean(width + pen))


def pinball_loss(y_true, y_pred, q):
    e = y_true - y_pred
    return float(np.mean(np.where(e >= 0, q * e, (q - 1) * e)))


def crps_gaussian(y_true, mu, sigma):
    """CRPS для гауссовского распределения прогноза. Меньше = лучше."""
    sigma = np.maximum(sigma, 1e-8)
    z = (y_true - mu) / sigma
    return float(np.mean(
        sigma * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))
    ))


def score_interval(y_true, y_lo, y_mid, y_hi, sigma=None):
    """Агрегированные вероятностные метрики в рабочем пространстве модели."""
    lo_q, hi_q = _alpha2quantiles(COVERAGE)
    cov   = picp(y_true, y_lo, y_hi)
    width = miw(y_lo, y_hi)
    wink  = winkler_score(y_true, y_lo, y_hi)
    pb    = (pinball_loss(y_true, y_lo, lo_q) +
             pinball_loss(y_true, y_hi, hi_q)) / 2
    crps  = crps_gaussian(y_true, y_mid, sigma) if sigma is not None else np.nan
    return dict(coverage=round(cov, 3), width=round(width, 4),
                winkler=round(wink, 4), pinball=round(pb, 4),
                crps=round(crps, 4) if not np.isnan(crps) else np.nan)


# ── Загрузка данных ───────────────────────────────────────────────────────────


def load_data():
    df = pd.read_excel(CATALOG_PATH, sheet_name=SHEET)
    df = build_features(df)
    df = df[df["Jmax"].fillna(0) >= 10].copy()
    train_df = df[df[COL_CYCLE].isin(TRAIN_CYCLES)].copy()
    test_df  = df[df[COL_CYCLE] == TEST_CYCLE].copy()
    print(f"Train SC23+SC24: {len(train_df)}  |  Test SC25: {len(test_df)}  (Jmax>=10)")
    return train_df, test_df


# ── Обучение + оценка ─────────────────────────────────────────────────────────


def prob_fit_and_score(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    feature_cols: list,
    target_col: str,
    log_target: bool,
    models: dict = None,
) -> dict:
    """
    Обучает вероятностные модели, считает интервальные метрики.

    Метрики считаются в «рабочем» пространстве:
      log_target=True  → log10(y)  (не экспоненцируем)
      log_target=False → y в исходных единицах

    Возвращает dict с ключами:
      cv_metrics, test_metrics, cv_intervals, test_intervals,
      fitted, test_true_metric, y_tr_metric, log_target, has_test,
      cycle_names, cv_mode
    """
    if models is None:
        models = make_prob_models()

    X_tr_raw, y_tr_raw, _, cycle_tr = prepare_xy(train_df, feature_cols, target_col)
    X_te_raw, y_te_raw, _, _        = prepare_xy(test_df,  feature_cols, target_col)

    if len(X_tr_raw) == 0:
        raise ValueError(f"Train пустой: features={feature_cols}, target={target_col}")

    sx = StandardScaler()
    X_tr = sx.fit_transform(X_tr_raw)
    has_test = len(X_te_raw) > 0
    X_te = sx.transform(X_te_raw) if has_test else np.empty((0, X_tr.shape[1]))

    # Масштабирование y + рабочее пространство для метрик
    sy = StandardScaler()
    if log_target:
        y_tr_metric = np.log10(np.clip(y_tr_raw, 1e-6, None))
        y_tr_s = sy.fit_transform(y_tr_metric.reshape(-1, 1)).ravel()
        y_te_metric = np.log10(np.clip(y_te_raw, 1e-6, None)) if has_test else np.array([])
    else:
        y_tr_metric = y_tr_raw
        y_tr_s = sy.fit_transform(y_tr_raw.reshape(-1, 1)).ravel()
        y_te_metric = y_te_raw if has_test else np.array([])

    # CV-стратегия
    unique_cycles = sorted(set(c for c in cycle_tr if not np.isnan(c)))
    if len(unique_cycles) >= 2:
        cv_splits   = make_cycle_cv_splits(cycle_tr)
        cycle_names = [f"SC{int(c)}" for c in unique_cycles]
        cv_mode     = "cross_cycle"
    else:
        strata = (pd.cut(pd.Series(y_tr_s), bins=5, labels=False, duplicates="drop")
                  .fillna(0).astype(int).values)
        cv_splits   = list(KF_STRAT.split(X_tr, strata))
        cycle_names = []
        cv_mode     = "stratified_kfold"

    cv_metrics_all   = {}
    test_metrics_all = {}
    cv_intervals     = {}
    test_intervals   = {}
    fitted_models    = {}

    for name, mdl in models.items():
        # ── CV: собираем интервалы по фолдам ─────────────────────────────────
        lo_cv  = np.full(len(y_tr_s), np.nan)
        mid_cv = np.full(len(y_tr_s), np.nan)
        hi_cv  = np.full(len(y_tr_s), np.nan)

        for tr_idx, val_idx in cv_splits:
            m = clone(mdl)
            m.fit(X_tr[tr_idx], y_tr_s[tr_idx])
            lo_s, mid_s, hi_s = m.predict_interval(X_tr[val_idx])
            lo_cv[val_idx]  = _from_scaled(lo_s,  sy, log_target)
            mid_cv[val_idx] = _from_scaled(mid_s, sy, log_target)
            hi_cv[val_idx]  = _from_scaled(hi_s,  sy, log_target)

        sigma_cv = _get_sigma_cv(mdl, lo_cv, hi_cv)
        cv_metrics_all[name] = score_interval(y_tr_metric, lo_cv, mid_cv, hi_cv, sigma_cv)
        cv_intervals[name]   = (lo_cv, mid_cv, hi_cv)

        # ── Полный train → test ───────────────────────────────────────────────
        m_full = clone(mdl)
        m_full.fit(X_tr, y_tr_s)
        fitted_models[name] = m_full

        if has_test:
            lo_s, mid_s, hi_s = m_full.predict_interval(X_te)
            lo_te  = _from_scaled(lo_s,  sy, log_target)
            mid_te = _from_scaled(mid_s, sy, log_target)
            hi_te  = _from_scaled(hi_s,  sy, log_target)
            sigma_te = _get_sigma_te(m_full, lo_s, hi_s, sy)
            test_metrics_all[name] = score_interval(y_te_metric, lo_te, mid_te, hi_te, sigma_te)
            test_intervals[name]   = (lo_te, mid_te, hi_te)
        else:
            test_metrics_all[name] = {k: np.nan for k in
                                      ["coverage", "width", "winkler", "pinball", "crps"]}
            test_intervals[name] = (np.array([]), np.array([]), np.array([]))

    return dict(
        cv_metrics=cv_metrics_all,
        test_metrics=test_metrics_all,
        cv_intervals=cv_intervals,
        test_intervals=test_intervals,
        fitted=fitted_models,
        test_true_metric=y_te_metric,
        y_tr_metric=y_tr_metric,
        log_target=log_target,
        has_test=has_test,
        cycle_names=cycle_names,
        cv_mode=cv_mode,
    )


def _from_scaled(y_s, sy, log_target):
    """Инвертируем StandardScaler, но НЕ экспоненцируем — метрики в log10."""
    return sy.inverse_transform(np.asarray(y_s).reshape(-1, 1)).ravel()


def _get_sigma_cv(mdl, lo, hi):
    """Грубая оценка sigma из ширины интервала."""
    if not isinstance(mdl, GaussianWrapper):
        return None
    z = stats.norm.ppf(0.5 + COVERAGE / 2)
    sigma = (hi - lo) / (2 * z)
    return sigma


def _get_sigma_te(m_full, lo_s, hi_s, sy):
    if not isinstance(m_full, GaussianWrapper):
        return None
    z = stats.norm.ppf(0.5 + COVERAGE / 2)
    # sigma в scaled пространстве → sigma в log10 пространстве
    sigma_s = (hi_s - lo_s) / (2 * z)
    return sigma_s * sy.scale_[0]


# ── Графики ───────────────────────────────────────────────────────────────────

def plot_intervals(result: dict, title: str, out_path: Path,
                   ylabel: str = "Значение", log_scale: bool = False):
    """
    Для каждой модели: точечный прогноз + 80% интервал vs. истинные значения
    (отсортировано по y_true).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    models   = list(result["test_intervals"].keys())
    y_true   = result["test_true_metric"]
    has_test = result["has_test"]
    n_models = len(models)

    if not has_test or len(y_true) == 0:
        print(f"  [skip] {out_path.name} — нет тестовых данных")
        return

    sort_idx = np.argsort(y_true)
    yt_s     = y_true[sort_idx]
    x_ax     = np.arange(len(yt_s))

    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                             figsize=(5.5 * cols, 4 * rows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, name in zip(axes_flat, models):
        lo, mid, hi = result["test_intervals"][name]
        if len(lo) == 0:
            ax.set_visible(False)
            continue
        lo_s, mid_s, hi_s = lo[sort_idx], mid[sort_idx], hi[sort_idx]
        m = result["test_metrics"][name]
        color = MODEL_COLORS.get(name, "#333")

        ax.fill_between(x_ax, lo_s, hi_s, alpha=0.25, color=color, label="80% CI")
        ax.plot(x_ax, mid_s, color=color, lw=1.5, label="Медиана")
        ax.scatter(x_ax, yt_s, s=22, color="k", zorder=4, label="Факт")

        cov = m.get("coverage", np.nan)
        wink = m.get("winkler", np.nan)
        ax.set_title(f"{name}\nCoverage={cov:.0%}  Winkler={wink:.3f}", fontsize=8)
        ax.set_xlabel("Событие (отсортировано по факту)", fontsize=7)
        ax.set_ylabel(ylabel, fontsize=7)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6)

    for ax in axes_flat[n_models:]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_calibration(result: dict, title: str, out_path: Path):
    """
    Диаграмма калибровки: номинальное покрытие vs. наблюдаемое.
    Рисуется только для Gaussian-моделей (BayesRidge, GPR_RBF).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    y_true = result["y_tr_metric"]
    levels = np.linspace(0.05, 0.95, 19)

    gaussian_models = {
        n: m for n, m in result["fitted"].items()
        if isinstance(m, GaussianWrapper)
    }
    if not gaussian_models:
        print(f"  [skip] {out_path.name} — нет Gaussian-моделей")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Идеал")

    for name, m_full in gaussian_models.items():
        # используем CV-интервалы (уже посчитаны, но только для одного уровня)
        # переходим к predict_std через fitted model на train данных
        # Берём cv-интервалы и оцениваем sigma из них
        lo_cv, mid_cv, hi_cv = result["cv_intervals"][name]
        z_80 = stats.norm.ppf(0.5 + COVERAGE / 2)
        sigma_cv = (hi_cv - lo_cv) / (2 * z_80)
        sigma_cv = np.maximum(sigma_cv, 1e-8)

        empirical = []
        for lvl in levels:
            z = stats.norm.ppf(0.5 + lvl / 2)
            covered = np.mean(
                (y_true >= mid_cv - z * sigma_cv) &
                (y_true <= mid_cv + z * sigma_cv)
            )
            empirical.append(covered)

        color = MODEL_COLORS.get(name, "#333")
        ax.plot(levels, empirical, "o-", color=color,
                ms=4, lw=1.5, label=name)

    ax.set_xlabel("Номинальное покрытие")
    ax.set_ylabel("Наблюдаемое покрытие")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_metrics_bar(results_dict: dict, title: str, out_path: Path,
                     phase: str = "test"):
    """
    Сравнительная гистограмма метрик для нескольких моделей.
    phase = 'test' или 'cv'
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_key = f"{phase}_metrics"

    models  = list(next(iter(results_dict.values()))[metrics_key].keys())
    metrics = ["coverage", "width", "winkler", "pinball"]
    n_m     = len(metrics)
    n_mod   = len(models)

    fig, axes = plt.subplots(1, n_m, figsize=(4 * n_m, 4.5))
    bar_w = 0.6
    x = np.arange(n_mod)

    for ax, metric in zip(axes, metrics):
        vals = []
        for name in models:
            # results_dict может быть {target: result} или просто result
            if "cv_metrics" in results_dict:
                v = results_dict[metrics_key].get(name, {}).get(metric, np.nan)
            else:
                # single-target dict passed directly
                v = results_dict.get(metrics_key, {}).get(name, {}).get(metric, np.nan)
            vals.append(v)

        colors = [MODEL_COLORS.get(m, "#aaa") for m in models]
        bars = ax.bar(x, vals, width=bar_w, color=colors, edgecolor="white")

        if metric == "coverage":
            ax.axhline(COVERAGE, color="red", lw=1.5, ls="--", label=f"{COVERAGE:.0%} цель")
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=8)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                        f"{v:.0%}", ha="center", va="bottom", fontsize=7)
        else:
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, v + max(vals)*0.01,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
        ax.set_title(metric, fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")
