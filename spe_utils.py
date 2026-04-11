"""
Общие утилиты для pipeline обучения моделей СПС.
Используются в train.py и compare.py.
"""

import re
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, KFold, StratifiedKFold, RandomizedSearchCV
from sklearn.base import clone
from sklearn.metrics import root_mean_squared_error, r2_score, mean_squared_log_error
from sklearn.feature_selection import mutual_info_regression

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ──────────────────── колонки источника ──────────────────────────────────────

COL_CYCLE         = "Цикл"
COL_COORDS        = "Гелиокоординаты"
COL_CLASS         = "Класс вспышки"
COL_CME           = "Скорость солн. Ветра"
COL_JMAX          = "Максимальная интенсивность"
COL_FLUENCE       = "Флюэс вспышки"
COL_CME_W         = "Угол раствора КВМ"
COL_CME_PA        = "CME_PA_deg"
COL_FL_BEGIN      = "NOAA_Flare_Begin"
COL_FL_END        = "NOAA_Flare_End"
COL_TDELTA        = "Фаза нарастания (часы)"
COL_TFLARE        = "Время долета частиц (часы)"
COL_GOES_PEAK     = "GOES_Peak_Flux"    # W/m², пиковый поток XRS 1-8Å
COL_GOES_RISE_MIN = "GOES_Rise_Min"     # минуты нарастания вспышки до пика

KF       = KFold(n_splits=5, shuffle=True, random_state=42)
KF_STRAT = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

MODEL_COLORS = {
    "Linear":     "#1f77b4",
    "Ridge":      "#aec7e8",
    "Huber":      "#17becf",
    "Forest":     "#8c564b",
    "ExtraTrees": "#c49c94",
    "Boosting":   "#ff7f0e",
    "SVR":        "#2ca02c",
    "GPR_RBF":    "#d62728",
}

# ──────────────────── инженерия признаков ────────────────────────────────────

_XRAY_BASE = {"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}


def xray_to_power(class_str) -> float | None:
    if pd.isna(class_str):
        return None
    m = re.match(r"([ABCMX])(\d+(?:\.\d+)?)", str(class_str).strip().upper())
    return _XRAY_BASE[m.group(1)] * float(m.group(2)) if m else None


def parse_helio_lon(coord_str) -> float | None:
    """'S14W34' -> 34, 'N12E45' -> -45  (W > 0, E < 0)"""
    if pd.isna(coord_str):
        return None
    m = re.search(r"([EW])(\d+)", str(coord_str).upper())
    return float(m.group(2)) * (1 if m.group(1) == "W" else -1) if m else None


def parse_helio_lat(coord_str) -> float | None:
    """'S14W34' -> -14, 'N12E45' -> 12"""
    if pd.isna(coord_str):
        return None
    m = re.search(r"([NS])(\d+)", str(coord_str).upper())
    return float(m.group(2)) * (1 if m.group(1) == "N" else -1) if m else None


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["helio_lon"]     = out[COL_COORDS].apply(parse_helio_lon)
    out["helio_lat"]     = out[COL_COORDS].apply(parse_helio_lat)
    out["flare_power"]   = out[COL_CLASS].apply(xray_to_power)
    out["cme_velocity"]  = pd.to_numeric(out[COL_CME],    errors="coerce")
    out["t_delta_flare"] = pd.to_numeric(out[COL_TFLARE], errors="coerce")
    out["Jmax"]          = pd.to_numeric(out[COL_JMAX],   errors="coerce")
    out["T_delta"]       = pd.to_numeric(out[COL_TDELTA], errors="coerce")

    # ── Новые признаки ────────────────────────────────────────────────────────
    # Флюэс рентгеновского излучения вспышки (J/cm²): интеграл — лучше пика
    fluence = pd.to_numeric(out[COL_FLUENCE], errors="coerce").clip(lower=1e-6)
    out["fluence"]           = fluence
    out["log_fluence"]       = np.log10(fluence)

    # Длительность вспышки по NOAA (минуты)
    fl_begin = pd.to_numeric(out[COL_FL_BEGIN], errors="coerce")
    fl_end   = pd.to_numeric(out[COL_FL_END],   errors="coerce")
    dur = fl_end - fl_begin
    dur = dur.where(dur.between(1, 700), other=np.nan)   # чистим midnight-crossing
    out["flare_dur_min"]     = dur
    out["log_flare_dur_min"] = np.log10(dur.clip(lower=1.0))

    # Угол раствора КВМ (градусы): 360 = гало-КВМ, типично 0-360
    out["cme_width_deg"]     = pd.to_numeric(out[COL_CME_W],  errors="coerce")

    # Позиционный угол КВМ (градусы): связан с направлением выброса
    out["cme_pa_deg"]        = pd.to_numeric(out[COL_CME_PA], errors="coerce")

    # ── GOES XRS: пиковый поток и время нарастания ────────────────────────────
    # Доступны только при чтении листа "Флюэс GOES"; при чтении других листов
    # колонки отсутствуют — безопасно возвращаем NaN.
    if COL_GOES_PEAK in out.columns:
        goes_peak = pd.to_numeric(out[COL_GOES_PEAK], errors="coerce").clip(lower=1e-10)
    else:
        goes_peak = pd.Series(np.nan, index=out.index)
    out["goes_peak_flux"]     = goes_peak
    out["log_goes_peak_flux"] = np.log10(goes_peak)

    if COL_GOES_RISE_MIN in out.columns:
        goes_rise = pd.to_numeric(out[COL_GOES_RISE_MIN], errors="coerce")
    else:
        goes_rise = pd.Series(np.nan, index=out.index)
    out["goes_rise_min"]     = goes_rise
    out["log_goes_rise_min"] = np.log10(goes_rise.clip(lower=1.0))

    # ── Логарифмированные базовые признаки ───────────────────────────────────
    out["log_flare_power"]   = np.log10(out["flare_power"].clip(lower=1e-12))
    out["log_cme_velocity"]  = np.log10(out["cme_velocity"].clip(lower=1.0))
    return out


# ──────────────────── подготовка матриц X / y ────────────────────────────────

def prepare_xy(df: pd.DataFrame, feature_cols: list, target_col: str):
    """Возвращает X_raw, y, valid_idx, cycle_labels — строки без NaN/inf."""
    cols = feature_cols + [target_col]
    work = df[cols].copy()
    for c in cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    mask = work.apply(np.isfinite).all(axis=1)
    work = work[mask]
    # Метки цикла (если есть в df), иначе NaN
    cycle_labels = (
        pd.to_numeric(df.loc[work.index, COL_CYCLE], errors="coerce").values
        if COL_CYCLE in df.columns else np.full(len(work), np.nan)
    )
    return work[feature_cols].to_numpy(), work[target_col].to_numpy(), work.index, cycle_labels


def make_cycle_cv_splits(cycle_labels: np.ndarray) -> list[tuple]:
    """
    Leave-one-cycle-out CV: для каждого уникального цикла возвращает
    (train_indices, val_indices).
    Пример: SC23+SC24 → [(SC24_idx, SC23_idx), (SC23_idx, SC24_idx)]
    """
    cycles = sorted(set(c for c in cycle_labels if not np.isnan(c)))
    splits = []
    for c in cycles:
        val  = np.where(cycle_labels == c)[0]
        trn  = np.where(cycle_labels != c)[0]
        if len(trn) > 0 and len(val) > 0:
            splits.append((trn, val))
    return splits


# ──────────────────── модели с подбором гиперпараметров ──────────────────────

_GPR_KERNEL = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)

_PARAM_GRIDS = {
    "Forest": {
        "n_estimators":    [200, 500],
        "max_depth":       [None, 10, 20],
        "min_samples_leaf":[1, 3, 5],
        "max_features":    ["sqrt", "log2"],
    },
    "ExtraTrees": {
        "n_estimators":    [200, 500],
        "max_depth":       [None, 10, 20],
        "min_samples_leaf":[1, 3, 5],
    },
    "Boosting": {
        "n_estimators":    [100, 200, 300],
        "learning_rate":   [0.05, 0.1, 0.2],
        "max_depth":       [3, 5],
        "subsample":       [0.8, 1.0],
    },
    "SVR": {
        "C":       [1, 10, 100],
        "epsilon": [0.05, 0.1, 0.3],
    },
    "Ridge": {
        "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
    },
    "Huber": {
        "epsilon": [1.1, 1.35, 1.5, 2.0],
        "alpha":   [0.0001, 0.001, 0.01],
    },
    "GPR_RBF": {
        "alpha":               [1e-3, 1e-2, 0.1],          # noise regularization
        "kernel__k1__k2__length_scale": [0.5, 1.0, 2.0],   # RBF length scale
    },
}

_BASE_MODELS = {
    "Linear":     LinearRegression(),
    "Ridge":      Ridge(),
    "Huber":      HuberRegressor(epsilon=1.35, max_iter=500),
    "Forest":     RandomForestRegressor(random_state=42),
    "ExtraTrees": ExtraTreesRegressor(random_state=42),
    "Boosting":   GradientBoostingRegressor(random_state=42),
    "SVR":        SVR(kernel="rbf"),
    "GPR_RBF":    GaussianProcessRegressor(
                      kernel=_GPR_KERNEL,
                      n_restarts_optimizer=5,
                      random_state=42,
                      normalize_y=True,
                  ),
}


def make_models(tune: bool = False, X_tr=None, y_tr=None) -> dict:
    """
    tune=True: оборачивает модели с RandomizedSearchCV (нужны X_tr, y_tr).
    Для LinearRegression подбор не нужен.
    """
    if not tune:
        return {
            "Linear":     LinearRegression(),
            "Ridge":      Ridge(alpha=1.0),
            "Huber":      HuberRegressor(epsilon=1.35, max_iter=500),
            "Forest":     RandomForestRegressor(n_estimators=200, random_state=42),
            "ExtraTrees": ExtraTreesRegressor(n_estimators=200, random_state=42),
            "Boosting":   GradientBoostingRegressor(n_estimators=200, random_state=42),
            "SVR":        SVR(kernel="rbf", C=10.0, epsilon=0.1),
            "GPR_RBF":    GaussianProcessRegressor(
                              kernel=_GPR_KERNEL,
                              n_restarts_optimizer=5,
                              random_state=42,
                              normalize_y=True,
                          ),
        }

    out = {
        "Linear":  LinearRegression(),
        # GPR тюнится через встроенный optimizer ядра при fit(), не через grid search
        "GPR_RBF": GaussianProcessRegressor(
            kernel=_GPR_KERNEL, n_restarts_optimizer=10,
            random_state=42, normalize_y=True,
        ),
    }
    for name, base in _BASE_MODELS.items():
        if name in ("Linear", "GPR_RBF") or name not in _PARAM_GRIDS:
            continue
        search = RandomizedSearchCV(
            base, _PARAM_GRIDS[name],
            n_iter=12, cv=KF, scoring="neg_mean_squared_error",
            random_state=42, n_jobs=-1, refit=True,
        )
        search.fit(X_tr, y_tr)
        out[name] = search.best_estimator_
        print(f"    {name} best: {search.best_params_}")
    return out


# ──────────────────── метрики ─────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, log_target: bool) -> dict:
    """
    Для log_target=True считаем метрики в log10-пространстве (RMSLE, R²_log, Spearman_log).
    RMSE в исходных единицах всегда.
    """
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], np.clip(y_pred[mask], 0, None)

    rmse = root_mean_squared_error(yt, yp)
    metrics = {"RMSE": rmse}

    if log_target and (yt > 0).all():
        yt_log = np.log10(yt)
        yp_log = np.log10(np.clip(yp, 1e-3, None))
        metrics["R2_log"]     = r2_score(yt_log, yp_log)
        metrics["Spearman"]   = spearmanr(yt, yp).statistic
        # RMSLE в log10-единицах (интерпретируемо: 1.0 = ошибка в 10 раз)
        metrics["RMSLE_log10"] = float(np.sqrt(np.mean((yt_log - yp_log) ** 2)))
        # Naive baseline: predict geometric mean of train (passed separately if needed)
    else:
        metrics["R2"]       = r2_score(yt, yp)
        metrics["Spearman"] = spearmanr(yt, yp).statistic
        metrics["MedAE"]    = float(np.median(np.abs(yt - yp)))

    return metrics


# ──────────────────── обучение + оценка ──────────────────────────────────────

def fit_and_score(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    log_target: bool,
    tune: bool = False,
) -> dict:
    """
    Обучает все модели и возвращает словарь с результатами.
    CV-стратегия: cross-cycle (leave-one-cycle-out) если в train_df >= 2 циклов,
    иначе StratifiedKFold(5) по бинам log10(target).
    """
    X_tr_raw, y_tr_raw, idx_tr, cycle_tr = prepare_xy(train_df, feature_cols, target_col)
    X_te_raw, y_te_raw, idx_te, _        = prepare_xy(test_df,  feature_cols, target_col)

    if len(X_tr_raw) == 0:
        raise ValueError(f"Train set empty for features {feature_cols} / target {target_col}")

    # Масштабирование X
    sx = StandardScaler()
    X_tr = sx.fit_transform(X_tr_raw)
    # Тест может быть пустым (напр. новый признак без данных в SC25)
    has_test = len(X_te_raw) > 0
    X_te = sx.transform(X_te_raw) if has_test else np.empty((0, X_tr.shape[1]))

    # Цель в масштабированном пространстве
    sy = StandardScaler()
    if log_target:
        y_tr_log = np.log10(np.clip(y_tr_raw, 1e-6, None))
        y_tr_s   = sy.fit_transform(y_tr_log.reshape(-1, 1)).ravel()
    else:
        y_tr_s = sy.fit_transform(y_tr_raw.reshape(-1, 1)).ravel()

    # ── Выбор CV-стратегии ────────────────────────────────────────────────────
    unique_cycles = sorted(set(c for c in cycle_tr if not np.isnan(c)))
    if len(unique_cycles) >= 2:
        # Cross-cycle CV: leave-one-cycle-out (физически честная оценка)
        cv_splits   = make_cycle_cv_splits(cycle_tr)
        cv_mode     = "cross_cycle"
        cycle_names = [f"SC{int(c)}" for c in unique_cycles]
    else:
        # Fallback: StratifiedKFold по бинам log10(target)
        if log_target:
            strata    = pd.cut(pd.Series(y_tr_s), bins=5, labels=False,
                               duplicates="drop").fillna(0).astype(int).values
            cv_splits = list(KF_STRAT.split(X_tr, strata))
        else:
            cv_splits = KF
        cv_mode     = "stratified_kfold"
        cycle_names = []

    if tune:
        print("  Подбор гиперпараметров...")
        models = make_models(tune=True, X_tr=X_tr, y_tr=y_tr_s)
    else:
        models = make_models(tune=False)

    cv_metrics, test_metrics = {}, {}
    fold_metrics: dict[str, list[dict]] = {}   # [model_name] -> [fold_0_metrics, fold_1_metrics, ...]
    fitted, test_preds = {}, {}

    for name, mdl in models.items():
        # ── Per-fold CV ───────────────────────────────────────────────────────
        fold_results = []
        for fold_i, (tr_idx, val_idx) in enumerate(cv_splits):
            mdl_fold = clone(mdl)
            mdl_fold.fit(X_tr[tr_idx], y_tr_s[tr_idx])
            y_val_s = mdl_fold.predict(X_tr[val_idx])
            y_val   = _inv_scale(y_val_s, sy, log_target)
            fold_results.append(compute_metrics(y_tr_raw[val_idx], y_val, log_target))
        fold_metrics[name] = fold_results

        # Агрегированная CV-метрика (по всем фолдам вместе через cross_val_predict)
        y_cv_s = cross_val_predict(mdl, X_tr, y_tr_s, cv=cv_splits)
        y_cv   = _inv_scale(y_cv_s, sy, log_target)
        cv_metrics[name] = compute_metrics(y_tr_raw, y_cv, log_target)

        # ── Fit на полном train → test ────────────────────────────────────────
        mdl.fit(X_tr, y_tr_s)
        if has_test:
            y_te_s_pred = mdl.predict(X_te)
            y_te_pred   = _inv_scale(y_te_s_pred, sy, log_target)
            test_metrics[name] = compute_metrics(y_te_raw, y_te_pred, log_target)
            test_preds[name]   = y_te_pred
        else:
            # Тестовых данных нет (напр. признак отсутствует в SC25)
            test_metrics[name] = {"RMSE": np.nan, "RMSLE_log10": np.nan,
                                  "R2_log": np.nan, "Spearman": np.nan,
                                  "R2": np.nan, "MedAE": np.nan}
            test_preds[name]   = np.array([])
        fitted[name] = mdl

    return {
        "cv_metrics":   cv_metrics,
        "test_metrics": test_metrics,
        "fold_metrics": fold_metrics,   # per-fold подробности
        "cycle_names":  cycle_names,    # ['SC23', 'SC24'] или []
        "cv_mode":      cv_mode,
        "cv_rmse":      {n: cv_metrics[n]["RMSE"]  for n in cv_metrics},
        "test_rmse":    {n: test_metrics[n]["RMSE"] for n in test_metrics},
        "fitted":       fitted,
        "test_preds":   test_preds,
        "test_true":    y_te_raw,
        "test_idx":     idx_te,
        "X_tr":         X_tr,
        "X_tr_raw":     X_tr_raw,
        "y_tr_s":       y_tr_s,
        "y_tr_raw":     y_tr_raw,
        "log_target":   log_target,
        "sx": sx, "sy": sy,
    }


def _inv_scale(y_s, sy, log_target):
    y = sy.inverse_transform(y_s.reshape(-1, 1)).ravel()
    if log_target:
        y = 10.0 ** y      # обратно из log10
    return np.clip(y, 0, None)


# ──────────────────── важность признаков ──────────────────────────────────────

def _builtin_importance(model, n_feats: int) -> np.ndarray:
    if hasattr(model, "feature_importances_"):
        fi = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        fi = np.abs(np.asarray(model.coef_, dtype=float).ravel()[:n_feats])
    else:
        return np.zeros(n_feats, dtype=float)
    fi = np.nan_to_num(fi)
    s = fi.sum()
    return fi / s * 100 if s > 0 else fi


def _mi_importance(X_raw: np.ndarray, y: np.ndarray, feature_cols: list) -> np.ndarray:
    """MI на сырых признаках с log10-трансформацией широкодиапазонных колонок."""
    X_mi = X_raw.copy().astype(float)
    for i, col in enumerate(feature_cols):
        if col in ("flare_power", "cme_velocity"):
            X_mi[:, i] = np.log10(np.abs(X_mi[:, i]) + 1e-12)
    mi = mutual_info_regression(X_mi, y, random_state=42)
    mi = np.nan_to_num(mi)
    s = mi.sum()
    return mi / s * 100 if s > 0 else mi


def compute_importances(result: dict, feature_cols: list) -> dict:
    """
    Возвращает dict[(model_name, method)] = array (n_feats,) в %
    Методы: builtin, shap, mutual_info
    """
    fitted   = result["fitted"]
    X_tr     = result["X_tr"]
    X_tr_raw = result["X_tr_raw"]
    y_tr_s   = result["y_tr_s"]
    n_feats  = X_tr.shape[1]
    out = {}

    # 1. Builtin
    for name, mdl in fitted.items():
        out[(name, "builtin")] = _builtin_importance(mdl, n_feats)

    # 2. SHAP
    try:
        import shap
        _shap_ok = True
    except ImportError:
        _shap_ok = False

    for name, mdl in fitted.items():
        sv = None
        if _shap_ok:
            try:
                if hasattr(mdl, "feature_importances_"):
                    sv = shap.TreeExplainer(mdl).shap_values(X_tr)
                elif hasattr(mdl, "coef_"):
                    sv = shap.LinearExplainer(
                        mdl, X_tr, feature_perturbation="interventional"
                    ).shap_values(X_tr)
                elif isinstance(mdl, GaussianProcessRegressor):
                    # GPR: KernelExplainer слишком медленный — пропускаем SHAP
                    sv = None
                else:
                    bg = shap.sample(X_tr, min(80, len(X_tr)))
                    sv = shap.KernelExplainer(mdl.predict, bg).shap_values(
                        X_tr, nsamples=100, silent=True
                    )
            except Exception as e:
                warnings.warn(f"SHAP {name}: {e}")

        if sv is not None:
            if isinstance(sv, list):
                sv = sv[0]
            si = np.mean(np.abs(sv), axis=0)
        else:
            si = np.zeros(n_feats, dtype=float)

        si = np.nan_to_num(si)
        s = si.sum()
        out[(name, "shap")] = si / s * 100 if s > 0 else si

    # 3. Mutual information (одно значение для всех моделей)
    mi_pct = _mi_importance(X_tr_raw, result["y_tr_raw"], feature_cols)
    for name in fitted:
        out[(name, "mutual_info")] = mi_pct

    return out


# ──────────────────── графики ─────────────────────────────────────────────────

METHODS = ["builtin", "shap", "mutual_info"]
METHOD_TITLES = {
    "builtin":     "Builtin (coef / feature_importances_)",
    "shap":        "SHAP (mean |phi|)",
    "mutual_info": "Mutual information",
}


def plot_importances(
    importances: dict,
    feature_labels: list,
    title: str,
    out_path: Path,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    models  = sorted({k[0] for k in importances})
    methods = [m for m in METHODS if any((n, m) in importances for n in models)]
    n_feats  = len(feature_labels)
    n_models = len(models)

    fig, axes = plt.subplots(
        len(methods), 1,
        figsize=(max(10, n_models * 2.5), 3.5 * len(methods)),
    )
    if len(methods) == 1:
        axes = [axes]

    bar_w = 0.7 / n_feats
    cmap  = plt.cm.get_cmap("Set2", n_feats)
    xg    = np.arange(n_models)

    for ax, method in zip(axes, methods):
        for fi, lbl in enumerate(feature_labels):
            offset = (fi - (n_feats - 1) / 2) * bar_w
            vals = [importances.get((m, method), np.zeros(n_feats))[fi] for m in models]
            bars = ax.bar(xg + offset, vals, width=bar_w,
                          color=cmap(fi), label=lbl, edgecolor="white", linewidth=0.5)
            for rect, v in zip(bars, vals):
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    v + 0.3,
                    f"{v:.1f}%",
                    ha="center", va="bottom", fontsize=6.5,
                )

        ax.set_xticks(xg)
        ax.set_xticklabels(models, fontsize=9)
        ax.set_ylabel("Вклад, %")
        ax.set_title(METHOD_TITLES.get(method, method), fontsize=9, loc="left")
        ax.set_ylim(0, None)
        ax.grid(axis="y", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.0, 1.0),
               ncol=1, title="Признак", fontsize=9)
    fig.suptitle(title, fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_scatter(
    result: dict,
    feature_labels: list,
    title: str,
    out_path: Path,
    log_scale: bool = False,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    preds   = result["test_preds"]
    y_true  = result["test_true"]
    n = len(preds)
    cols_n = min(3, n)
    rows = (n + cols_n - 1) // cols_n

    fig, axes = plt.subplots(rows, cols_n, figsize=(5 * cols_n, 4.5 * rows))
    axes_flat = np.array(axes).flatten()

    for ax, (name, y_pred) in zip(axes_flat, preds.items()):
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        yt, yp = y_true[mask], np.clip(y_pred[mask], 0, None)
        m = result["test_metrics"][name]
        color = MODEL_COLORS.get(name, "#333")

        lo = min(yt.min(), yp.min())
        hi = max(yt.max(), yp.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5)
        ax.scatter(yt, yp, s=20, color=color, alpha=0.7, edgecolor="w", lw=0.3)

        if log_scale:
            ax.set_xscale("log"); ax.set_yscale("log")
            subtitle = (f"RMSLE={m.get('RMSLE_log10',0):.2f} log10-ед  "
                        f"R2(log)={m.get('R2_log',0):.2f}  "
                        f"Spearman={m.get('Spearman',0):.2f}")
        else:
            subtitle = (f"RMSE={m['RMSE']:.1f}  "
                        f"R2={m.get('R2',0):.2f}  "
                        f"Spearman={m.get('Spearman',0):.2f}")

        ax.set_title(f"{name}\n{subtitle}", fontsize=8)
        ax.set_xlabel("Факт"); ax.set_ylabel("Прогноз")
        ax.grid(alpha=0.3, which="both" if log_scale else "major")

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_residuals(
    result: dict,
    title: str,
    out_path: Path,
    log_target: bool = False,
):
    """Распределение остатков (residuals) для каждой модели."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    preds  = result["test_preds"]
    y_true = result["test_true"]
    n = len(preds)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (name, y_pred) in zip(axes, preds.items()):
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        yt, yp = y_true[mask], np.clip(y_pred[mask], 0, None)

        if log_target:
            resid = np.log10(np.clip(yp, 1e-3, None)) - np.log10(yt)
            xlabel = "log10(pred) - log10(true)"
        else:
            resid = yp - yt
            xlabel = "pred - true"

        color = MODEL_COLORS.get(name, "#333")
        ax.hist(resid, bins=20, color=color, alpha=0.7, edgecolor="w")
        ax.axvline(0, color="k", lw=1.5, ls="--")
        ax.axvline(np.median(resid), color="red", lw=1, ls=":", label=f"median={np.median(resid):.2f}")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle(f"Распределение остатков — {title}", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")
