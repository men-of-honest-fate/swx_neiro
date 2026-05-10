"""
compare_metrics.py
==================
Считает метрики hybrid W/E × no_vel для двух порогов J_max (≥10 и ≥1)
и печатает текстовое сравнение. Сохраняет CSV в этой же папке.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT         = Path(__file__).parent
PROJECT_ROOT = ROOT.parent
STAGE0       = PROJECT_ROOT / "stage 0"

sys.path.insert(0, str(STAGE0))
sys.path.insert(0, str(PROJECT_ROOT))

import metrics_summary as M
from spe_utils import build_features, COL_CYCLE


def make_load_splits(jmax_min: float):
    def _load():
        df = build_features(pd.read_excel(
            PROJECT_ROOT / "data" / "ОБЪЕДИНЕННЫЙ КАТАЛОГ СПС 23-25.xlsx",
            sheet_name="Флюэс GOES",
        ))
        cycle  = pd.to_numeric(df[COL_CYCLE], errors="coerce")
        tdelta = pd.to_numeric(df["T_delta"], errors="coerce")
        rise   = pd.to_numeric(df["goes_rise_min"], errors="coerce")
        mask = (
            (df["Jmax"].fillna(0) >= jmax_min)
            & (tdelta.fillna(0) <= 40)
            & (rise.fillna(0) <= 120)
        )
        full   = df[mask].copy()
        tr_all = full[cycle.isin([23, 24])].copy()
        te_all = full[cycle.isin([25])].copy()
        return {
            "West": (tr_all[tr_all["helio_lon"] > 0].copy(),
                     te_all[te_all["helio_lon"] > 0].copy()),
            "East": (tr_all[tr_all["helio_lon"] < 0].copy(),
                     te_all[te_all["helio_lon"] < 0].copy()),
            "All":  (tr_all.copy(), te_all.copy()),
        }
    return _load


_orig_target_weights = M.target_weights


def _make_target_weights(clip_lo: float):
    def _tw(jmax):
        yl  = np.log10(np.clip(jmax, clip_lo, None))
        raw = (yl - yl.min() + 0.5) ** M.ALPHA_TW
        return raw / raw.mean()
    return _tw


def run_for_threshold(jmax_min: float) -> pd.DataFrame:
    M.load_splits     = make_load_splits(jmax_min)
    # Нижний clip target-весов = порогу выборки: для ≥10 — clip 10 (как раньше),
    # для ≥1 — clip 1, чтобы события 1–10 pfu участвовали в реверсивном reweighting.
    M.target_weights  = _make_target_weights(jmax_min)
    splits = M.load_splits()
    targets = [("Jmax", True), ("T_delta", False)]
    rows = []
    M.run_experiment("ew_no_vel", M.FS_NO_VEL, ["hybrid"], splits, targets, rows)
    df = pd.DataFrame(rows)
    df["jmax_min"] = jmax_min
    return df


def best_per_group_target_fs(df: pd.DataFrame) -> pd.DataFrame:
    """Лучшая модель по metric_value (минимум) для каждой (group, target, fs)."""
    out = []
    for (g, t, fs), sub in df.groupby(["group", "target", "feature_set"], sort=False):
        sub_ok = sub.dropna(subset=["metric_value"])
        if sub_ok.empty:
            continue
        idx = sub_ok["metric_value"].idxmin()
        row = sub_ok.loc[idx]
        out.append({
            "group": g, "target": t, "feature_set": fs,
            "best_model": row["model"],
            "metric_name": row["metric_name"],
            "metric_value": float(row["metric_value"]),
            "R2": float(row["R2"]) if pd.notna(row["R2"]) else np.nan,
            "CC": float(row["CC"]) if pd.notna(row["CC"]) else np.nan,
            "n_train": int(row["n_train"]),
            "n_test":  int(row["n_test"]),
        })
    return pd.DataFrame(out)


FS_ORDER     = ["Базовая", "Флюэс вместо пика", "Обе координаты", "Координаты+флюэс"]
GROUP_ORDER  = ["West", "East", "All"]
TARGET_ORDER = ["Jmax", "T_delta"]


def fmt_table(b10: pd.DataFrame, b1: pd.DataFrame, target: str, group: str) -> str:
    hdr_metric = "RMSLE log₁₀" if target == "Jmax" else "RMSE (ч)"
    lines = []
    lines.append(f"── {group} · {target} ({hdr_metric}, ↓ = лучше) ──")
    lines.append(f"{'Набор':<22} {'≥10 pfu':>12} {'≥1 pfu':>12} {'Δ':>9}  {'модель ≥10 → ≥1':<30}  {'n_test ≥10/≥1':>14}")
    for fs in FS_ORDER:
        r10 = b10[(b10.group == group) & (b10.target == target) & (b10.feature_set == fs)]
        r1  = b1 [(b1 .group == group) & (b1 .target == target) & (b1 .feature_set == fs)]
        if r10.empty or r1.empty:
            continue
        v10, v1 = r10.iloc[0]["metric_value"], r1.iloc[0]["metric_value"]
        m10, m1 = r10.iloc[0]["best_model"],   r1.iloc[0]["best_model"]
        n10, n1 = r10.iloc[0]["n_test"],        r1.iloc[0]["n_test"]
        delta = v1 - v10
        sign  = "↓" if delta < 0 else ("↑" if delta > 0 else "=")
        lines.append(
            f"{fs:<22} {v10:>12.3f} {v1:>12.3f} {sign}{abs(delta):>7.3f}  "
            f"{m10 + ' → ' + m1:<30}  {str(n10) + '/' + str(n1):>14}"
        )
    return "\n".join(lines)


def main():
    print("=== Прогон порог ≥10 pfu ===")
    df_10 = run_for_threshold(10.0)
    print("=== Прогон порог ≥1 pfu ===")
    df_1  = run_for_threshold(1.0)

    raw = pd.concat([df_10, df_1], ignore_index=True)
    raw_path = ROOT / "metrics_no_vel_hybrid_raw.csv"
    raw.to_csv(raw_path, index=False, encoding="utf-8-sig")
    print(f"\nRaw: {raw_path}")

    b10 = best_per_group_target_fs(df_10)
    b1  = best_per_group_target_fs(df_1)

    best_combined = pd.concat([
        b10.assign(jmax_min=10.0),
        b1.assign(jmax_min=1.0),
    ], ignore_index=True)
    best_path = ROOT / "metrics_no_vel_hybrid_best.csv"
    best_combined.to_csv(best_path, index=False, encoding="utf-8-sig")
    print(f"Best: {best_path}")

    print()
    print("############################################################")
    print("# hybrid W/E · no_vel — сравнение порогов J_max ≥10 vs ≥1 #")
    print("############################################################")
    print()
    for tgt in TARGET_ORDER:
        for g in GROUP_ORDER:
            print(fmt_table(b10, b1, tgt, g))
            print()


if __name__ == "__main__":
    main()
