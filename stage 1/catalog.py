"""
Чтение и фильтрация каталога СПС для этапа 1 (LSTM-наукастинг).

Конфигурация — копия `hybrid_no_vel_jmax1/run.py` (J_max ≥ 1 pfu, T_delta ≤ 40 ч,
GOES_rise ≤ 120 мин), но возвращаем единый DataFrame со всеми событиями
(W+E, train+test) и явно разобранными `onset_dt` / `peak_dt`.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT         = Path(__file__).parent
PROJECT_ROOT = ROOT.parent

sys.path.insert(0, str(PROJECT_ROOT))

from spe_utils import build_features, COL_CYCLE  # noqa: E402

CATALOG_PATH  = PROJECT_ROOT / "data" / "ОБЪЕДИНЕННЫЙ КАТАЛОГ СПС 23-25.xlsx"
CATALOG_SHEET = "Флюэс GOES"

JMAX_MIN       = 1.0   # порог отбора J_max
TDELTA_MAX_H   = 40.0  # часы
GOES_RISE_MAX_MIN = 120.0


def _to_datetime(val) -> pd.Timestamp | None:
    if pd.isna(val):
        return None
    try:
        return pd.to_datetime(val)
    except Exception:
        return None


def _group(helio_lon: float | None) -> str | None:
    if helio_lon is None or pd.isna(helio_lon):
        return None
    if helio_lon > 0:
        return "West"
    if helio_lon < 0:
        return "East"
    return "Central"


def load_catalog() -> pd.DataFrame:
    """Читает каталог, фильтрует, добавляет служебные колонки.

    Возвращаемые колонки сверху обычных полей:
      event_id  — уникальный id вида YYYYMMDD_NN
      onset_dt  — pd.Timestamp начала СПС
      peak_dt   — pd.Timestamp пика СПС
      group     — "West" / "East" / "Central"
      split     — "train" (SC23+24) или "test" (SC25)
    """
    df = build_features(pd.read_excel(CATALOG_PATH, sheet_name=CATALOG_SHEET))

    cycle     = pd.to_numeric(df[COL_CYCLE],       errors="coerce")
    tdelta    = pd.to_numeric(df["T_delta"],       errors="coerce")
    goes_rise = pd.to_numeric(df["goes_rise_min"], errors="coerce")

    mask = (
        (df["Jmax"].fillna(0) >= JMAX_MIN)
        & (tdelta.fillna(0) <= TDELTA_MAX_H)
        & (goes_rise.fillna(0) <= GOES_RISE_MAX_MIN)
        & cycle.isin([23, 24, 25])
    )
    out = df.loc[mask].copy().reset_index(drop=True)

    out["onset_dt"] = out["Время начала"].apply(_to_datetime)
    out["peak_dt"]  = out["Время максимума"].apply(_to_datetime)

    bad = out["onset_dt"].isna() | out["peak_dt"].isna() | (out["peak_dt"] <= out["onset_dt"])
    n_bad = int(bad.sum())
    if n_bad:
        print(f"[catalog] откинуто {n_bad} событий с битым onset/peak")
    out = out.loc[~bad].copy().reset_index(drop=True)

    out["group"] = out["helio_lon"].apply(_group)
    out["split"] = np.where(pd.to_numeric(out[COL_CYCLE], errors="coerce") == 25,
                            "test", "train")

    out["event_id"] = (
        out["onset_dt"].dt.strftime("%Y%m%d") + "_"
        + out.groupby(out["onset_dt"].dt.strftime("%Y%m%d")).cumcount().add(1).astype(str).str.zfill(2)
    )

    return out


def summary(df: pd.DataFrame) -> None:
    n_train = (df["split"] == "train").sum()
    n_test  = (df["split"] == "test").sum()
    print(f"== Каталог: {len(df)} событий (train={n_train}, test={n_test}) ==")
    by_group = df.groupby(["group", "split"]).size().unstack(fill_value=0)
    print(by_group)


if __name__ == "__main__":
    df = load_catalog()
    summary(df)
    print(df[["event_id", "Цикл", "group", "split",
              "onset_dt", "peak_dt", "Jmax", "T_delta"]].head(10).to_string(index=False))
