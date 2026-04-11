"""
Дополняет каталог СПС данными рентгеновской вспышки из GOES XRS.

Алгоритм:
  1. Читает лист "Фильтр для обучения" из объединённого каталога
  2. Для каждого события с известным временем вспышки (NOAA_Flare_Begin_dt)
     скачивает GOES XRS FITS/NetCDF данные через SunPy (кеш на диске)
  3. По окну [begin, end] вычисляет:
       - Флюэс вспышки  [J/m²]   — трапециевидное интегрирование xrsb
       - GOES_Peak_Flux [W/m²]   — максимальный поток 1-8Å
       - GOES_Peak_Time [UTC]    — время максимума
       - GOES_Rise_Min  [мин]    — время нарастания (begin → peak)
  4. Копирует лист "Фильтр для обучения" → "Флюэс GOES" и обновляет колонки

Запуск:  python data/goes_fluence.py
"""

import sys
import copy
import warnings
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).parent.parent
CATALOG   = ROOT / "data" / "ОБЪЕДИНЕННЫЙ КАТАЛОГ СПС 23-25.xlsx"
SHEET_SRC = "Фильтр для обучения"
SHEET_DST = "Флюэс GOES"
CACHE_DIR = ROOT / "goes_xrs_cache"
CACHE_DIR.mkdir(exist_ok=True)

TODAY = pd.Timestamp.today().normalize()

# ── SunPy ─────────────────────────────────────────────────────────────────────
try:
    from sunpy.net import Fido, attrs as a
    import sunpy.timeseries as ts
except ImportError:
    print("SunPy не установлен. Запустите: pip install 'sunpy[net,timeseries]' mpl-animators")
    sys.exit(1)

# ── Вспомогательные функции ───────────────────────────────────────────────────

def _parse_hhmm(val, base_date) -> pd.Timestamp | None:
    """HHMM float (552.0 → 05:52) к Timestamp на дату base_date."""
    if pd.isna(val):
        return None
    hhmm = int(val)
    h, m = hhmm // 100, hhmm % 100
    if not (0 <= h <= 23 and 0 <= m <= 59):
        return None
    return pd.Timestamp(base_date).replace(hour=h, minute=m, second=0, microsecond=0)


def _flare_end(begin: pd.Timestamp, end_hhmm, pad_min: int = 2) -> pd.Timestamp:
    """Конец вспышки + небольшой запас. Если нет данных — begin + 2 часа."""
    if pd.isna(end_hhmm):
        return begin + timedelta(hours=2)
    end = _parse_hhmm(end_hhmm, begin.date())
    if end is None:
        return begin + timedelta(hours=2)
    if end <= begin:
        end += timedelta(days=1)
    return end + timedelta(minutes=pad_min)


# ── Кеш GOES XRS (по дням) ────────────────────────────────────────────────────

_day_cache: dict[str, pd.DataFrame | None] = {}


def _cache_path(date: pd.Timestamp) -> Path:
    return CACHE_DIR / f"goes_xrs_{date.strftime('%Y%m%d')}.parquet"


def _best_xrsb_col(df: pd.DataFrame) -> str | None:
    """Возвращает имя колонки 1-8Å (xrsb) или None."""
    for col in df.columns:
        cl = col.lower()
        if "xrsb" in cl or "b_flux" in cl:
            return col
    for col in df.columns:
        if "long" in col.lower():
            return col
    if len(df.columns) >= 2:
        return df.columns[1]
    return None


def load_day(date: pd.Timestamp) -> pd.DataFrame | None:
    """
    Загружает GOES XRS за дату. Кеширует в parquet.
    Возвращает DataFrame с колонкой 'flux' (W/m², канал 1-8 Å).
    """
    key = date.strftime("%Y%m%d")
    if key in _day_cache:
        return _day_cache[key]

    pq = _cache_path(date)
    if pq.exists():
        df = pd.read_parquet(pq)
        _day_cache[key] = df
        return df

    date_str = date.strftime("%Y-%m-%d")
    print(f"    Скачиваю GOES XRS {date_str} ...", end=" ", flush=True)

    try:
        result = Fido.search(
            a.Time(date_str, date_str),
            a.Instrument.xrs,
        )
        if result.file_num == 0:
            print("нет данных")
            _day_cache[key] = None
            return None

        files = Fido.fetch(result[0, 0],
                           path=str(CACHE_DIR / "{file}"),
                           progress=False)
        if not files:
            print("ошибка загрузки")
            _day_cache[key] = None
            return None

        goes_ts = ts.TimeSeries(files[0])
        df_raw = goes_ts.to_dataframe()

        col = _best_xrsb_col(df_raw)
        if col is None:
            print("нет xrsb колонки")
            _day_cache[key] = None
            return None

        df = df_raw[[col]].rename(columns={col: "flux"}).copy()
        df["flux"] = pd.to_numeric(df["flux"], errors="coerce").clip(lower=0)
        df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
        df = df.sort_index()

        df.to_parquet(pq)
        print(f"OK ({len(df)} точек)")
        _day_cache[key] = df
        return df

    except Exception as e:
        print(f"ОШИБКА: {e}")
        _day_cache[key] = None
        return None


# ── Вычисление XRS-параметров ─────────────────────────────────────────────────

def compute_xrs_params(begin: pd.Timestamp, end: pd.Timestamp) -> dict | None:
    """
    По окну [begin, end] вычисляет из GOES XRS 1-8Å:
      fluence    — J/m²
      peak_flux  — W/m²
      peak_time  — Timestamp UTC
      rise_min   — минуты от begin до peak
    Возвращает dict или None если данных нет.
    """
    days = pd.date_range(begin.date(), end.date(), freq="D")
    frames = []
    for d in days:
        df_day = load_day(pd.Timestamp(d))
        if df_day is not None:
            frames.append(df_day)

    if not frames:
        return None

    df = pd.concat(frames).sort_index()
    mask = (df.index >= begin) & (df.index <= end)
    sub = df.loc[mask, "flux"].dropna()

    if len(sub) < 2:
        return None

    # Флюэнс
    t_sec = np.array([(t - sub.index[0]).total_seconds() for t in sub.index])
    fluence = float(np.trapz(sub.values, t_sec))
    if fluence <= 0:
        return None

    # Пик
    peak_idx = sub.idxmax()
    peak_flux = float(sub[peak_idx])
    rise_min  = (peak_idx - begin).total_seconds() / 60.0

    return {
        "fluence":   fluence,
        "peak_flux": peak_flux,
        "peak_time": peak_idx,
        "rise_min":  rise_min,
    }


# ── Главная функция ───────────────────────────────────────────────────────────

def main():
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print(f"Каталог: {CATALOG}")
    df = pd.read_excel(CATALOG, sheet_name=SHEET_SRC)
    print(f"Загружено строк: {len(df)}")
    print(f"Охват: до {TODAY.date()}")

    df["_begin"] = pd.to_datetime(df["NOAA_Flare_Begin_dt"], errors="coerce")
    df["_end"]   = df.apply(
        lambda r: _flare_end(r["_begin"], r["NOAA_Flare_End"])
        if pd.notna(r["_begin"]) else None,
        axis=1,
    )

    unique_flares = (
        df[df["_begin"].notna()]
        .drop_duplicates(subset=["_begin"])
        [["_begin", "_end"]]
        .copy()
    )
    print(f"Уникальных вспышек для обработки: {len(unique_flares)}")
    print(f"Событий без времени вспышки: {df['_begin'].isna().sum()}")
    print()

    params_map: dict[pd.Timestamp, dict | None] = {}

    for i, (_, row) in enumerate(unique_flares.iterrows(), 1):
        begin, end = row["_begin"], row["_end"]
        print(f"  [{i:3d}/{len(unique_flares)}] {begin}  ->  {end.strftime('%H:%M')}", end=" ")
        p = compute_xrs_params(begin, end)
        params_map[begin] = p
        if p is not None:
            print(f"→ fluence={p['fluence']:.4f} J/m²  peak={p['peak_flux']:.2e} W/m²"
                  f"  @{p['peak_time'].strftime('%H:%M')}  rise={p['rise_min']:.0f} min")
        else:
            print("→ нет данных")

    # ── Запись колонок ────────────────────────────────────────────────────────
    n_updated = n_missing = n_nodata = 0

    for idx, row in df.iterrows():
        begin = row["_begin"]
        if pd.isna(begin):
            n_missing += 1
            continue
        p = params_map.get(begin)
        if p is not None:
            df.at[idx, "Флюэс вспышки"]  = p["fluence"]
            df.at[idx, "GOES_Peak_Flux"]  = p["peak_flux"]
            df.at[idx, "GOES_Peak_Time"]  = p["peak_time"]
            df.at[idx, "GOES_Rise_Min"]   = p["rise_min"]
            n_updated += 1
        else:
            n_nodata += 1

    df = df.drop(columns=["_begin", "_end"])

    print(f"\n{'='*55}")
    print(f"  Обновлено:            {n_updated}")
    print(f"  Нет данных GOES:      {n_nodata}")
    print(f"  Нет времени вспышки:  {n_missing}")
    print(f"{'='*55}")

    # ── Запись в Excel ────────────────────────────────────────────────────────
    print(f"\nЗапись в '{CATALOG.name}', лист '{SHEET_DST}'...")

    with pd.ExcelWriter(CATALOG, engine="openpyxl",
                        mode="a", if_sheet_exists="replace") as writer:
        df.to_excel(writer, sheet_name=SHEET_DST, index=False)

        wb = writer.book
        if SHEET_DST in wb.sheetnames and SHEET_SRC in wb.sheetnames:
            ws_dst = wb[SHEET_DST]
            ws_src = wb[SHEET_SRC]
            for row_dst, row_src in zip(ws_dst.iter_rows(), ws_src.iter_rows()):
                for cell_dst, cell_src in zip(row_dst, row_src):
                    try:
                        cell_dst.fill = copy.copy(cell_src.fill)
                    except Exception:
                        pass

    print(f"Готово. Лист '{SHEET_DST}' сохранён.")
    new_cols = ["Флюэс вспышки", "GOES_Peak_Flux", "GOES_Peak_Time", "GOES_Rise_Min"]
    for col in new_cols:
        if col in df.columns:
            print(f"  {col}: {df[col].notna().sum()} / {len(df)} заполнено")


if __name__ == "__main__":
    main()
