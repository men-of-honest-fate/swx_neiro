"""
Загрузка 5-минутных рядов протонного потока GOES ≥10 МэВ из NCEI.

Три формата по эпохам:
  • EPS    (GOES 8-12, 1995–2009): помесячный  g{NN}_eps_5m_*.nc
            proxy ≥10 МэВ:  sum(p3..p7_flux_ic)  [≥8.7 МэВ integral]
  • EPEAD  (GOES 13-15, 2010–2019): помесячный  g{NN}_epead_p17ew_5m_*.nc
            proxy ≥10 МэВ:  Σ ΔE·avg(P{i}E_COR_FLUX, P{i}W_COR_FLUX) для P3..P6
  • SGPS   (GOES 16-19, 2020+): дневной  sci_sgps-l2-avg5m_g{NN}_d{YYYYMMDD}_v*.nc
            proxy ≥10 МэВ:  Σ ΔE·AvgDiffProtonFlux по channels с lower≥10 МэВ
                              (SGPS ch5..ch12), усреднение по 2 sensor_units

Готовый NOAA SWPC `≥10 МэВ integral` в L2 архиве отсутствует — используется proxy.
Абсолютные значения масштабируются к каталожному `Jmax_real` через нормализацию
prior отдельно (LSTM работает в логарифмических приращениях).

Кеши:
  goes_proton_cache/_raw/{eps|epead|sgps}_g{NN}_{YYYYMM[DD]}.nc — сырые netCDF
  goes_proton_cache/proton_p10_g{NN}_{YYYYMMDD}.parquet — распарсенные дневные
"""

from __future__ import annotations

import re
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

ROOT         = Path(__file__).parent
PROJECT_ROOT = ROOT.parent
CACHE_DIR    = PROJECT_ROOT / "goes_proton_cache"
RAW_DIR      = CACHE_DIR / "_raw"
CACHE_DIR.mkdir(exist_ok=True)
RAW_DIR.mkdir(exist_ok=True)

NCEI_AVG = "https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/avg"
NCEI_R   = "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes"

# ── Реестр эпох. Список satellites — в порядке предпочтения (primary первым). ─

EPOCHS: list[dict] = [
    # SC23
    {"start": "1995-01-01", "end": "1998-06-30", "format": "eps",   "satellites": [8, 9]},
    {"start": "1998-07-01", "end": "2003-04-30", "format": "eps",   "satellites": [8, 10, 11]},
    {"start": "2003-05-01", "end": "2008-02-29", "format": "eps",   "satellites": [11, 10, 12]},
    {"start": "2008-03-01", "end": "2009-12-31", "format": "eps",   "satellites": [12, 10]},
    # SC24
    {"start": "2010-01-01", "end": "2017-12-31", "format": "epead", "satellites": [13, 15, 14]},
    {"start": "2018-01-01", "end": "2019-12-31", "format": "epead", "satellites": [15, 14]},
    # SC25
    {"start": "2020-01-01", "end": "2099-12-31", "format": "sgps",  "satellites": [16, 18, 17, 19]},
]


def select_epoch(date: pd.Timestamp) -> dict | None:
    d = pd.Timestamp(date).normalize()
    for ep in EPOCHS:
        if pd.Timestamp(ep["start"]) <= d <= pd.Timestamp(ep["end"]):
            return ep
    return None


# ── HTTP / negative cache ────────────────────────────────────────────────────

_neg_cache: set[str] = set()


def _get(url: str, timeout: int = 60) -> requests.Response | None:
    if url in _neg_cache:
        return None
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            return r
        if r.status_code == 404:
            _neg_cache.add(url)
    except Exception as e:
        print(f"    HTTP err {url}: {e}")
    return None


def _list_dir(url: str) -> list[str]:
    r = _get(url, timeout=30)
    if r is None:
        return []
    return [m for m in re.findall(r'href="([^"?][^"]*)"', r.text)
            if not m.startswith("http") and not m.startswith("/")]


def _month_last_day(year: int, month: int) -> int:
    return (datetime(year + (month == 12), (month % 12) + 1, 1) - timedelta(days=1)).day


# ── Сырые загрузки + парсинг через xarray ────────────────────────────────────

def _open_nc(path: Path):
    import xarray as xr
    return xr.open_dataset(path)


def _eps_url(year: int, month: int, sat: int) -> str:
    last = _month_last_day(year, month)
    return (f"{NCEI_AVG}/{year:04d}/{month:02d}/goes{sat:02d}/netcdf/"
            f"g{sat:02d}_eps_5m_{year:04d}{month:02d}01_{year:04d}{month:02d}{last:02d}.nc")


def _epead_url(year: int, month: int, sat: int) -> str:
    last = _month_last_day(year, month)
    return (f"{NCEI_AVG}/{year:04d}/{month:02d}/goes{sat:02d}/netcdf/"
            f"g{sat:02d}_epead_p17ew_5m_{year:04d}{month:02d}01_{year:04d}{month:02d}{last:02d}.nc")


def _sgps_url(date: pd.Timestamp, sat: int) -> str | None:
    base = (f"{NCEI_R}/goes{sat}/l2/data/sgps-l2-avg5m/"
            f"{date.year:04d}/{date.month:02d}/")
    needle = f"sci_sgps-l2-avg5m_g{sat}_d{date.year:04d}{date.month:02d}{date.day:02d}_"
    for f in _list_dir(base):
        if f.startswith(needle) and f.endswith(".nc"):
            return base + f
    return None


def _download(url: str | None, raw_path: Path, label: str) -> Path | None:
    if raw_path.exists():
        return raw_path
    if url is None:
        return None
    print(f"    {label} ...", end=" ", flush=True)
    r = _get(url)
    if r is None:
        print("404")
        return None
    raw_path.write_bytes(r.content)
    print(f"OK ({len(r.content)/1024:.0f} KB)")
    return raw_path


# ── EPS extract (GOES 8-12) ──────────────────────────────────────────────────

def _eps_extract(ds) -> pd.DataFrame:
    """sum p3..p7_flux_ic → ≥8.7 МэВ integral (≈ pfu)."""
    if "time_tag" not in ds.data_vars and "time_tag" not in ds.coords:
        return pd.DataFrame()
    t = pd.to_datetime(ds["time_tag"].values)
    parts = [ds[v].values for v in ("p3_flux_ic", "p4_flux_ic", "p5_flux_ic",
                                     "p6_flux_ic", "p7_flux_ic")
             if v in ds.data_vars]
    if not parts:
        return pd.DataFrame()
    flux = np.sum(parts, axis=0).astype(float)
    flux[flux < 0] = np.nan
    return pd.DataFrame({"flux": flux}, index=t)


# ── EPEAD extract (GOES 13-15) ───────────────────────────────────────────────
# Energy bins (effective edges, MeV) для расчёта ΔE:
EPEAD_EDGES = {
    3: (8.7, 14.5),
    4: (15.0, 40.0),
    5: (38.0, 82.0),
    6: (84.0, 200.0),
}  # P3..P6 — без P7 (cosmic ray contamination на спокойном фоне)


def _epead_extract(ds) -> pd.DataFrame:
    if "time_tag" not in ds.data_vars and "time_tag" not in ds.coords:
        return pd.DataFrame()
    t = pd.to_datetime(ds["time_tag"].values)
    integral = np.zeros(len(t), dtype=float)
    used = 0
    for i, (lo, hi) in EPEAD_EDGES.items():
        e_var = f"P{i}E_COR_FLUX"
        w_var = f"P{i}W_COR_FLUX"
        if e_var not in ds.data_vars or w_var not in ds.data_vars:
            continue
        ew = np.where(np.isfinite(ds[e_var].values), ds[e_var].values, np.nan)
        ww = np.where(np.isfinite(ds[w_var].values), ds[w_var].values, np.nan)
        avg = np.nanmean(np.stack([ew, ww]), axis=0)   # average E/W
        integral += np.nan_to_num(avg, nan=0.0) * (hi - lo)
        used += 1
    if used == 0:
        return pd.DataFrame()
    integral[integral < 0] = np.nan
    return pd.DataFrame({"flux": integral}, index=t)


# ── SGPS extract (GOES 16-19) ────────────────────────────────────────────────

def _sgps_extract(ds) -> pd.DataFrame:
    """Σ ΔE · AvgDiffProtonFlux по channels с lower≥10 МэВ.
    Поддерживает обе схемы: 'time' coord (новые файлы) и
    'L2_SciData_TimeStamp' data_var (файлы 2020-2022)."""
    if "time" in ds.coords:
        t = pd.to_datetime(ds["time"].values)
    elif "L2_SciData_TimeStamp" in ds.data_vars:
        t = pd.to_datetime(ds["L2_SciData_TimeStamp"].values)
    else:
        return pd.DataFrame()

    flux = ds["AvgDiffProtonFlux"].values   # [time, sensor_units, channels]
    lo   = ds["DiffProtonLowerEnergy"].values
    hi   = ds["DiffProtonUpperEnergy"].values

    mask = lo[0] >= 10000.0   # каналы с lower ≥ 10 МэВ (keV)
    dE   = (hi[0] - lo[0]) * mask         # ширина бина в keV
    integral = np.einsum("tsc,c->ts", flux, dE)
    avg = np.nanmean(integral, axis=1)
    avg[avg < 0] = np.nan
    return pd.DataFrame({"flux": avg}, index=t)


# ── Кеш per-day ──────────────────────────────────────────────────────────────

_day_cache: dict[str, pd.DataFrame | None] = {}


def _cache_path(date: pd.Timestamp, sat: int) -> Path:
    return CACHE_DIR / f"proton_p10_g{sat:02d}_{date.strftime('%Y%m%d')}.parquet"


def _load_for_satellite(date: pd.Timestamp, sat: int, fmt: str) -> pd.DataFrame | None:
    if fmt == "eps":
        raw = RAW_DIR / f"eps_g{sat:02d}_{date.year:04d}{date.month:02d}.nc"
        path = _download(_eps_url(date.year, date.month, sat), raw,
                         f"GOES-{sat} EPS {date.year}-{date.month:02d}")
        if path is None:
            return None
        return _eps_extract(_open_nc(path))

    if fmt == "epead":
        raw = RAW_DIR / f"epead_g{sat:02d}_{date.year:04d}{date.month:02d}.nc"
        path = _download(_epead_url(date.year, date.month, sat), raw,
                         f"GOES-{sat} EPEAD {date.year}-{date.month:02d}")
        if path is None:
            return None
        return _epead_extract(_open_nc(path))

    if fmt == "sgps":
        raw = RAW_DIR / f"sgps_g{sat:02d}_{date.strftime('%Y%m%d')}.nc"
        url = _sgps_url(date, sat)
        path = _download(url, raw,
                         f"GOES-{sat} SGPS {date.strftime('%Y-%m-%d')}")
        if path is None:
            return None
        return _sgps_extract(_open_nc(path))

    return None


def load_day(date: pd.Timestamp) -> pd.DataFrame | None:
    epoch = select_epoch(date)
    if epoch is None:
        return None

    for sat in epoch["satellites"]:
        cache = _cache_path(date, sat)
        key = cache.name
        if key in _day_cache:
            return _day_cache[key]
        if cache.exists():
            df = pd.read_parquet(cache)
            _day_cache[key] = df
            return df

        df = _load_for_satellite(date, sat, epoch["format"])
        if df is None or df.empty:
            continue

        # Для месячного файла отрезать дневной срез
        if epoch["format"] in ("eps", "epead"):
            mask = df.index.normalize() == date.normalize()
            df = df[mask]
            if df.empty:
                continue

        df.to_parquet(cache)
        _day_cache[key] = df
        return df

    return None


def load_protons_window(onset: pd.Timestamp, peak: pd.Timestamp,
                        pad_min: int = 30) -> pd.DataFrame | None:
    onset = pd.Timestamp(onset)
    peak  = pd.Timestamp(peak)
    if peak <= onset:
        return None

    t0 = onset - pd.Timedelta(minutes=pad_min)
    t1 = peak  + pd.Timedelta(minutes=pad_min)

    days = pd.date_range(t0.date(), t1.date(), freq="D")
    frames = [load_day(pd.Timestamp(d)) for d in days]
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return None

    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    sub = df.loc[(df.index >= t0) & (df.index <= t1)].copy()
    return sub if not sub.empty else None


# ── CLI ──────────────────────────────────────────────────────────────────────

def smoke_test(n: int = 6) -> None:
    sys.path.insert(0, str(ROOT))
    from catalog import load_catalog

    df = load_catalog().sort_values("onset_dt")
    per_cycle = max(1, n // 3)
    pick = pd.concat([df[df["Цикл"] == c].head(per_cycle) for c in (23, 24, 25)]).head(n)

    for _, row in pick.iterrows():
        print(f"\n→ {row['event_id']}  cycle={row['Цикл']}  "
              f"{row['onset_dt']} → {row['peak_dt']}")
        sub = load_protons_window(row["onset_dt"], row["peak_dt"])
        if sub is None:
            print("  нет данных")
            continue
        f = sub["flux"].dropna()
        print(f"  точек: {len(sub)}, диапазон: {sub.index.min()} … {sub.index.max()}")
        if not f.empty:
            print(f"  flux: min={f.min():.3g}  max={f.max():.3g}  "
                  f"(каталог Jmax={row['Jmax']:.2f} pfu)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if cmd == "smoke":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
        smoke_test(n)
    else:
        print("Использование: python data.py [smoke [N]]")
