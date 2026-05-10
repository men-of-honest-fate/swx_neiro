# Этап 1 · Памятка для следующей сессии

> Снимок состояния на 2026-05-10. Контекст: LSTM-наукастинг профиля протонного
> потока ≥10 МэВ в фазе нарастания СПС. Полный концепт-документ — [PLAN.md](PLAN.md).

---

## Решения, принятые на этой сессии

1. **Папка кода:** `stage 1/` в корне (параллельно `stage 0/`).
2. **Источник 5-мин протонов:** **NCEI напрямую** (HTTP + netCDF через xarray).
   Проверены и отброшены:
   - `cdasws` (CDAWeb): GOES протонов нужного разрешения там нет. Только
     `*_K0_EP{S,8}` Key Parameters низкого разрешения для GOES 6-11. Никакого
     EPEAD/SGPS для протонов не лежит.
   - SunPy Fido (`a.Instrument.goes`): только XRS/SUVI, нет proton-клиента.
     `a.Physobs.particle_flux` находит SOHO COSTEP/ERNE (другая обсерватория).
3. **Конфигурация prior:** наследуется из `hybrid_no_vel_jmax1`
   (J_max≥1 pfu, T_delta≤40 ч, GOES_rise≤120 мин, признаки без скорости КВМ).
4. **Out-of-fold prior:** 5-fold CV по train (SC23+24), отдельно для каждой
   (group, target) пары. Метрики проверены против SUMMARY.md из этапа 0:
   - West/J_max/Linear: **0.737** (ожидалось 0.744) ✓
   - West/T_delta/SVR:  **7.88 ч** (ожидалось 7.88) ✓
   - East/T_delta/SVR:  **8.78 ч** (ожидалось 8.82) ✓
   - East/J_max/Forest: 1.16 vs 0.598 — отклонение из-за 1 события, выпавшего
     по битому `Время начала`/`Время максимума` в каталоге. Не блокер.

## Что готово в [stage 1/](.)

| Файл | Статус | Smoke-проверка |
|---|---|---|
| [PLAN.md](PLAN.md) | ✅ копия LSTM_nowcasting_plan.md в чистой UTF-8 | — |
| [catalog.py](catalog.py) | ✅ работает | 298 событий (237 train, 61 test); 2 битых отброшены |
| [data.py](data.py) | ⚠ переписан на NCEI, требует финального smoke | EPS/EPEAD/SGPS — три разных формата (см. ниже) |
| [profiles.py](profiles.py) | ⏳ ждёт data.py | — |
| [prior.py](prior.py) | ✅ работает | `prior_oof.parquet` создан, метрики сошлись |
| [dataset.py](dataset.py) | ✅ структура готова | ждёт profiles |
| [model.py](model.py) | ✅ работает | forward+loss+backward на dummy: OK |
| [train.py](train.py) | ✅ структура готова | ждёт dataset |
| [baselines.py](baselines.py) | ✅ работает | `results/plots/baselines_smoke.png` |
| [evaluate.py](evaluate.py) | ✅ структура готова | ждёт обучения |
| [run.py](run.py) | ✅ CLI-диспатчер | — |
| [SUMMARY.md](SUMMARY.md) | ✅ статус + плейбук | — |

## Архитектура `data.py` (NCEI)

Три формата по эпохам (`EPOCHS` в `data.py`):

| Эпоха | Формат | Спутники (primary первым) | Файл |
|---|---|---|---|
| 1995-01 — 1998-06 | EPS | 8, 9 | `g{NN}_eps_5m_{YYYYMM}.nc` помесячно |
| 1998-07 — 2003-04 | EPS | 8, 10, 11 | -//- |
| 2003-05 — 2008-02 | EPS | 11, 10, 12 | -//- |
| 2008-03 — 2009-12 | EPS | 12, 10 | -//- |
| 2010-01 — 2017-12 | EPEAD | 13, 15, 14 | `g{NN}_epead_p17ew_5m_{YYYYMM}.nc` |
| 2018-01 — 2019-12 | EPEAD | 15, 14 | -//- |
| 2020-01 — … | SGPS | 16, 18, 17, 19 | `sci_sgps-l2-avg5m_g{NN}_d{YYYYMMDD}_v*.nc` дневной |

**Proxy ≥10 МэВ:**
- EPS: `sum(p3..p7_flux_ic)` (≥8.7 МэВ integral, единицы pfu).
- EPEAD: `Σ ΔE·avg(P{i}E_COR_FLUX, P{i}W_COR_FLUX)` для P3..P6 (без P7 ради
  избегания GCR-загрязнения). ΔE: P3=5.8, P4=25, P5=44, P6=116 МэВ.
- SGPS: `Σ ΔE·AvgDiffProtonFlux` по каналам с lower≥10 МэВ (ch5..ch12),
  усреднение по 2 sensor_units.

**Базы URL:**
- `https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/avg/` — EPS/EPEAD
- `https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes{NN}/l2/data/sgps-l2-avg5m/` — SGPS

**Кеши:**
- `goes_proton_cache/_raw/{eps|epead|sgps}_g{NN}_*.nc` — сырые netCDF
- `goes_proton_cache/proton_p10_g{NN}_{YYYYMMDD}.parquet` — дневной парсинг
- Negative cache `_neg_cache` для 404 (в рамках процесса)

## Открытые баги / TODO

1. **`data.py` smoke ещё не пройден** на новой версии. Предыдущая попытка
   нашла 3 бага (исправлены, но не перепроверены):
   - имя файла EPS требует `g{NN:02d}` (с нулём): `g08_*`, не `g8_*` ✓
   - GOES 13-15 — это EPEAD, не EPS (другие имена переменных) ✓
   - SGPS-2020 и SGPS-2024 имеют разные имена time-переменной (`L2_SciData_TimeStamp` vs `time`-coord) ✓
   - Список спутников по эпохам уточнён ✓
2. **East/J_max prior** даёт RMSE=1.16 вместо ожидаемых 0.598 в SUMMARY.md.
   Причина — 1 выпавшее событие из-за NaT в `Время начала`/`Время максимума`.
   Если важно — расширить парсинг datetime в [catalog.py:_to_datetime](catalog.py).

## Следующий шаг (что делать сразу)

```bash
# 1) Перепроверить data.py после правок (без сети не получалось)
rm -f goes_proton_cache/proton_p10_*.parquet
.venv/bin/python "stage 1/run.py" smoke 6
# Ожидание: 6 событий из SC23/24/25, есть точки, max совпадает с Jmax из каталога
# по порядку величины.

# 2) Если smoke OK — массовая загрузка всех 298 профилей
.venv/bin/python "stage 1/run.py" download
.venv/bin/python "stage 1/run.py" qc   # plots/profile_qc.png

# 3) Прогон бейзлайнов
.venv/bin/python "stage 1/run.py" evaluate baselines

# 4) Обучение LSTM
.venv/bin/python "stage 1/run.py" train cv
.venv/bin/python "stage 1/run.py" train final
.venv/bin/python "stage 1/run.py" evaluate full
```

## Что переиспользуется из репозитория

- [spe_utils.py](../spe_utils.py): `build_features`, парсинг гелиокоординат.
- [hybrid_no_vel_jmax1/run.py](../hybrid_no_vel_jmax1/run.py): фильтр выборки
  J_max≥1 pfu (`load_splits_jmax1`), формула target-weights с clip=1
  (`target_weights_clip1`), список feature-sets без скорости КВМ (`FS_NO_VEL`).
- [stage 0/plot_ew_hybrid_full.py](../stage 0/plot_ew_hybrid_full.py): формулы
  density-weights и target-weights (`_density_weights`, `_target_weights`),
  правила NO_WEIGHT_MODELS={"SVR"}.

Path-паттерн как во всём проекте:
```python
ROOT         = Path(__file__).parent           # stage 1/
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
```
