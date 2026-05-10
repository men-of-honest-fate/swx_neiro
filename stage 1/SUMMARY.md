# Этап 1 · LSTM-наукастинг СПС — статус и инструкция

> Статус на 2026-05-10: каркас собран, данные GOES p ≥10 МэВ ещё не скачаны.

## Что готово

| Модуль | Файл | Smoke-проверка |
|---|---|---|
| Каталог + фильтрация (J_max≥1, T_delta≤40 ч, GOES_rise≤120 мин) + парсинг onset/peak | [catalog.py](catalog.py) | ✅ 298 событий (237 train, 61 test) |
| Загрузка 5-мин протонных рядов через `cdasws` + per-day parquet кеш | [data.py](data.py) | ⚠ требует сети + сверки dataset_id |
| Срез [onset, peak], log/Δlog, контроль качества | [profiles.py](profiles.py) | ⏳ ждёт data.py |
| Out-of-fold prior через 5-fold CV (Linear/SVR/Forest, density/target weights) | [prior.py](prior.py) | ✅ Test RMSE West/J_max=0.737, T_delta=7.88 ч; East/T_delta=8.78 ч |
| PyTorch Dataset со скользящим окном + group K-fold | [dataset.py](dataset.py) | ⏳ ждёт profiles |
| LSTM (prior→h₀, encoder, one-shot decoder) + loss | [model.py](model.py) | ✅ forward+loss+backward на dummy |
| Train loop, k-fold CV, early stopping | [train.py](train.py) | ⏳ ждёт dataset |
| Бейзлайны: persistence, power-law fit, prior-only (гамма) | [baselines.py](baselines.py) | ✅ smoke-график построен |
| Метрики, ΔJ_max(t), case studies | [evaluate.py](evaluate.py) | ⏳ ждёт обучение |
| CLI-диспатчер | [run.py](run.py) | ✅ |

## Открытые блокеры

1. **Реестр CDAWeb-датасетов (`data.py:DATASET_REGISTRY`)** не выверен. На первом запуске нужна сеть:
   ```bash
   python "stage 1/run.py" inspect            # перебор кандидатов по ключевым словам
   python "stage 1/run.py" smoke 5            # скачать 5 событий и распечатать их структуру
   ```
   После этого подправить `dataset_id` / `var_name` в `DATASET_REGISTRY` для всех 4 эпох:
   - SC23 (1996-04-30): `GOES8_EPS-1MIN` / `P1` (≥10 МэВ)
   - SC23 (2003-09): `GOES11_EPS-1MIN` / `P1`
   - SC24: `G13_EPEAD-PCHAN-PFLUX-1MIN` / `P1_FLUX`
   - SC25: `DN_MAGSPD-3S_G16_SGPS-L2-AVG5M` / `AvgIntProtonFlux`

2. **Полная загрузка профилей.** После выверки реестра:
   ```bash
   python "stage 1/run.py" download           # ~1-2 ч скачивания + parquet'ов
   python "stage 1/run.py" qc                 # plots/profile_qc.png
   ```

## Полный пайплайн запуска

```bash
# 1) убедиться, что cdasws установлен
pip install -r req.txt

# 2) сверить dataset_id (онлайн)
python "stage 1/run.py" inspect
python "stage 1/run.py" smoke 5

# 3) скачать профили + контроль качества
python "stage 1/run.py" download

# 4) out-of-fold prior (быстро, без сети)
python "stage 1/run.py" prior

# 5) бейзлайны
python "stage 1/run.py" evaluate baselines

# 6) обучение LSTM
python "stage 1/run.py" train cv             # 5 фолдов
python "stage 1/run.py" train final          # финальная модель на SC23+24

# 7) оценка LSTM + сравнение с бейзлайнами
python "stage 1/run.py" evaluate full
```

## Конфигурация (изменить в `train.py:DEFAULT_CONFIG`)

```python
hidden_size  = 64
num_layers   = 2
dropout      = 0.0
lam_peak     = 1.0          # вес L_peak в общем лоссе
lr           = 1e-3
batch_size   = 32
max_epochs   = 200
patience     = 25
k_min        = 6            # минимум 30 мин наблюдений до прогноза
n_max        = 250          # максимум 21 ч на профиль
```

Сетка для подбора (PLAN.md §5): hidden_size ∈ {32,64,128}, lr ∈ {1e-3, 3e-4, 1e-4},
dropout ∈ {0, 0.1, 0.2}, λ ∈ {0.5, 1.0, 2.0}, k_min ∈ {6, 12}.

## Метрики prior (после `python "stage 1/run.py" prior`)

Сверка с `hybrid_no_vel_jmax1/SUMMARY.md` (West Test / J_max ≥ 1 pfu):

| Группа · цель · модель | Stage 0 SUMMARY | Stage 1 prior_oof |
|---|---:|---:|
| West/J_max/Linear (Флюэс) | RMSLE = 0.744 | **0.737** |
| West/T_delta/SVR (Флюэс)  | RMSE  = 7.88 ч | **7.88 ч** |
| East/J_max/Forest (Базовая) | RMSLE = 0.598 | 1.16 (см. ниже) |
| East/T_delta/SVR (Флюэс)  | RMSE  = 8.82 ч | **8.78 ч** |

East/J_max отклоняется на ~0.5 RMSLE — выпало 1 событие из-за NaT в `Время начала`/
`Время максимума`, влияние при `n_test=10` чувствительное. Для prior это не блокер
(LSTM использует prior как «намёк», а не ground truth), но если хочется
точного совпадения — расширить парсинг datetime в `catalog._to_datetime`.

## Что дальше (после успешной базовой модели)

Из PLAN.md §10:
1. Скорость КВМ как доп. канал энкодера.
2. Составные события — обучение на чистых, тест на всех.
3. Рентген (SXR) как второй канал — этап 2.
4. Attention layer — для интерпретируемости.
5. Uncertainty (MC Dropout / quantile regression).
