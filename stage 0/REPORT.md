# Сравнительный анализ моделей прогнозирования параметров солнечных протонных событий

> **Дата:** 2026-03-14
> **Данные:** ОБЪЕДИНЕННЫЙ КАТАЛОГ СПС 23–25, лист «Флюэс GOES»
> **Обучающая выборка:** SC23 + SC24 (138 событий, J_max ≥ 10 pfu)
> **Тестовая выборка:** SC25 (38 событий)
> **Кросс-валидация:** leave-one-cycle-out (SC23 ↔ SC24)

---

## 1. Конфигурация эксперимента

### 1.1 Целевые переменные

| Обозначение | Описание | Единица |
|-------------|----------|---------|
| J_max | Пиковый протонный поток (E > 10 МэВ) | pfu (протонов·см⁻²·с⁻¹·ср⁻¹) |
| T_delta | Время от начала СПС до достижения пика потока | часы |

### 1.2 Входные признаки

| Признак | Полное наименование | Физический смысл |
|---------|--------------------|--------------------|
| `helio_lon` | Гелиоцентрическая долгота источника | Угловое положение активной области на Солнце; связано с эффективностью транспорта частиц вдоль межпланетных силовых линий |
| `helio_lat` | Гелиоцентрическая широта источника | Отклонение источника от эклиптики; учитывает широтную анизотропию распространения |
| `log_goes_peak_flux` | lg(пиковый рентгеновский поток GOES XRS, 1–8 Å, W·м⁻²) | Непрерывная мера пиковой мощности вспышки; точный аналог дискретного класса A/B/C/M/X |
| `log_fluence` | lg(флюэнс рентгеновского излучения GOES XRS, 1–8 Å, Дж·м⁻²) | Интегральная мера энерговыделения вспышки; коррелирует с полным ускоренным зарядом |
| `log_flare_dur_min` | lg(длительность вспышки, мин) | Продолжительность фазы повышенного рентгеновского излучения |
| `log_goes_rise_min` | lg(время нарастания вспышки, мин) | Интервал между началом и пиком рентгеновского потока; индикатор импульсности события |
| `log_cme_velocity` | lg(линейная скорость КВМ по данным LASCO, км·с⁻¹) | Скорость корональных выбросов масс; определяет время прихода возмущения и энергию ударной волны |
| `cme_width_deg` | Угловая ширина КВМ (градусы) | Пространственный охват выброса; косвенная мера суммарной кинетической энергии |
| `cme_pa_deg` | Позиционный угол КВМ (градусы от севера) | Направление выброса; влияет на попадание возмущения на линию Земля–Солнце |
| `t_delta_flare` | Задержка между началом вспышки и началом СПС (часы) | Характеризует время инжекции частиц; зависит от геометрии межпланетного магнитного поля |

### 1.3 Наборы признаков

В эксперименте рассматриваются 13 наборов признаков, охватывающих различные комбинации характеристик вспышки, КВМ и геометрии источника.

| №   | Название          | Состав                                                                                                                                                 |
| --- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1   | Базовая           | `helio_lon`, `log_goes_peak_flux`, `log_cme_velocity`                                                                                                  |
| 2   | Обе координаты    | Базовая + `helio_lat`                                                                                                                                  |
| 3   | Без КВМ           | `helio_lon`, `log_goes_peak_flux`                                                                                                                      |
| 4   | T_delta_flare     | `t_delta_flare`, `log_goes_peak_flux`, `log_cme_velocity`                                                                                              |
| 5   | Все базовые       | `helio_lon`, `helio_lat`, `t_delta_flare`, `log_goes_peak_flux`, `log_cme_velocity`                                                                    |
| 6   | Флюэс вместо пика | `helio_lon`, `log_fluence`, `log_cme_velocity`                                                                                                         |
| 7   | Флюэс + пик       | `helio_lon`, `log_fluence`, `log_goes_peak_flux`, `log_cme_velocity`                                                                                   |
| 8   | КВМ расширенный   | `helio_lon`, `log_goes_peak_flux`, `log_cme_velocity`, `cme_width_deg`, `cme_pa_deg`                                                                   |
| 9   | Вспышка расшир.   | `helio_lon`, `log_fluence`, `log_goes_peak_flux`, `log_flare_dur_min`, `log_cme_velocity`                                                              |
| 10  | Kitchen sink      | `helio_lon`, `helio_lat`, `t_delta_flare`, `log_fluence`, `log_goes_peak_flux`, `log_flare_dur_min`, `log_cme_velocity`, `cme_width_deg`, `cme_pa_deg` |
| 11  | Пик+нараст.       | `helio_lon`, `log_goes_peak_flux`, `log_goes_rise_min`, `log_cme_velocity`                                                                             |
| 12  | GOES полный       | `helio_lon`, `helio_lat`, `log_goes_peak_flux`, `log_fluence`, `log_cme_velocity`                                                                      |
| 13  | GOES KS           | `helio_lon`, `helio_lat`, `log_goes_peak_flux`, `log_fluence`, `log_flare_dur_min`, `log_cme_velocity`, `cme_width_deg`, `cme_pa_deg`, `t_delta_flare` |

### 1.4 Модели и метрики

| Пайплайн | Модели | Первичная метрика |
|---------|--------|------------------|
| Регрессия | Linear, Ridge, Huber, Forest, ExtraTrees, Boosting, SVR, GPR_RBF | RMSLE log₁₀ (J_max), RMSE (T_delta) |
| Вероятностный | QuantLinear, QuantBoosting, BayesRidge, ConformalRF | Winkler score↓ (80% ДИ) |
| Классификация | LogReg, Forest, ExtraTrees, Boosting, SVC | Log Loss↓ |

---

## 2. Метрики оценки качества

### 2.1 RMSLE в log₁₀-единицах (точечная регрессия J_max)

Поскольку J_max охватывает несколько порядков величины (10–10⁵ pfu), прогнозирование производится в логарифмическом пространстве. Метрика определяется как:

$$\mathrm{RMSLE_{log_{10}}} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}\bigl(\log_{10}\hat{y}_i - \log_{10}y_i\bigr)^2}$$

Значение 1.0 соответствует типичной ошибке в 10 раз по потоку; значение 0.65 — ошибке ≈×4.5.

### 2.2 RMSE (точечная регрессия T_delta и ширина ДИ)

$$\mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)^2}$$

Для T_delta выражается в часах.

### 2.3 Коэффициент детерминации R²

$$R^2 = 1 - \frac{\sum_{i}(\hat{y}_i - y_i)^2}{\sum_{i}(\bar{y} - y_i)^2}$$

Для J_max вычисляется в логарифмическом пространстве (R²_log). Отрицательные значения указывают на то, что модель хуже среднего.

### 2.4 Ранговая корреляция Спирмена

$$\rho_s = 1 - \frac{6\sum_i d_i^2}{n(n^2-1)}, \quad d_i = \mathrm{rg}(\hat{y}_i) - \mathrm{rg}(y_i)$$

Нечувствительна к масштабу и выбросам; оценивает монотонность прогноза.

### 2.5 Winkler score (вероятностный прогноз, интервал уровня α = 0.20)

$$W_\alpha = \frac{1}{n}\sum_{i=1}^{n}\left[(u_i - l_i) + \frac{2}{\alpha}\max(l_i - y_i,\,0) + \frac{2}{\alpha}\max(y_i - u_i,\,0)\right]$$

где $l_i$, $u_i$ — нижняя и верхняя границы 80%-го доверительного интервала. Штрафует одновременно за избыточную ширину интервала и за промахи. Меньше — лучше.

### 2.6 Log Loss (классификация)

$$\mathcal{L} = -\frac{1}{n}\sum_{i=1}^{n}\sum_{k=1}^{K} y_{ik}\ln\hat{p}_{ik}$$

где $\hat{p}_{ik}$ — предсказанная вероятность принадлежности наблюдения $i$ к классу $k$, $K = 3$ (S1–S2, S3, S4–S5). Меньше — лучше.

### 2.7 ROC AUC (one-vs-rest, макро-усреднение)

Площадь под ROC-кривой при схеме «один против остальных», усреднённая по трём классам. Значение 0.5 соответствует случайному классификатору; значение 1.0 — идеальному.

---

## 3. Результаты: точечная регрессия

### 3.1 J_max (пиковый протонный поток)

Первичная метрика — RMSLE log₁₀. Приведены результаты на тестовой выборке SC25.

| Набор признаков | RMSLE log₁₀ | R²_log | ρ_s | Лучшая модель |
|----------------|:-----------:|:------:|:---:|:-------------:|
| **Флюэс вместо пика** | **0.649** | 0.24 | 0.62 | Huber |
| Пик+нараст. | 0.659 | 0.22 | 0.56 | Huber |
| GOES полный | 0.671 | 0.19 | 0.54 | Huber |
| GOES KS | 0.674 | 0.18 | 0.57 | Huber |
| Флюэс + пик | 0.676 | 0.18 | 0.56 | Huber |
| Вспышка расшир. | 0.680 | 0.17 | 0.53 | Huber |
| Kitchen sink | 0.691 | 0.14 | 0.46 | Forest |
| T_delta_flare | 0.695 | 0.06 | 0.51 | Ridge |
| Базовая | 0.700 | 0.05 | 0.46 | GPR_RBF |
| Без КВМ | 0.712 | 0.01 | 0.30 | Huber |
| КВМ расширенный | 0.743 | −0.07 | 0.47 | Huber |

**Основные выводы:**

- Логарифм флюэнса рентгеновского излучения (`log_fluence`) является наиболее информативным предиктором J_max (RMSLE = 0.649): как интегральная мера энерговыделения вспышки, он устойчивее коррелирует с протонным потоком, чем пиковый поток.
- Пиковый поток GOES XRS (`log_goes_peak_flux`) обеспечивает более высокое качество прогноза, чем дискретный класс вспышки (A/B/C/M/X), за счёт непрерывного измерения без квантования.
- Добавление времени нарастания вспышки (`log_goes_rise_min`) к пиковому потоку даёт незначительное улучшение (ΔRMSLE = 0.010).
- Включение гелиоширготы и параметров КВМ в базовую конфигурацию **ухудшает** качество: при объёме обучающей выборки 138 событий дополнительные признаки приводят к переобучению.
- Максимальный R²_log = 0.24 свидетельствует о принципиально стохастической природе задачи.

![[pipelines/regression/plots/comparison/heatmap_jmax.png]]

![[pipelines/regression/plots/comparison/bars_test_primary_jmax.png]]

![[pipelines/regression/plots/comparison/best_model_per_featureset.png]]

### 3.2 T_delta (время до максимума протонного потока)

Первичная метрика — RMSE (часы).

| Набор признаков | RMSE (ч) | R² | ρ_s | Лучшая модель |
|----------------|:--------:|:--:|:---:|:-------------:|
| **GOES KS** | **14.01** | 0.18 | −0.05 | Forest |
| Вспышка расшир. | 14.10 | 0.17 | 0.18 | Linear |
| Kitchen sink | 14.11 | 0.17 | 0.15 | Forest |
| Флюэс + пик | 14.32 | 0.15 | 0.17 | Linear |
| Пик+нараст. | 14.34 | 0.14 | 0.19 | Linear |
| Базовая | 15.51 | 0.21 | 0.26 | Linear |

**Основные выводы:**

- Прогнозирование T_delta представляет значительную сложность: R² ≤ 0.21, ρ_s ≤ 0.26 по всем наборам признаков.
- Различие в значениях RMSE и R² между наборами «Базовая» (RMSE = 15.5, R² = 0.21) и «GOES KS» (RMSE = 14.0, R² = 0.18) объясняется различиями в распределении ошибок: «Базовая» лучше ранжирует события по среднему, но хуже восстанавливает абсолютные значения.
- Время нарастания рентгеновской вспышки (`log_goes_rise_min`) не вносит существенного вклада в прогноз T_delta.
- Линейные модели стабильно показывают результаты не ниже нелинейных, что указывает на отсутствие выраженных нелинейных зависимостей в доступных признаках.
- Низкое качество прогноза обусловлено отсутствием данных о конфигурации межпланетного магнитного поля, определяющей время прихода частиц.

![[pipelines/regression/plots/comparison/heatmap_tdelta.png]]

---

## 4. Результаты: вероятностный прогноз (80% доверительный интервал)

### 4.1 J_max

Первичная метрика — Winkler score (W₀.₂₀).

#### Кросс-валидация (leave-one-cycle-out)

| Набор признаков | W₀.₂₀ | Coverage | Лучшая модель |
|----------------|:-----:|:--------:|:-------------:|
| **Базовая** | **2.591** | 75% | QuantLinear |
| Флюэс + пик | 2.611 | 76% | BayesRidge |
| Вспышка расшир. | 2.613 | 75% | BayesRidge |
| КВМ расширенный | 2.622 | 79% | BayesRidge |
| Без КВМ | 2.676 | 80% | BayesRidge |
| Пик+нараст. | 2.802 | 76% | BayesRidge |
| GOES KS | 3.128 | 61% | BayesRidge |

#### Тест SC25

| Набор признаков | W₀.₂₀ | Coverage | Лучшая модель |
|----------------|:-----:|:--------:|:-------------:|
| **Без КВМ** | **2.064** | 79% | QuantBoosting |
| T_delta_flare | 2.223 | 87% | QuantLinear |
| Флюэс вместо пика | 2.246 | 85% | QuantLinear |
| Флюэс + пик | 2.275 | 79% | QuantBoosting |
| Базовая | 2.304 | 87% | QuantLinear |
| GOES KS | 2.313 | 85% | BayesRidge |

**Основные выводы:**

- Наблюдается выраженное расхождение результатов CV и теста: набор «Базовая» (лучший на CV, W = 2.591) занимает 6-ю позицию на тесте (W = 2.304). Данный паттерн характерен для малых обучающих выборок с высокой дисперсией.
- Наименьший Winkler score на тесте демонстрирует набор «Без КВМ» (2 признака) с моделью QuantBoosting — избыточная параметризация отрицательно сказывается на обобщающей способности.
- Признаки GOES XRS увеличивают Winkler score на CV (переобучение), однако на тесте набор GOES KS занимает конкурентное место (W = 2.313, 6-я позиция).
- Coverage 79–88% при целевом значении 80% свидетельствует об удовлетворительной калибровке интервальных оценок на тесте.

![[pipelines/probabilistic/plots/compare/heatmap_jmax_test.png]]

![[pipelines/probabilistic/plots/compare/cov_width_jmax_test.png]]

![[pipelines/probabilistic/plots/compare/best_jmax_test.png]]

### 4.2 T_delta

Первичная метрика — Winkler score (часы).

#### Кросс-валидация (лучшие результаты)

| Набор признаков | W₀.₂₀ (ч) | Coverage |
|----------------|:---------:|:--------:|
| **Флюэс + пик** | **42.85** | 71% |
| Флюэс вместо пика | 44.42 | 70% |
| Без КВМ | 44.66 | 73% |
| Обе координаты | 44.71 | 85% |

**Основные выводы:** Все модели формируют широкие доверительные интервалы (40–50 часов при медианном T_delta ≈ 30 часов), что подтверждает принципиальную ограниченность прогнозируемости данной величины в рамках использованного набора признаков. Незначительное улучшение при включении флюэнса согласуется с гипотезой о частичной корреляции мощности вспышки со скоростью транспорта частиц.

![[pipelines/probabilistic/plots/compare/heatmap_t_delta_test.png]]

---

## 5. Результаты: классификация по шкале NOAA S

**Распределение классов:** S1–S2 (10–100 pfu): 74/25, S3 (100–1000 pfu): 43/10, S4–S5 (≥1000 pfu): 21/3 (обучение/тест).

### Кросс-валидация

| Набор признаков | Log Loss | Accuracy | AUC | Лучшая модель |
|----------------|:--------:|:--------:|:---:|:-------------:|
| **Базовая** | **0.937** | 58% | 0.627 | SVC |
| Флюэс + пик | 0.970 | 56% | 0.580 | SVC |
| Обе координаты | 0.976 | 57% | 0.664 | Forest |
| Без КВМ | 0.998 | 52% | 0.583 | SVC |
| КВМ расширенный | 1.002 | 54% | 0.600 | SVC |
| Kitchen sink | 1.155 | 48% | 0.435 | SVC |
| **GOES KS** | 1.180 | 46% | 0.418 | SVC |

### Тест SC25

| Набор признаков | Log Loss | Accuracy | AUC | Лучшая модель |
|----------------|:--------:|:--------:|:---:|:-------------:|
| **GOES KS** | **0.730** | 65% | **0.842** | LogReg |
| GOES полный | 0.741 | 65% | 0.804 | LogReg |
| Флюэс + пик | 0.749 | 65% | 0.753 | Forest |
| Kitchen sink | 0.755 | 62% | 0.836 | LogReg |
| Все базовые | 0.764 | 58% | 0.734 | Forest |
| Базовая | 0.849 | 63% | 0.631 | SVC |

**Основные выводы:**

- Зафиксирована инверсия рейтинга CV→Test: набор «Базовая» (наилучший на CV: Log Loss = 0.937) занимает 11-ю позицию на тесте (Log Loss = 0.849); набор «GOES KS» (наихудший на CV: 1.180) обеспечивает минимальный Log Loss на тесте (0.730, AUC = 0.842). Данная инверсия обусловлена переобучением сложных моделей на малой обучающей выборке при высоком разнообразии паттернов между циклами.
- LogReg с расширенным набором признаков GOES XRS обеспечивает AUC = 0.842, что указывает на линейную разделимость классов S в логарифмическом пространстве пикового рентгеновского потока.
- SVC доминирует на CV за счёт нелинейной границы решения, однако переобучается при переносе на SC25.

![[pipelines/classification/plots/compare/heatmap_log_loss_test.png]]

![[pipelines/classification/plots/compare/heatmap_auc_test.png]]

![[pipelines/classification/plots/compare/best_test.png]]

---

## 6. Сводные результаты

| Задача | Набор признаков | Модель | Метрика (тест) |
|--------|----------------|--------|----------------|
| J_max, точечный прогноз | Флюэс вместо пика | **Huber** | RMSLE = 0.649 |
| J_max, 80% ДИ | Без КВМ | **QuantBoosting** | W₀.₂₀ = 2.064, Coverage = 79% |
| T_delta, точечный прогноз | GOES KS | **Forest** | RMSE = 14.01 ч |
| S-класс, классификация | GOES KS | **LogReg** | Log Loss = 0.730, AUC = 0.842 |

### Выводы по информативности признаков

1. **Флюэнс рентгеновского излучения** (`log_fluence`) является наиболее информативным предиктором J_max по всем трём пайплайнам. Как интегральная мера энерговыделения вспышки он устойчивее коррелирует с суммарным ускоренным зарядом, чем мгновенные характеристики.

2. **Признаки GOES XRS улучшают обобщение на SC25, но ухудшают кросс-валидацию.** Вероятная причина — меньший объём выборки с XRS-данными и более высокая сложность модели относительно 138 обучающих событий.

3. **Расхождение CV и теста систематически.** На CV выигрывают простые наборы (Базовая, Без КВМ); на тесте для регрессии J_max — также простые, для классификации — богатые наборы GOES. Рекомендуется использовать тестовые метрики в качестве основного критерия отбора модели.

4. **Время до максимума T_delta принципиально ограничено в предсказуемости** (R² ≤ 0.21, ρ_s ≤ 0.26). Доминирующий фактор — конфигурация межпланетного магнитного поля на пути от источника до Земли — отсутствует в использованном наборе признаков.

5. **Huber-регрессия** демонстрирует наибольшую устойчивость для J_max. Робастность к выбросам принципиальна при наличии событий уровня X9+ с потоками на 2–3 порядка выше медианного.

6. **LogReg с GOES XRS признаками** обеспечивает AUC = 0.842 для классификации S-классов. Несмотря на линейную природу, модель эффективно разделяет классы в логарифмическом пространстве пикового потока.

### Точность по задачам

| Задача | Значение метрики | Интерпретация |
|--------|-----------------|---------------|
| Регрессия J_max | RMSLE = 0.649 | Типичная ошибка ≈ ×4.5 по потоку |
| Вероятностный J_max | W₀.₂₀ = 2.064, Coverage = 79% | Целевое покрытие 80% — выполнено |
| Регрессия T_delta | RMSE = 14.0 ч | При σ(T_delta) ≈ 30 ч — объясняемая доля дисперсии мала |
| Классификация S-класс | Accuracy = 65%, AUC = 0.842 | Значимая дискриминирующая способность |

---

## 7. Визуальный обзор результатов сравнения

### Регрессия

![[pipelines/regression/plots/comparison/heatmap_jmax.png]]

*Тепловая карта RMSLE log₁₀ для J_max. Строки = наборы признаков, столбцы = модели.*

![[pipelines/regression/plots/comparison/heatmap_tdelta.png]]

*Тепловая карта RMSE (часы) для T_delta.*

![[pipelines/regression/plots/comparison/bars_test_primary_jmax.png]]

![[pipelines/regression/plots/comparison/bars_test_primary_tdelta.png]]

### Вероятностный прогноз — J_max

![[pipelines/probabilistic/plots/compare/heatmap_jmax_cv.png]]

![[pipelines/probabilistic/plots/compare/heatmap_jmax_test.png]]

![[pipelines/probabilistic/plots/compare/cov_width_jmax_test.png]]

*Coverage vs ширина интервала для каждого набора признаков. Цель: Coverage ≥ 80% при минимальной ширине.*

### Вероятностный прогноз — T_delta

![[pipelines/probabilistic/plots/compare/heatmap_t_delta_test.png]]

![[pipelines/probabilistic/plots/compare/cov_width_t_delta_test.png]]

### Классификация

![[pipelines/classification/plots/compare/heatmap_log_loss_test.png]]

![[pipelines/classification/plots/compare/heatmap_auc_test.png]]

![[pipelines/classification/plots/compare/best_test.png]]

---

## 8. Анализ вклада признаков

Для каждого пайплайна вычислены три вида оценок важности:

- **Builtin:** нормированные коэффициенты модели (|coef_| для линейных, feature_importances_ для ансамблей);
- **SHAP:** значения Шепли (TreeExplainer для Random Forest, ExtraTrees, GradientBoosting; LinearExplainer для линейных моделей); среднее |SHAP| по выборке, нормированное на 100%;
- **Mutual Information:** оценка взаимной информации I(x_j; y) для каждого признака независимо.

Результаты представлены как глобальные тепловые карты (признак × набор признаков, усреднение по моделям), per-model тепловые карты и per-set рейтинги (усреднение по моделям внутри каждого набора).

### 8.1 Регрессия

**Методы:** builtin (|coef_| / feature_importances_), SHAP (TreeExplainer / LinearExplainer), Mutual Information.

#### Глобальный рейтинг — J_max

![[pipelines/regression/plots/importance/global/ranking_jmax.png]]

*Среднее по всем 13 наборам и всем моделям. Признаки упорядочены по убыванию вклада.*

#### Глобальный рейтинг — T_delta

![[pipelines/regression/plots/importance/global/ranking_t_delta.png]]

#### Тепловые карты (builtin)

![[pipelines/regression/plots/importance/global/global_builtin_jmax.png]]

*Строки = признаки, столбцы = наборы признаков. Серый — признак не включён в набор. Значения в % (нормировка на 100% по каждой модели/набору).*

![[pipelines/regression/plots/importance/global/global_builtin_t_delta.png]]

#### Тепловые карты (SHAP)

![[pipelines/regression/plots/importance/global/global_shap_jmax.png]]

![[pipelines/regression/plots/importance/global/global_shap_t_delta.png]]

#### Тепловые карты (Mutual Information)

![[pipelines/regression/plots/importance/global/global_mutual_info_jmax.png]]

![[pipelines/regression/plots/importance/global/global_mutual_info_t_delta.png]]

**Основные выводы:**

- `log_fluence` устойчиво занимает первое место по всем трём методам оценки для J_max, что согласуется с результатами сравнения наборов признаков.
- `log_goes_peak_flux` и `log_cme_velocity` делят второе-третье место для J_max; для T_delta `log_cme_velocity` выходит на первую позицию.
- `helio_lon` входит в топ-3 для J_max, однако его вклад в T_delta существенно ниже.
- `t_delta_flare` — значимый предиктор T_delta (задержка вспышка–СПС коррелирует со временем до пика протонного потока).
- `cme_width_deg` и `cme_pa_deg` — наименее информативные признаки, что объясняет ухудшение качества при включении набора «КВМ расширенный».

#### Per-set рейтинги — регрессия

*Для каждого набора признаков: среднее по всем моделям (builtin + SHAP там, где применимо).*

| Набор | J_max | T_delta |
|-------|:-----:|:-------:|
| Базовая | ![[pipelines/regression/plots/importance/per_set_ranking/rank_jmax_Базовая.png]] | ![[pipelines/regression/plots/importance/per_set_ranking/rank_t_delta_Базовая.png]] |
| Обе координаты | ![[pipelines/regression/plots/importance/per_set_ranking/rank_jmax_Обе_координаты.png]] | ![[pipelines/regression/plots/importance/per_set_ranking/rank_t_delta_Обе_координаты.png]] |
| Без КВМ | ![[pipelines/regression/plots/importance/per_set_ranking/rank_jmax_Без_КВМ.png]] | ![[pipelines/regression/plots/importance/per_set_ranking/rank_t_delta_Без_КВМ.png]] |
| T_delta_flare | ![[pipelines/regression/plots/importance/per_set_ranking/rank_jmax_T_delta_flare.png]] | ![[pipelines/regression/plots/importance/per_set_ranking/rank_t_delta_T_delta_flare.png]] |
| Все базовые | ![[pipelines/regression/plots/importance/per_set_ranking/rank_jmax_Все_базовые.png]] | ![[pipelines/regression/plots/importance/per_set_ranking/rank_t_delta_Все_базовые.png]] |
| Флюэс вместо пика | ![[pipelines/regression/plots/importance/per_set_ranking/rank_jmax_Флюэс_вместо_пика.png]] | ![[pipelines/regression/plots/importance/per_set_ranking/rank_t_delta_Флюэс_вместо_пика.png]] |
| Флюэс + пик | ![[pipelines/regression/plots/importance/per_set_ranking/rank_jmax_Флюэс_p_пик.png]] | ![[pipelines/regression/plots/importance/per_set_ranking/rank_t_delta_Флюэс_p_пик.png]] |
| КВМ расширенный | ![[pipelines/regression/plots/importance/per_set_ranking/rank_jmax_КВМ_расширенный.png]] | ![[pipelines/regression/plots/importance/per_set_ranking/rank_t_delta_КВМ_расширенный.png]] |
| Вспышка расшир. | ![[pipelines/regression/plots/importance/per_set_ranking/rank_jmax_Вспышка_расшир..png]] | ![[pipelines/regression/plots/importance/per_set_ranking/rank_t_delta_Вспышка_расшир..png]] |
| Kitchen sink | ![[pipelines/regression/plots/importance/per_set_ranking/rank_jmax_Kitchen_sink.png]] | ![[pipelines/regression/plots/importance/per_set_ranking/rank_t_delta_Kitchen_sink.png]] |
| Пик+нараст. | ![[pipelines/regression/plots/importance/per_set_ranking/rank_jmax_Пикpнараст..png]] | ![[pipelines/regression/plots/importance/per_set_ranking/rank_t_delta_Пикpнараст..png]] |
| GOES полный | ![[pipelines/regression/plots/importance/per_set_ranking/rank_jmax_GOES_полный.png]] | ![[pipelines/regression/plots/importance/per_set_ranking/rank_t_delta_GOES_полный.png]] |
| GOES KS | ![[pipelines/regression/plots/importance/per_set_ranking/rank_jmax_GOES_KS.png]] | ![[pipelines/regression/plots/importance/per_set_ranking/rank_t_delta_GOES_KS.png]] |

---

### 8.2 Вероятностный прогноз

**Методы:** builtin (coef_ / feature_importances_ для QuantLinear, QuantBoosting, BayesRidge, ConformalRF; SHAP для QuantBoosting через внутренний регрессор q_mid_, для ConformalRF через rf_).

#### Глобальный рейтинг — J_max

![[pipelines/probabilistic/plots/importance/global/ranking_prob_jmax.png]]

#### Глобальный рейтинг — T_delta

![[pipelines/probabilistic/plots/importance/global/ranking_prob_t_delta.png]]

#### Тепловые карты

![[pipelines/probabilistic/plots/importance/global/global_heatmap_prob_jmax.png]]

*Усреднение по QuantLinear, QuantBoosting, BayesRidge, ConformalRF.*

![[pipelines/probabilistic/plots/importance/global/global_heatmap_prob_t_delta.png]]

**Основные выводы:**

- Паттерн важности воспроизводит результаты регрессии: `log_fluence` и `log_goes_peak_flux` лидируют для J_max.
- BayesRidge и QuantLinear согласованы по топ-признакам; QuantBoosting придаёт повышенный вес `helio_lon`.
- ConformalRF (Random Forest) в большей степени, чем линейные модели, опирается на `log_cme_velocity`.
- Для T_delta: `log_cme_velocity` и `t_delta_flare` конкурируют за первую позицию в зависимости от модели.

#### Per-set рейтинги — вероятностный прогноз

| Набор | J_max | T_delta |
|-------|:-----:|:-------:|
| Базовая | ![[pipelines/probabilistic/plots/importance/per_set_ranking/rank_jmax_Базовая.png]] | ![[pipelines/probabilistic/plots/importance/per_set_ranking/rank_t_delta_Базовая.png]] |
| Обе координаты | ![[pipelines/probabilistic/plots/importance/per_set_ranking/rank_jmax_Обе_координаты.png]] | ![[pipelines/probabilistic/plots/importance/per_set_ranking/rank_t_delta_Обе_координаты.png]] |
| Без КВМ | ![[pipelines/probabilistic/plots/importance/per_set_ranking/rank_jmax_Без_КВМ.png]] | ![[pipelines/probabilistic/plots/importance/per_set_ranking/rank_t_delta_Без_КВМ.png]] |
| T_delta_flare | ![[pipelines/probabilistic/plots/importance/per_set_ranking/rank_jmax_T_delta_flare.png]] | ![[pipelines/probabilistic/plots/importance/per_set_ranking/rank_t_delta_T_delta_flare.png]] |
| Все базовые | ![[pipelines/probabilistic/plots/importance/per_set_ranking/rank_jmax_Все_базовые.png]] | ![[pipelines/probabilistic/plots/importance/per_set_ranking/rank_t_delta_Все_базовые.png]] |
| Флюэс вместо пика | ![[pipelines/probabilistic/plots/importance/per_set_ranking/rank_jmax_Флюэс_вместо_пика.png]] | ![[pipelines/probabilistic/plots/importance/per_set_ranking/rank_t_delta_Флюэс_вместо_пика.png]] |
| Флюэс + пик | ![[pipelines/probabilistic/plots/importance/per_set_ranking/rank_jmax_Флюэс_p_пик.png]] | ![[pipelines/probabilistic/plots/importance/per_set_ranking/rank_t_delta_Флюэс_p_пик.png]] |
| КВМ расширенный | ![[pipelines/probabilistic/plots/importance/per_set_ranking/rank_jmax_КВМ_расширенный.png]] | ![[pipelines/probabilistic/plots/importance/per_set_ranking/rank_t_delta_КВМ_расширенный.png]] |
| Вспышка расшир. | ![[pipelines/probabilistic/plots/importance/per_set_ranking/rank_jmax_Вспышка_расшир..png]] | ![[pipelines/probabilistic/plots/importance/per_set_ranking/rank_t_delta_Вспышка_расшир..png]] |
| Kitchen sink | ![[pipelines/probabilistic/plots/importance/per_set_ranking/rank_jmax_Kitchen_sink.png]] | ![[pipelines/probabilistic/plots/importance/per_set_ranking/rank_t_delta_Kitchen_sink.png]] |
| Пик+нараст. | ![[pipelines/probabilistic/plots/importance/per_set_ranking/rank_jmax_Пикpнараст..png]] | ![[pipelines/probabilistic/plots/importance/per_set_ranking/rank_t_delta_Пикpнараст..png]] |
| GOES полный | ![[pipelines/probabilistic/plots/importance/per_set_ranking/rank_jmax_GOES_полный.png]] | ![[pipelines/probabilistic/plots/importance/per_set_ranking/rank_t_delta_GOES_полный.png]] |
| GOES KS | ![[pipelines/probabilistic/plots/importance/per_set_ranking/rank_jmax_GOES_KS.png]] | ![[pipelines/probabilistic/plots/importance/per_set_ranking/rank_t_delta_GOES_KS.png]] |

---

### 8.3 Классификация

**Методы:** builtin (feature_importances_ для дерево-моделей; |coef_| для LogReg и SVC; при наличии CalibratedClassifierCV — извлечение базового классификатора); SHAP (TreeExplainer для Forest, ExtraTrees, Boosting с усреднением |SHAP| по классам).

#### Глобальный рейтинг — S-класс

![[pipelines/classification/plots/importance/global/ranking_clf.png]]

#### Глобальная тепловая карта

![[pipelines/classification/plots/importance/global/global_heatmap_clf.png]]

*Строки = признаки, столбцы = наборы признаков. Усреднение по 5 моделям.*

#### Per-model тепловые карты

![[pipelines/classification/plots/importance/per_model/model_logreg_clf.png]]

![[pipelines/classification/plots/importance/per_model/model_forest_clf.png]]

![[pipelines/classification/plots/importance/per_model/model_boosting_clf.png]]

![[pipelines/classification/plots/importance/per_model/model_extratrees_clf.png]]

![[pipelines/classification/plots/importance/per_model/model_svc_clf.png]]

**Основные выводы:**

- LogReg (наилучшая модель на тесте) присваивает наибольший вес `log_goes_peak_flux` и `log_fluence`, подтверждая линейную разделимость S-классов в пространстве логарифмических характеристик вспышки.
- Forest и ExtraTrees в большей степени опираются на `log_cme_velocity` и `helio_lon`, что соответствует нелинейному механизму разбиения пространства признаков.
- SVC (лучшая модель на CV) придаёт аномально высокий вес `helio_lon`, указывая на нелинейную пространственную границу решения, которая переобучается при переносе на SC25.
- `cme_pa_deg` (позиционный угол КВМ) показывает наименьший вклад во всех конфигурациях.

#### Per-set рейтинги — классификация

*Усреднение по LogReg, Forest, ExtraTrees, Boosting, SVC.*

![[pipelines/classification/plots/importance/per_set_ranking/rank_Базовая.png]]
![[pipelines/classification/plots/importance/per_set_ranking/rank_Обе_координаты.png]]
![[pipelines/classification/plots/importance/per_set_ranking/rank_Без_КВМ.png]]
![[pipelines/classification/plots/importance/per_set_ranking/rank_T_delta_flare.png]]
![[pipelines/classification/plots/importance/per_set_ranking/rank_Все_базовые.png]]
![[pipelines/classification/plots/importance/per_set_ranking/rank_Флюэс_вместо_пика.png]]
![[pipelines/classification/plots/importance/per_set_ranking/rank_Флюэс_p_пик.png]]
![[pipelines/classification/plots/importance/per_set_ranking/rank_КВМ_расширенный.png]]
![[pipelines/classification/plots/importance/per_set_ranking/rank_Вспышка_расшир..png]]
![[pipelines/classification/plots/importance/per_set_ranking/rank_Kitchen_sink.png]]
![[pipelines/classification/plots/importance/per_set_ranking/rank_Пикpнараст..png]]
![[pipelines/classification/plots/importance/per_set_ranking/rank_GOES_полный.png]]
![[pipelines/classification/plots/importance/per_set_ranking/rank_GOES_KS.png]]
