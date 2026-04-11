import pandas as pd
import numpy as np
from datetime import timedelta
import re

C23_START = pd.Timestamp("1996-08-01")  # минимум ~авг 1996 → старт SC23 [web:6]
C24_START = pd.Timestamp("2008-12-01")  # минимум дек 2008 → старт SC24 [web:14]
C25_START = pd.Timestamp("2019-12-01")  # минимум дек 2019 → старт SC25 [web:22]

def process_unified_sps_data(file_path: str, sheet_name: str = "ОБЩАЯ БД") -> pd.DataFrame:
    """
    Универсальная функция обработки данных солнечных протонных событий (СПС).
    Объединяет функционал process_23_cycle, process_24_cycle и process_25_cycle.

    Параметры:
    ----------
    file_path : str
        Путь к Excel файлу с данными СПС
    sheet_name : str, optional
        Название листа в Excel файле (по умолчанию "ОБЩАЯ БД")

    Возвращает:
    -----------
    pd.DataFrame
        Обработанный датафрейм с колонками:
        - Event_date: дата события
        - Tmax_parsed: время максимума события
        - Jmax_parsed: максимальный поток протонов (pfu)
        - Flare_power: мощность вспышки (Вт/м²)
        - T_delta_flare: временной интервал от вспышки до начала СПС (часы)
        - T_delta_SPE: временной интервал от начала СПС до максимума (часы)

    Примечания:
    -----------
    - Автоматически обрабатывает множественные значения Tmax/Jmax
    - Корректно парсит научную нотацию (1.6∙103 → 1600)
    - Поддерживает различные форматы дат (YYYYMMDD-DDD и YYYY.MM.DD)
    """

    # Загрузка данных
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    initial_count = len(df)

    # ========================================================================
    # ПАРСИНГ ДАТЫ СОБЫТИЯ
    # ========================================================================
    def parse_event_date(event_str: str) -> pd.Timestamp:
        """
        Универсальный парсер даты события.
        Поддерживает форматы: YYYYMMDD-DDD, YYYY.MM.DD, YYYY.MM.DD-DDD
        """
        try:
            event_str = str(event_str).strip()

            # Формат: YYYYMMDD-DDD
            if '-' in event_str:
                date_part = event_str.split('-')[0]
                if len(date_part) == 8:
                    return pd.to_datetime(date_part, format='%Y%m%d')

            # Формат: YYYY.MM.DD
            if '.' in event_str:
                date_part = event_str.split()[0].split('-')[0]
                return pd.to_datetime(date_part, format='%Y.%m.%d')

        except:
            pass
        return pd.NaT

    df['Event_date'] = df['Event name'].apply(parse_event_date)


    # ========================================================================
    # ПАРСИНГ T0 (начало солнечного протонного события)
    # ========================================================================
    def parse_t0(event_date: pd.Timestamp, t0_str: str) -> pd.Timestamp:
        """
        Парсинг времени начала СПС.
        Форматы: HHh, DDdHHhMMm
        """
        if pd.isna(event_date):
            return pd.NaT

        try:
            t0_str = str(t0_str).strip()

            # Формат: DDdHHhMMm
            match = re.match(r'(\d{1,2})d(\d{1,2})h(\d{1,2})m', t0_str)
            if match:
                days, hours, minutes = map(int, match.groups())
                dt = event_date.replace(day=days) + timedelta(hours=hours, minutes=minutes)
                if dt < event_date:
                    dt += pd.DateOffset(months=1)
                return dt

            # Формат: HHh
            match = re.match(r'(\d{1,2})h', t0_str)
            if match:
                hours = int(match.group(1))
                return event_date + timedelta(hours=hours)
        except:
            pass

        return pd.NaT

    df['T0_datetime'] = df.apply(lambda x: parse_t0(x['Event_date'], x['T0']), axis=1)


    # ========================================================================
    # ПАРСИНГ TMAX (время максимума СПС)
    # ========================================================================
    def parse_tmax(event_date: pd.Timestamp, tmax_str: str) -> list:
        """
        Парсинг времени максимума СПС с поддержкой множественных значений.
        Форматы: DDdHHh, DDdHHhMMm, DDdHHHH (слитная запись)

        Возвращает список временных меток для обработки событий с несколькими максимумами.
        """
        if pd.isna(event_date):
            return [pd.NaT]

        results = []
        try:
            # Разделение по переносам строк для множественных значений
            parts = str(tmax_str).strip().split('\n')

            for part in parts:
                part = part.strip()

                # Формат: DDdHHhMMm
                match = re.match(r'(\d{1,2})d(\d{1,2})h(\d{1,2})m', part)
                if match:
                    days, hours, minutes = map(int, match.groups())
                    dt = event_date.replace(day=days) + timedelta(hours=hours, minutes=minutes)
                    if dt < event_date:
                        dt += pd.DateOffset(months=1)
                    results.append(dt)
                    continue

                # Формат: DDdHHh
                match = re.match(r'(\d{1,2})d(\d{1,2})h', part)
                if match:
                    days, hours = map(int, match.groups())
                    dt = event_date.replace(day=days) + timedelta(hours=hours)
                    if dt < event_date:
                        dt += pd.DateOffset(months=1)
                    results.append(dt)
                    continue

                # Формат: DDdHHHH (слитная запись типа 05d0220)
                match = re.match(r'(\d{1,2})d(\d{4})', part)
                if match:
                    days = int(match.group(1))
                    time_str = match.group(2)
                    hours = int(time_str[:2])
                    minutes = int(time_str[2:])
                    dt = event_date.replace(day=days) + timedelta(hours=hours, minutes=minutes)
                    if dt < event_date:
                        dt += pd.DateOffset(months=1)
                    results.append(dt)
                    continue
        except:
            pass

        return results if results else [pd.NaT]

    df['Tmax_parsed'] = df.apply(lambda x: parse_tmax(x['Event_date'], x['Tmax']), axis=1)


    # ========================================================================
    # ПАРСИНГ JMAX (максимальный поток протонов)
    # ========================================================================
    def parse_jmax(jmax_str: str) -> list:
        """
        Парсинг максимального потока протонов с корректной обработкой научной нотации.

        Примеры преобразований:
        - '860\n1.6∙103' → [860.0, 1600.0]
        - '7.2 103' → [7200.0]
        - '1.8 104' → [18000.0]

        Возвращает список значений для синхронизации с Tmax.
        """
        results = []

        try:
            parts = str(jmax_str).strip().split('\n')

            for part in parts:
                part = part.strip()

                # Обработка научной нотации: число∙степень или число степень
                match = re.match(r'([\d.]+)\s*[∙\s]\s*(\d+)', part)
                if match:
                    mantissa = float(match.group(1))
                    exponent_str = match.group(2)

                    # Определяем правильную степень
                    # 103 → 10^3, 104 → 10^4
                    if len(exponent_str) == 3 and exponent_str[0] == '1':
                        exponent = int(exponent_str[1:])
                    else:
                        exponent = len(exponent_str)

                    value = mantissa * (10 ** exponent)
                    results.append(value)
                    continue

                # Обычное число
                try:
                    value = float(part)
                    results.append(value)
                except ValueError:
                    results.append(np.nan)
        except:
            pass

        return results if results else [np.nan]

    df['Jmax_parsed'] = df['Jmax'].apply(parse_jmax)


    # ========================================================================
    # ПАРСИНГ ВРЕМЕНИ ВСПЫШКИ
    # ========================================================================
    def parse_flare_time(event_date: pd.Timestamp, flare_str: str) -> pd.Timestamp:
        """
        Парсинг времени солнечной вспышки.
        Форматы: DDdHHhMMm, DDdHHhMM
        Игнорирует метки SC (Solar CME).
        """
        if pd.isna(event_date):
            return pd.NaT

        try:
            flare_str = str(flare_str).strip()

            # Удаляем метки SC
            if 'SC' in flare_str:
                flare_str = flare_str.split('SC')[0].strip()

            # Формат: DDdHHh MMm или DDdHHhMMm
            match = re.search(r'(\d{1,2})d(\d{1,2})h\s*(\d{1,2})m', flare_str)
            if match:
                days, hours, minutes = map(int, match.groups())
                return event_date.replace(day=days) + timedelta(hours=hours, minutes=minutes)

            # Формат: DDdHHhMM (без 'm')
            match = re.search(r'(\d{1,2})d(\d{1,2})h(\d{2})(?!m)', flare_str)
            if match:
                days, hours, minutes = map(int, match.groups())
                return event_date.replace(day=days) + timedelta(hours=hours, minutes=minutes)
        except:
            pass

        return pd.NaT

    df['Flare_datetime'] = df.apply(lambda x: parse_flare_time(x['Event_date'], x['Flare T0']), axis=1)


    # ========================================================================
    # ПАРСИНГ КЛАССА ВСПЫШКИ
    # ========================================================================
    def parse_flare_class(flare_str: str) -> float:
        """
        Преобразование класса солнечной вспышки в мощность (Вт/м²).

        Шкала классов:
        - A: 1.0e-8 Вт/м² (< 10⁻⁷)
        - B: 1.0e-7 Вт/м² (10⁻⁷ – 10⁻⁶)
        - C: 1.0e-6 Вт/м² (10⁻⁶ – 10⁻⁵)
        - M: 1.0e-5 Вт/м² (10⁻⁵ – 10⁻⁴)
        - X: 1.0e-4 Вт/м² (> 10⁻⁴)

        Примеры:
        - 'C4.4/SF' → 4.4e-6
        - 'X2.1/3B' → 2.1e-4
        - 'M1,5' → 1.5e-5
        """
        if pd.isna(flare_str) or flare_str in ['-', '...', '']:
            return None

        try:
            # Очистка строки
            flare_str = str(flare_str).replace('>', '').replace('~', '').replace('"', '')
            flare_str = flare_str.replace('С', 'C')  # Кириллица → латиница
            flare_str = flare_str.replace(',', '.')  # Запятая → точка

            # Регулярное выражение для поиска класса вспышки
            pattern = re.compile(r'([ABCMX])(\d+\.?\d*)')

            # Проверяем все части строки (до и после '/')
            parts = flare_str.split('/') if '/' in flare_str else [flare_str]

            for part in parts:
                match = pattern.search(part)
                if match:
                    class_letter = match.group(1)
                    multiplier = float(match.group(2))

                    # Таблица коэффициентов мощности
                    power_table = {
                        'A': 1.0e-8,
                        'B': 1.0e-7,
                        'C': 1.0e-6,
                        'M': 1.0e-5,
                        'X': 1.0e-4
                    }

                    return multiplier * power_table.get(class_letter, 0)
        except:
            pass

        return None

    def assign_cycle(dt: pd.Timestamp) -> float:
        if pd.isna(dt):
            return np.nan
        if dt >= C25_START:
            return 25.0
        if dt >= C24_START:
            return 24.0
        if dt >= C23_START:
            return 23.0
        return np.nan

    df['Flare_power'] = df['Class of flare'].apply(parse_flare_class)

    # ========================================================================
    # РАЗДЕЛЕНИЕ СТРОК С МНОЖЕСТВЕННЫМИ ЗНАЧЕНИЯМИ
    # ========================================================================
    # Приведение к спискам
    for col in ['Tmax_parsed', 'Jmax_parsed']:
        df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [x])

    # Создание синхронизированных пар (Tmax, Jmax)
    df['_temp_'] = df.apply(
        lambda row: list(zip(row['Tmax_parsed'], row['Jmax_parsed'])), axis=1
    )

    if 'Cycle' in df.columns:
        df['Cycle'] = pd.to_numeric(df['Cycle'], errors='coerce')
        missing_mask = df['Cycle'].isna()
        df.loc[missing_mask, 'Cycle'] = df.loc[missing_mask, 'Event_date'].apply(assign_cycle)
    else:
        pass

    # Разворачивание строк
    df = df.explode('_temp_', ignore_index=True)
    df[['Tmax_parsed', 'Jmax_parsed']] = pd.DataFrame(df['_temp_'].tolist(), index=df.index)
    df = df.drop(columns=['_temp_'])


    # ========================================================================
    # ВЫЧИСЛЕНИЕ ВРЕМЕННЫХ ИНТЕРВАЛОВ
    # ========================================================================
    # Δt от вспышки до начала СПС (часы)
    df['T_delta_flare'] = (df['T0_datetime'] - df['Flare_datetime']).dt.total_seconds() / 3600
        
    # Δt от начала СПС до его максимума (часы)
    df['T_delta_SPE'] = (df['Tmax_parsed'] - df['T0_datetime']).dt.total_seconds() / 3600
    # Страховочный фикс: если после парсинга T_delta_SPE < 0 — Tmax попал в неверный месяц
    mask_month_edge = df["T_delta_SPE"] < 0
    df.loc[mask_month_edge, "Tmax_parsed"] = (
        df.loc[mask_month_edge, "Tmax_parsed"].apply(
            lambda t: t + pd.DateOffset(months=1) if pd.notna(t) else t
        )
    )
    df["T_delta_SPE"] = (df["Tmax_parsed"] - df["T0_datetime"]).dt.total_seconds() / 3600


    # ========================================================================
    # ФОРМИРОВАНИЕ ИТОГОВОГО ДАТАФРЕЙМА
    # ========================================================================
    final_cols = [
        'Event_date',
        'Tmax_parsed',
        'Jmax_parsed',
        'Flare_power',
        'T_delta_flare',
        'T_delta_SPE',
        'Cycle'
    ]

    df_final = df[final_cols].copy()

    # Вывод статистики обработки
    print(f"{'='*60}")
    print(f"СТАТИСТИКА ОБРАБОТКИ ДАННЫХ")
    print(f"{'='*60}")
    print(f"Исходных записей: {initial_count}")
    print(f"Обработано событий: {len(df_final)}")
    print(f"  - Валидные даты: {df_final['Event_date'].notna().sum()}")
    print(f"  - Валидные Tmax: {df_final['Tmax_parsed'].notna().sum()}")
    print(f"  - Валидные Jmax: {df_final['Jmax_parsed'].notna().sum()}")
    print(f"  - Валидные Flare_power: {df_final['Flare_power'].notna().sum()}")
    print(f"  - Валидные T_delta_flare: {df_final['T_delta_flare'].notna().sum()}")
    print(f"  - Валидные T_delta_SPE: {df_final['T_delta_SPE'].notna().sum()}")
    print(f"{'='*60}")

    return df_final


def filter_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Фильтрация аномальных записей по критериям:
    1. Временные аномалии (месяц Tmax ≠ месяц Event_date или T_delta_SPE > 180 часов)
    2. Физические аномалии (Flare_power < 1e-8 или Jmax > 1e5)

    Параметры:
    ----------
    df : pd.DataFrame
        Датафрейм с обработанными данными СПС

    Возвращает:
    -----------
    pd.DataFrame
        Отфильтрованный датафрейм без аномалий
    """

    # Удаление строк с пропущенными критическими значениями
    df = df.dropna(
        subset=['Event_date', 'Tmax_parsed', 'T_delta_flare', 'T_delta_SPE', 'Jmax_parsed', 'Flare_power']
    )

    # Фильтр временных аномалий
    time_mask = (
        (df['Tmax_parsed'].dt.month != df['Event_date'].dt.month) | 
        (df['T_delta_SPE'] > 180)
    )

    # Фильтр физических аномалий
    physics_mask = (
        (df['Flare_power'] < 1e-8) |
        (df['Jmax_parsed'] > 1e5)
    )

    # Комбинированная маска
    combined_mask = time_mask | physics_mask

    # Статистика
    stats = {
        'Всего записей': len(df),
        'Временные аномалии': time_mask.sum(),
        'Физические аномалии': physics_mask.sum(),
        'Общее количество аномалий': combined_mask.sum(),
        'Процент аномалий': f"{combined_mask.mean()*100:.2f}%"
    }

    print(f"\n{'='*60}")
    print("СТАТИСТИКА ФИЛЬТРАЦИИ АНОМАЛИЙ")
    print(f"{'='*60}")
    for key, value in stats.items():
        print(f"{key}: {value}")
    print(f"{'='*60}\n")

    return df[~combined_mask]


def remove_duplicates_by_event_date_and_delta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Удаление дубликатов по комбинации Event_date и T_delta_flare.
    Оставляет первую строку для каждой уникальной комбинации.

    Параметры:
    ----------
    df : pd.DataFrame
        Датафрейм с обработанными данными СПС

    Возвращает:
    -----------
    pd.DataFrame
        Датафрейм без дубликатов
    """

    # Проверка наличия необходимых колонок
    required_columns = ['Event_date', 'T_delta_flare']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Колонка '{col}' не найдена в DataFrame")

    # Преобразование типов данных
    if not pd.api.types.is_datetime64_any_dtype(df['Event_date']):
        df['Event_date'] = pd.to_datetime(df['Event_date'], errors='coerce')

    if not pd.api.types.is_numeric_dtype(df['T_delta_flare']):
        df['T_delta_flare'] = pd.to_numeric(df['T_delta_flare'], errors='coerce')

    # Округление T_delta_flare для надежного сравнения
    df['T_delta_flare_rounded'] = np.round(df['T_delta_flare'], 2)

    # Подсчет дубликатов
    duplicates_count = df.duplicated(subset=['Event_date', 'T_delta_flare_rounded']).sum()

    # Удаление дубликатов
    df_cleaned = df.drop_duplicates(
        subset=['Event_date', 'T_delta_flare_rounded'], 
        keep='first'
    ).reset_index(drop=True)

    # Удаление временной колонки
    df_cleaned = df_cleaned.drop(columns=['T_delta_flare_rounded'])

    print(f"Удалено {duplicates_count} дублирующихся строк\n")

    return df_cleaned


def filter_by_jmax(df: pd.DataFrame, threshold: float = 10) -> pd.DataFrame:
    """
    Фильтрация записей по пороговому значению Jmax.

    Параметры:
    ----------
    df : pd.DataFrame
        Датафрейм с обработанными данными СПС
    threshold : float, optional
        Пороговое значение Jmax (по умолчанию 10 pfu)

    Возвращает:
    -----------
    pd.DataFrame
        Датафрейм с записями, где Jmax > threshold
    """

    # Приведение к числовому типу
    jmax_numeric = pd.to_numeric(df['Jmax_parsed'], errors='coerce')

    # Фильтр
    mask = jmax_numeric > threshold
    filtered_count = mask.sum()

    print(f"Отобрано {filtered_count} записей с Jmax > {threshold} pfu\n")

    return df[mask].copy()

def report_negative_t_deltas(
    df: pd.DataFrame,
    which: str = "both",
    save_path: str | None = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Выводит и возвращает строки с отрицательными T_delta значениями.

    Параметры:
      - which: 'flare' | 'spe' | 'both'
        'flare' — только T_delta_flare < 0
        'spe'   — только T_delta_SPE < 0
        'both'  — строки, где хотя бы одно из T_delta < 0
      - save_path: путь к CSV для сохранения результата (или None)
      - verbose: печатать сводку и первые строки результата

    Возвращает:
      DataFrame с найденными строками и колонками:
        ['Event_date','Tmax_parsed','Jmax_parsed','Flare_power',
         'T_delta_flare','T_delta_SPE','cycle']
    """
    # Гарантируем числовой тип для масок
    tdf = df.copy()
    for c in ("T_delta_flare", "T_delta_SPE"):
        if c in tdf.columns:
            tdf[c] = pd.to_numeric(tdf[c], errors="coerce")

    mask_flare = tdf["T_delta_flare"] < 0 if "T_delta_flare" in tdf.columns else False
    mask_spe   = tdf["T_delta_SPE"]   < 0 if "T_delta_SPE"   in tdf.columns else False

    if which == "flare":
        mask = mask_flare
    elif which == "spe":
        mask = mask_spe
    else:  # both
        mask = mask_flare | mask_spe

    cols = [
        "Event_date",
        "Tmax_parsed",
        "Jmax_parsed",
        "Flare_power",
        "T_delta_flare",
        "T_delta_SPE",
        "cycle",
    ]
    cols = [c for c in cols if c in tdf.columns]
    neg_df = tdf.loc[mask, cols].copy()

    if verbose:
        total = len(tdf)
        cnt_flare = int(mask_flare.sum()) if isinstance(mask_flare, pd.Series) else 0
        cnt_spe = int(mask_spe.sum()) if isinstance(mask_spe, pd.Series) else 0
        cnt_any = int(mask.sum()) if isinstance(mask, pd.Series) else 0
        print("=" * 60)
        print("ОТРИЦАТЕЛЬНЫЕ T_delta ЗНАЧЕНИЯ")
        print("=" * 60)
        print(f"Всего строк: {total}")
        print(f"Отрицательных T_delta_flare: {cnt_flare}")
        print(f"Отрицательных T_delta_SPE  : {cnt_spe}")
        print(f"Итого найдено (which={which}): {cnt_any}")
        if "cycle" in neg_df.columns:
            print("\nРаспределение по циклам:")
            print(neg_df["cycle"].value_counts(dropna=False).sort_index())
        if not neg_df.empty:
            print("\nПервые строки результата:")
            print(neg_df.head(100).to_string(index=False))

    if save_path:
        neg_df.to_csv(save_path, index=False, encoding="utf-8")

    return neg_df

# =============================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# =============================================================================
if __name__ == "__main__":
    # Обработка данных
    df = process_unified_sps_data("data/БД СПС.xlsx")

    # Применение фильтров
    df = filter_anomalies(df)
    df = remove_duplicates_by_event_date_and_delta(df)
    df = filter_by_jmax(df, threshold=10)

    # Сохранение результата
    df.to_csv("data/processed.csv", index=False, encoding='utf-8')
    print("✓ Данные сохранены в 'processed.csv'")

    # Вывод итоговой статистики
    print(f"\n{'='*60}")
    print("ИТОГОВАЯ СТАТИСТИКА")
    print(f"{'='*60}")
    print(df.describe())

    # neg = report_negative_t_deltas(df, which="both", save_path="neg_t_deltas.csv")
