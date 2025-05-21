import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import re


def process_23_cycle(file_path: str) -> pd.DataFrame:
    """Обработка данных 23 солнечного цикла"""
    df = pd.read_excel(file_path, sheet_name="AllPages")
    df.columns = df.columns.str.replace(r"\n|\s+", "", regex=True)
    df = df.drop([name for name in df.columns if "Unnamed" in name], axis=1)
    print(df.info())

    # 1. Парсинг Event name
    df["Event_date"] = (
        df["Eventname"]
        .astype(str)
        .apply(
            lambda x: pd.to_datetime(x.split("-")[0], format="%Y%m%d", errors="coerce")
        )
    )

    # 2. Обработка T0
    def parse_t0(event_date, t0_str):
        try:
            return event_date + timedelta(hours=int(re.sub(r"\D", "", t0_str)))
        except:
            return pd.NaT

    df["T0_datetime"] = df.apply(lambda x: parse_t0(x["Event_date"], x["Tо"]), axis=1)

    # 3. Парсинг Tmax
    def parse_tmax(event_date, tmax_str):
        results = []
        for part in str(tmax_str).split("\n"):
            match = re.match(r"(\d{1,2})d(\d{1,2})h?", part.strip())
            if match:
                days = int(match.group(1))
                hours = int(match.group(2))
                results.append(event_date.replace(day=days) + timedelta(hours=hours))
        return results if results else [pd.NaT]

    df["Tmax_parsed"] = df.apply(
        lambda x: parse_tmax(x["Event_date"], x["Tmax"]), axis=1
    )

    def parse_jmax(jmax_str):
        results = []
        for part in str(jmax_str).split("\n"):
            results.append(part)

        return results

    df["Jmax_parsed"] = df.apply(lambda x: parse_jmax(x["Jmax"]), axis=1)

    def split_rows(df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Усовершенствованная функция для разделения строк с обработкой float
        """
        # Преобразуем одиночные значения в списки
        for col in columns:
            df[col] = df[col].apply(
                lambda x: [x] if not isinstance(x, (list, tuple)) else x
            )

        # Создаем временный столбец с кортежами значений
        df["_temp_"] = df.apply(
            lambda row: list(zip(*[row[col] for col in columns])), axis=1
        )

        # Разделяем строки и преобразуем кортежи в отдельные колонки
        df = df.explode("_temp_", ignore_index=True)
        df[columns] = pd.DataFrame(df["_temp_"].tolist(), index=df.index)

        return df.drop(columns=["_temp_"])

    df = split_rows(df, ["Tmax_parsed", "Jmax_parsed"])

    def parse_peak_time(event_date: pd.Timestamp, time_str: str) -> list:
        """Парсинг колонки 'Time of peak intensity' с обработкой специальных символов"""
        results = []

        # Разделение на отдельные временные метки
        parts = re.split(r"[●▲Ø○■□SC]+", str(time_str))

        for part in parts:
            # Очистка и нормализация строки
            clean_part = re.sub(r"[^\d<dhms]", "", part.strip()).lower()
            if not clean_part:
                continue

            # Обработка форматов с '<'
            if "<" in clean_part:
                clean_part = clean_part.replace("<", "")
                # Пропускаем метки с '<' как отдельный кейс (по ТЗ не требуется смещение)

            # Основные шаблоны
            patterns = [
                (r"(\d{1,2})d(\d{1,2})h(\d{1,2})m", 3),  # 04d05h58m
                (r"(\d{1,2})d(\d{1,2})h", 2),  # 24d13h
                (r"(\d{1,2})h(\d{1,2})m", 2),  # 05h58m
                (r"(\d{1,2})d", 1),  # 13d
                (r"(\d{1,2})h", 1),  # 11h
                (r"(\d{1,2})m", 1),  # 58m
            ]

            matched = False
            for pattern, groups in patterns:
                match = re.match(pattern, clean_part)
                if match:
                    try:
                        days = int(match.group(1)) if groups >= 1 else 0
                        hours = int(match.group(2)) if groups >= 2 else 0
                        minutes = int(match.group(3)) if groups >= 3 else 0

                        new_date = event_date.replace(day=days) + timedelta(
                            hours=hours, minutes=minutes
                        )
                        results.append(new_date)
                        matched = True
                        break
                    except (ValueError, AttributeError):
                        continue

            if not matched:
                results.append(pd.NaT)

        return results if results else [pd.NaT]

    def extract_first_peak_time(peak_time):
        """Извлекает первое значение из списка временных меток"""
        if isinstance(peak_time, list):
            return peak_time[0] if len(peak_time) > 0 else pd.NaT
        return peak_time

    df["Peak_time"] = df.apply(
        lambda x: parse_peak_time(x["Event_date"], x["Timeofpeakintensity"]), axis=1
    )
    df["Peak_time"] = df["Peak_time"].apply(extract_first_peak_time)

    # 6. Разделение класса вспышек
    def parse_flare_class(flare_str):
        """
        Извлекает интенсивность солнечной вспышки и конвертирует в мощность (Вт/м²)

        Параметры:
        flare_str (str): Строка с классом вспышки (например, "C4.4/SF" или "2N/X1.2")

        Возвращает:
        float: Мощность вспышки в Вт/м², None если невозможно определить
        """
        if pd.isna(flare_str) or flare_str == "-" or flare_str == "...":
            return None

        # Очистка строки от специальных символов
        flare_str = str(flare_str).replace(">", "").replace("~", "")

        # Замена кириллической 'С' на латинскую 'C' (проблема со строкой "С4.4/SF")
        flare_str = flare_str.replace("С", "C")

        # Регулярное выражение для поиска класса вспышки (A, B, C, M, X + число)
        intensity_pattern = re.compile(r"([ABCMX])(\d+\.?\d*)")

        # Проверка форматов записи (интенсивность/важность или важность/интенсивность)
        if "/" in flare_str:
            parts = flare_str.split("/", 1)
            intensity = None

            # Проверяем обе части на наличие интенсивности
            for part in parts:
                match = intensity_pattern.search(part)
                if match:
                    intensity = (match.group(1), float(match.group(2)))
                    break

            if intensity is None:
                return None
        else:
            # Проверяем всю строку
            match = intensity_pattern.search(flare_str)
            if match:
                intensity = (match.group(1), float(match.group(2)))
            else:
                return None

        # Преобразование класса вспышки в мощность по таблице
        class_letter, multiplier = intensity

        if class_letter == "A":
            return multiplier * 1.0e-8  # меньше 10⁻⁷
        elif class_letter == "B":
            return multiplier * 1.0e-7  # от 1.0×10⁻⁷ до 10⁻⁶
        elif class_letter == "C":
            return multiplier * 1.0e-6  # от 1.0×10⁻⁶ до 10⁻⁵
        elif class_letter == "M":
            return multiplier * 1.0e-5  # от 1.0×10⁻⁵ до 10⁻⁴
        elif class_letter == "X":
            return multiplier * 1.0e-4  # больше 10⁻⁴

        return None

    df["Flare_power"] = df["Classofflare"].apply(parse_flare_class)
    df = df.drop(["Eventname", "Tо", "Tmax", "Jmax"], axis=1)

    def add_time_delta_columns(df):
        """Добавление колонок с временными интервалами"""
        # Временные интервалы между событиями
        df["T_delta_flare"] = (
            df["T0_datetime"] - df["Peak_time"]
        ).dt.total_seconds() / 3600
        df["T_delta_SPE"] = (
            df["Tmax_parsed"] - df["T0_datetime"]
        ).dt.total_seconds() / 3600

        # Обработка выбросов и некорректных значений
        df.loc[df["T_delta_flare"] < -24, "T_delta_flare"] = (
            np.nan
        )  # Отрицательные значения больше суток
        df.loc[df["T_delta_SPE"] < 0, "T_delta_SPE"] = np.nan  # Отрицательные интервалы

        return df

    df = add_time_delta_columns(df)
    df = df.dropna(
        subset=[
            "Event_date",
            "Tmax_parsed",
            "T_delta_flare",
            "T_delta_SPE",
            "Jmax_parsed",
            "Flare_power",
        ]
    )

    return df


def process_24_cycle(file_path: str) -> pd.DataFrame:
    """Обработка данных 24 солнечного цикла"""
    df = pd.read_excel(file_path, sheet_name="AllPages")
    df.columns = df.columns.str.replace(r"\n|\s+", "", regex=True)
    df = df.drop([name for name in df.columns if "Unnamed" in name], axis=1)

    # 1. Парсинг Event name
    df["Event_date"] = (
        df["Eventname"]
        .astype(str)
        .apply(
            lambda x: pd.to_datetime(
                x.replace(".", "").split("-")[0], format="%Y%m%d", errors="coerce"
            )
        )
    )

    # 2. Обработка T0
    def parse_t0(event_date, t0_str):
        try:
            return event_date + timedelta(hours=int(re.sub(r"\D", "", t0_str)))
        except:
            return pd.NaT

    df["T0_datetime"] = df.apply(lambda x: parse_t0(x["Event_date"], x["Tо"]), axis=1)

    # 3. Парсинг Tmax
    def parse_tmax(event_date: pd.Timestamp, tmax_str: str) -> list:
        """Парсинг Tmax с абсолютными днями и обработкой ошибок"""
        results = []
        for part in str(tmax_str).strip().split("\n"):
            try:
                # Удаляем все нецифровые символы кроме 'd' и 'h'
                clean_part = re.sub(r"[^\dh]", "", part)

                # Обработка формата с днями и часами
                if "d" in clean_part:
                    days_str, hours_str = re.split(r"d", clean_part, maxsplit=1)
                    days = int(days_str) if days_str else 0
                    hours = int(hours_str.replace("h", "")) if hours_str else 0
                else:
                    if len(clean_part) > 3:
                        hours = clean_part[-3:]
                        days_str = clean_part.replace(hours, "", 1)
                        days = int(days_str) if days_str else 0
                        hours = int(hours.replace("h", "")) if clean_part else 0
                    else:
                        days = event_date.day
                        hours = int(clean_part.replace("h", "")) if clean_part else 0

                # Создание новой даты
                new_date = event_date.replace(day=days) + pd.Timedelta(hours=hours)
                results.append(new_date)

            except (ValueError, AttributeError, TypeError) as e:
                print(f"Ошибка парсинга '{part}': {str(e)}")
                results.append(pd.NaT)

        return results if results else [pd.NaT]

    def parse_jmax(jmax_str):
        results = []
        for part in str(jmax_str).split("\n"):
            results.append(part)

        return results

    def split_tmax_jmax(df: pd.DataFrame) -> pd.DataFrame:
        """Разделение строк с множественными значениями Tmax и Jmax"""
        # Нормализация данных
        for col in ["Tmax_parsed", "Jmax_parsed"]:
            df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [x])

        # Создание временного столбца с кортежами
        df["_temp_"] = df.apply(
            lambda row: list(zip(row["Tmax_parsed"], row["Jmax_parsed"])), axis=1
        )

        # Разделение строк
        df = df.explode("_temp_", ignore_index=True)

        # Восстановление колонок
        df[["Tmax_parsed", "Jmax_parsed"]] = pd.DataFrame(
            df["_temp_"].tolist(), index=df.index
        )

        return df.drop(columns=["_temp_"])

    df["Tmax_parsed"] = df.apply(
        lambda x: parse_tmax(event_date=x["Event_date"], tmax_str=x["Tmax"]), axis=1
    )
    df["Jmax_parsed"] = df.apply(lambda x: parse_jmax(x["Jmax"]), axis=1)
    df = split_tmax_jmax(df)

    # 5. Время пиковой интенсивности
    def parse_peak_time(event_date: pd.Timestamp, time_str: str) -> list:
        """Парсинг колонки 'Time of peak intensity' с обработкой специальных символов"""
        results = []

        # Разделение на отдельные временные метки
        parts = re.split(r"[●▲Ø○■□SC]+", str(time_str))

        for part in parts:
            # Очистка и нормализация строки
            clean_part = re.sub(r"[^\d<dhms]", "", part.strip()).lower()
            if not clean_part:
                continue

            # Обработка форматов с '<'
            if "<" in clean_part:
                clean_part = clean_part.replace("<", "")
                # Пропускаем метки с '<' как отдельный кейс (по ТЗ не требуется смещение)

            # Основные шаблоны
            patterns = [
                (r"(\d{1,2})d(\d{1,2})h(\d{1,2})m", 3),  # 04d05h58m
                (r"(\d{1,2})d(\d{1,2})h", 2),  # 24d13h
                (r"(\d{1,2})h(\d{1,2})m", 2),  # 05h58m
                (r"(\d{1,2})d", 1),  # 13d
                (r"(\d{1,2})h", 1),  # 11h
                (r"(\d{1,2})m", 1),  # 58m
            ]

            matched = False
            for pattern, groups in patterns:
                match = re.match(pattern, clean_part)
                if match:
                    try:
                        days = int(match.group(1)) if groups >= 1 else 0
                        hours = int(match.group(2)) if groups >= 2 else 0
                        minutes = int(match.group(3)) if groups >= 3 else 0

                        new_date = event_date.replace(day=days) + timedelta(
                            hours=hours, minutes=minutes
                        )
                        results.append(new_date)
                        matched = True
                        break
                    except (ValueError, AttributeError):
                        continue

            if not matched:
                results.append(pd.NaT)

        return results if results else [pd.NaT]

    def extract_first_peak_time(peak_time):
        """Извлекает первое значение из списка временных меток"""
        if isinstance(peak_time, list):
            return peak_time[0] if len(peak_time) > 0 else pd.NaT
        return peak_time

    # Применение функции к DataFrame
    df["Peak_time"] = df.apply(
        lambda x: parse_peak_time(x["Event_date"], x["Timeofpeakintensity"]), axis=1
    )
    df["Peak_time"] = df["Peak_time"].apply(extract_first_peak_time)

    # 6. Разделение класса вспышек
    def parse_flare_class(flare_str):
        """
        Извлекает интенсивность солнечной вспышки и конвертирует в мощность (Вт/м²)

        Параметры:
        flare_str (str): Строка с классом вспышки (например, "C4.4/SF" или "2N/X1.2")

        Возвращает:
        float: Мощность вспышки в Вт/м², None если невозможно определить
        """
        if pd.isna(flare_str) or flare_str == "-" or flare_str == "...":
            return None

        # Очистка строки от специальных символов
        flare_str = str(flare_str).replace(">", "").replace("~", "")

        # Замена кириллической 'С' на латинскую 'C' (проблема со строкой "С4.4/SF")
        flare_str = flare_str.replace("С", "C")

        # Регулярное выражение для поиска класса вспышки (A, B, C, M, X + число)
        intensity_pattern = re.compile(r"([ABCMX])(\d+\.?\d*)")

        # Проверка форматов записи (интенсивность/важность или важность/интенсивность)
        if "/" in flare_str:
            parts = flare_str.split("/", 1)
            intensity = None

            # Проверяем обе части на наличие интенсивности
            for part in parts:
                match = intensity_pattern.search(part)
                if match:
                    intensity = (match.group(1), float(match.group(2)))
                    break

            if intensity is None:
                return None
        else:
            # Проверяем всю строку
            match = intensity_pattern.search(flare_str)
            if match:
                intensity = (match.group(1), float(match.group(2)))
            else:
                return None

        # Преобразование класса вспышки в мощность по таблице
        class_letter, multiplier = intensity

        if class_letter == "A":
            return multiplier * 1.0e-8  # меньше 10⁻⁷
        elif class_letter == "B":
            return multiplier * 1.0e-7  # от 1.0×10⁻⁷ до 10⁻⁶
        elif class_letter == "C":
            return multiplier * 1.0e-6  # от 1.0×10⁻⁶ до 10⁻⁵
        elif class_letter == "M":
            return multiplier * 1.0e-5  # от 1.0×10⁻⁵ до 10⁻⁴
        elif class_letter == "X":
            return multiplier * 1.0e-4  # больше 10⁻⁴

        return None

    df["Flare_power"] = df["Classofflare"].apply(parse_flare_class)

    def add_time_delta_columns(df):
        """Добавление колонок с временными интервалами"""
        # Временные интервалы между событиями
        df["T_delta_flare"] = (
            df["T0_datetime"] - df["Peak_time"]
        ).dt.total_seconds() / 3600
        df["T_delta_SPE"] = (
            df["Tmax_parsed"] - df["T0_datetime"]
        ).dt.total_seconds() / 3600

        # Обработка выбросов и некорректных значений
        df.loc[df["T_delta_flare"] < -24, "T_delta_flare"] = (
            np.nan
        )  # Отрицательные значения больше суток
        df.loc[df["T_delta_SPE"] < 0, "T_delta_SPE"] = np.nan  # Отрицательные интервалы

        return df

    df = add_time_delta_columns(df)
    df = df.dropna(
        subset=[
            "Event_date",
            "Tmax",
            "Tmax_parsed",
            "T_delta_flare",
            "T_delta_SPE",
            "Jmax_parsed",
            "Flare_power",
        ]
    )

    return df


def process_25_cycle(file_path: str) -> pd.DataFrame:
    """Обработка данных солнечных протонных событий из xlsx файла"""
    
    # Загрузка данных из Excel файла
    df = pd.read_excel(file_path, sheet_name="Отфильтрованные")
    
    # Преобразование даты события в datetime формат
    df["Event_date"] = df["Event Date"].apply(
        lambda x: pd.to_datetime(x.split("-")[0], format="%Y.%m.%d", errors="coerce")
    )
    
    # Парсинг времени начала события (Start)
    def parse_start_time(event_date, start_str):
        try:
            match = re.search(r"(\d{1,2})h(\d{1,2})m", start_str)
            if match:
                hours = int(match.group(1))
                minutes = int(match.group(2))
                return event_date + timedelta(hours=hours, minutes=minutes)
            return pd.NaT
        except:
            return pd.NaT
    
    df["Start_datetime"] = df.apply(
        lambda x: parse_start_time(x["Event_date"], x["Start (Day/UT)"]), axis=1
    )
    
    # Парсинг времени максимума (Tmax1)
    def parse_tmax(event_date, tmax_str):
        try:
            match = re.search(r"(\d{1,2})d(\d{1,2})h(\d{1,2})m", tmax_str)
            if match:
                days = int(match.group(1))
                hours = int(match.group(2))
                minutes = int(match.group(3))
                
                # Создаем дату с днем из строки и месяцем/годом из event_date
                new_date = event_date.replace(day=days) + timedelta(hours=hours, minutes=minutes)
                return new_date
            return pd.NaT
        except:
            return pd.NaT
    
    df["Tmax_parsed"] = df.apply(
        lambda x: parse_tmax(x["Event_date"], x["Tmax1 (UT)"]), axis=1
    )
    
    # Парсинг времени вспышки (T0 FL)
    def parse_flare_time(event_date, flare_str):
        try:
            match = re.search(r"(\d{1,2})d(\d{1,2})h(\d{1,2})m", flare_str)
            if match:
                days = int(match.group(1))
                hours = int(match.group(2))
                minutes = int(match.group(3))
                
                new_date = event_date.replace(day=days) + timedelta(hours=hours, minutes=minutes)
                return new_date
            return pd.NaT
        except:
            return pd.NaT
    
    df["Flare_datetime"] = df.apply(
        lambda x: parse_flare_time(x["Event_date"], x["T0 FL (Day/UT)"]), axis=1
    )
    
    # Преобразование Jmax1 в числовой формат
    df["Jmax_parsed"] = pd.to_numeric(df["Jmax1 (pfu)"], errors="coerce")
    
    # Парсинг класса вспышки
    def parse_flare_class(flare_str):
        """
        Извлекает интенсивность солнечной вспышки и конвертирует в мощность (Вт/м²)
        """
        if pd.isna(flare_str) or flare_str == "-" or flare_str == "...":
            return None
            
        # Обработка случаев с несколькими вспышками
        if ";" in flare_str:
            parts = flare_str.split(";")
            flare_str = parts[0].strip()  # Берем первую вспышку
            
        # Очистка строки от специальных символов
        flare_str = str(flare_str).replace(">", "").replace("~", "").replace('"', '')
        
        # Замена кириллической 'С' на латинскую 'C'
        flare_str = flare_str.replace("С", "C")
        
        # Замена запятой на точку в числовых значениях
        flare_str = re.sub(r'(\d),(\d)', r'\1.\2', flare_str)
        
        # Регулярное выражение для поиска класса вспышки
        intensity_pattern = re.compile(r"([ABCMX])(\d+\.?\d*)")
        
        # Проверка форматов записи
        if "/" in flare_str:
            parts = flare_str.split("/", 1)
            intensity = None
            
            # Проверяем обе части на наличие интенсивности
            for part in parts:
                match = intensity_pattern.search(part)
                if match:
                    intensity = (match.group(1), float(match.group(2)))
                    break
                    
            if intensity is None:
                return None
        else:
            # Проверяем всю строку
            match = intensity_pattern.search(flare_str)
            if match:
                intensity = (match.group(1), float(match.group(2)))
            else:
                return None
                
        # Преобразование класса вспышки в мощность
        class_letter, multiplier = intensity
        
        if class_letter == "A":
            return multiplier * 1.0e-8  # меньше 10⁻⁷
        elif class_letter == "B":
            return multiplier * 1.0e-7  # от 1.0×10⁻⁷ до 10⁻⁶
        elif class_letter == "C":
            return multiplier * 1.0e-6  # от 1.0×10⁻⁶ до 10⁻⁵
        elif class_letter == "M":
            return multiplier * 1.0e-5  # от 1.0×10⁻⁵ до 10⁻⁴
        elif class_letter == "X":
            return multiplier * 1.0e-4  # больше 10⁻⁴
            
        return None
    
    df["Flare_power"] = df["Importance (Xray/Opt)"].apply(parse_flare_class)
    
    # Добавление временных интервалов
    def add_time_delta_columns(df):
        """Добавление колонок с временными интервалами"""
        
        # Временной интервал между вспышкой и началом СПС
        df["T_delta_flare"] = (
            df["Start_datetime"] - df["Flare_datetime"]
        ).dt.total_seconds() / 3600
        
        # Временной интервал между началом СПС и его максимумом
        df["T_delta_SPE"] = (
            df["Tmax_parsed"] - df["Start_datetime"]
        ).dt.total_seconds() / 3600
        
        # Обработка выбросов и некорректных значений
        # df.loc[df["T_delta_flare_to_start"] < -24, "T_delta_flare_to_start"] = np.nan
        # df.loc[df["T_delta_start_to_max"] < 0, "T_delta_start_to_max"] = np.nan
        # df.loc[df["T_delta_flare_to_max"] < -24, "T_delta_flare_to_max"] = np.nan
        
        return df
    
    df = add_time_delta_columns(df)
    
    # Создание финального DataFrame с нужными колонками
    result_df = df[[
        "Event_date",
        "Tmax_parsed",
        "Jmax_parsed",
        "Flare_power",
        "T_delta_flare",
        "T_delta_SPE"
    ]]
    
    return result_df

def filter_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Фильтрация аномальных строк по 5 ключевым критериям"""
    df = df.dropna(
        subset=[
            "Event_date",
            "Tmax_parsed",
            "T_delta_flare",
            "T_delta_SPE",
            "Jmax_parsed",
            "Flare_power",
        ]
    )

    # 1. Фильтр по временным меткам
    time_mask = (
        (df['Tmax_parsed'].dt.month != df['Event_date'].dt.month) | 
        (df['T_delta_SPE'] > 180)
    )
    
    # 2. Фильтр по физическим параметрам
    physics_mask = (
        (df['Flare_power'] < 1e-8) |
        (df['Jmax_parsed'] > 1e5)
        # (df['γ'] > 10)  # Максимальное значение по ГОСТ 25645.145-91
    )
    
    # 3. Фильтр по форматам данных
    # format_mask = (
    #     df['Flare_class'].str.contains(r'[^A-BC-MX0-9./]', na=True)
    #     df['Localization'].isna()
    # )
    
    # Комбинированная маска
    combined_mask = time_mask | physics_mask

    stats = {
        'Всего записей': len(df),
        'Временные аномалии': time_mask.sum(),
        'Физические аномалии': physics_mask.sum(),
        # 'Форматные ошибки': format_mask.sum(),
        'Общее количество аномалий': combined_mask.sum(),
        'Процент аномалий': f"{combined_mask.mean()*100:.2f}%"
    }
    print(stats)
    # Создание отчета об аномалиях
    # anomalies = df[combined_mask]
    # anomalies.to_excel('solar_anomalies.xlsx', index=False)
    
    return df[~combined_mask]

def convert_column_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Приведение типов данных для ключевых колонок солнечных событий.
    
    Параметры:
    df - исходный DataFrame
    
    Возвращает:
    DataFrame с корректными типами данных
    """
    # Словарь соответствия колонок и функций преобразования
    converters = {
        'Event_date': pd.to_datetime,
        'Peak_time': pd.to_datetime,
        'T0_datetime': pd.to_datetime,
        'Tmax_parsed': pd.to_datetime,
        'T_delta_flare': pd.to_numeric,
        'T_delta_SPE': pd.to_numeric,
        'Jmax_parsed': pd.to_numeric,
        'Flare_power': pd.to_numeric
    }
    
    for col, func in converters.items():
        if col in df.columns:
            # Для временных колонок удаляем часовой пояс если есть
            if 'time' in col.lower() or 'date' in col.lower():
                df[col] = df[col].astype(str).str.replace(r'\+0[0-9]:00$', '', regex=True)
            
            # Применяем преобразование с обработкой ошибок
            try:
                df[col] = func(df[col], errors='coerce')
            except TypeError:
                df[col] = func(df[col].astype(str), errors='coerce')
                
    return df

def remove_duplicates_by_event_date_and_delta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Удаляет дублирующиеся строки, оставляя только первую строку для каждой уникальной 
    комбинации даты события (Event_date) и временного интервала вспышки (T_delta_flare).
    
    Параметры:
    df (pd.DataFrame): Исходный DataFrame с данными о солнечных событиях
    
    Возвращает:
    pd.DataFrame: DataFrame без дубликатов по комбинации Event_date и T_delta_flare
    """
    # Проверка наличия необходимых колонок
    required_columns = ['Event_date', 'T_delta_flare']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Колонка '{col}' не найдена в DataFrame")
    
    # Преобразование Event_date в datetime, если это еще не сделано
    if not pd.api.types.is_datetime64_any_dtype(df['Event_date']):
        df['Event_date'] = pd.to_datetime(df['Event_date'], errors='coerce')
    
    # Преобразование T_delta_flare в числовой формат, если это еще не сделано
    if not pd.api.types.is_numeric_dtype(df['T_delta_flare']):
        df['T_delta_flare'] = pd.to_numeric(df['T_delta_flare'], errors='coerce')
    
    # Округление T_delta_flare для более надежного сравнения
    df['T_delta_flare_rounded'] = np.round(df['T_delta_flare'], 2)
    
    # Подсчет количества дубликатов перед удалением
    duplicates_count = df.duplicated(subset=['Event_date', 'T_delta_flare_rounded']).sum()
    
    # Удаление дубликатов, оставляя первую строку для каждой уникальной комбинации
    df_cleaned = df.drop_duplicates(
        subset=['Event_date', 'T_delta_flare_rounded'], 
        keep='first'
    ).reset_index(drop=True)
    
    # Удаление временной колонки с округленными значениями
    df_cleaned = df_cleaned.drop(columns=['T_delta_flare_rounded'])
    
    print(f"Удалено {duplicates_count} дублирующихся строк")
    
    return df_cleaned

df_23 = process_23_cycle("data/23 цикл.xlsx")
df_24 = process_24_cycle("data/24 цикл.xlsx")
df_25 = process_25_cycle("data/25 цикл.xlsx")

df = pd.concat(
    (
        df_23[
            [
                "Event_date",
                "T_delta_flare",
                "Tmax_parsed",
                "T_delta_SPE",
                "Jmax_parsed",
                "Flare_power",
            ]
        ],
        df_24[
            [
                "Event_date",
                "T_delta_flare",
                "Tmax_parsed",
                "T_delta_SPE",
                "Jmax_parsed",
                "Flare_power",
            ]
        ],
        # df_25[
        #     [
        #         "Event_date",
        #         "T_delta_flare",
        #         "Tmax_parsed",
        #         "T_delta_SPE",
        #         "Jmax_parsed",
        #         "Flare_power",
        #     ]
        # ],
    )
)
df = convert_column_types(df)
df = filter_anomalies(df)
df = remove_duplicates_by_event_date_and_delta(df)

df_25 = convert_column_types(df_25)
df_25 = filter_anomalies(df_25)
df_25 = remove_duplicates_by_event_date_and_delta(df_25)
