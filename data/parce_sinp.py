#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════
  Парсер каталога СПС 24/25-го цикла СА (сайт СИНП МГУ)
  Parser for SPE Catalog, Cycles 24-25 (swx.sinp.msu.ru)
═══════════════════════════════════════════════════════════════════

Парсит HTML-таблицу с сайта https://swx.sinp.msu.ru/apps/sep_events_cat/
и сохраняет в CSV.

Использование:
    # Из сохранённого HTML-файла (рекомендуется):
    python parse_sinp.py --html saved_page.html

    # Или скачать с сайта (нужен requests):
    python parse_sinp.py

    # Указать цикл и выходной файл:
    python parse_sinp.py --html page.html --cycle 24 --output cycle24.csv

Зависимости:
    pip install beautifulsoup4
    pip install requests  # только если без --html
"""

import re
import csv
import sys
import os
from datetime import datetime, timedelta

try:
    from bs4 import BeautifulSoup
except ImportError:
    print("Ошибка: pip install beautifulsoup4")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════
#  Вспомогательные функции
# ═══════════════════════════════════════════════════════════════

def parse_xray_class(class_str):
    """Извлечь рентгеновский класс и конвертировать в Вт/м²."""
    if not class_str:
        return None, None
    # Нормализация: запятая→точка, кириллические буквы
    cleaned = class_str.replace(',', '.').strip().rstrip(';')
    # Замена кириллических М, С на латинские
    cleaned = cleaned.replace('М', 'M').replace('С', 'C')
    # Берём первый класс если несколько через ";" или ","
    first = re.split(r'[;,]\s*(?=[ABCMX])', cleaned)[0].strip()
    m = re.search(r'([ABCMX])(\d+\.?\d*)', first)
    if not m:
        return None, None
    letter = m.group(1)
    num = float(m.group(2))
    scale = {'A': 1e-8, 'B': 1e-7, 'C': 1e-6, 'M': 1e-5, 'X': 1e-4}
    return f"{letter}{num}", scale.get(letter, 0) * num


def resolve_datetime_sinp(event_date_str, time_str, forward=True):
    """Конвертация времени из каталога СИНП в datetime.

    Форматы:
      "10h"             → день события, 10:00
      "19h"             → день события, 19:00
      "08d05h"          → 8-е число, 05:00
      "01d06h55m"       → 1-е число, 06:55
      "14d10h 05m"      → 14-е число, 10:05 (пробел перед m)
      "29d41h42m"       → опечатка, пропуск
      "<21d02h24m"      → убрать '<', парсить как обычно
      "26d03mh 24"      → опечатка, попытка восстановить

    forward=True  (T0, Tmax): если day_val < base_day → следующий месяц.
    forward=False (вспышка): если day_val > base_day → предыдущий месяц.
    """
    if not time_str or not event_date_str:
        return None

    # Очистка
    time_str = time_str.strip().lstrip('<').strip()
    # Убрать пробелы внутри
    time_str = re.sub(r'\s+', '', time_str)

    try:
        dt_base = datetime.strptime(event_date_str, '%Y-%m-%d')
    except ValueError:
        return None

    year = dt_base.year
    month = dt_base.month
    base_day = dt_base.day

    # Формат "DDdHHhMMm" или "DDdHHh"
    m = re.match(r'(\d+)d(\d+)h(\d+)?m?', time_str)
    if m:
        day_val = int(m.group(1))
        hour_val = int(m.group(2))
        min_val = int(m.group(3)) if m.group(3) else 0

        # Валидация
        if hour_val > 23 or min_val > 59 or day_val > 31:
            return None

        next_m = (year, month + 1) if month < 12 else (year + 1, 1)
        prev_m = (year, month - 1) if month > 1 else (year - 1, 12)

        if forward:
            # T0 / Tmax: если day_val < дня события — перешли в следующий месяц
            try_pairs = [next_m, (year, month)] if day_val < base_day else [(year, month), next_m]
        else:
            # Вспышка: если day_val > дня события — вспышка была в предыдущем месяце
            try_pairs = [prev_m, (year, month)] if day_val > base_day else [(year, month), prev_m]

        for yr, mo in try_pairs:
            try:
                return datetime(yr, mo, day_val, hour_val, min_val)
            except ValueError:
                continue
        return None
    
    # Формат "HHh" или "HHhMMm" (без дня — тот же день)
    m = re.match(r'(\d+)h(\d+)?m?$', time_str)
    if m:
        hour_val = int(m.group(1))
        min_val = int(m.group(2)) if m.group(2) else 0
        if hour_val > 23:
            return None
        try:
            return dt_base.replace(hour=hour_val, minute=min_val)
        except ValueError:
            return None
    
    return None


def parse_coordinates(loc_str):
    if not loc_str:
        return None
    m = re.search(r'([NS]\d+[EW]\d+)', loc_str)
    return m.group(1) if m else None


def parse_cme_data(cme_str):
    """'2077/360/360' → (velocity, width, PA). Обрабатывает пробелы."""
    if not cme_str:
        return None, None, None
    cleaned = re.sub(r'\s+', '', cme_str.strip())
    parts = cleaned.split('/')
    if len(parts) >= 3:
        try:
            v = int(parts[0].lstrip('0') or '0')
            w = int(parts[1].lstrip('0') or '0')
            p = int(parts[2].lstrip('0') or '0')
            return v, w, p
        except ValueError:
            pass
    return None, None, None


def hours_between(dt1, dt2):
    if dt1 is None or dt2 is None:
        return None
    return round((dt2 - dt1).total_seconds() / 3600, 4)


# ═══════════════════════════════════════════════════════════════
#  Парсинг HTML-таблицы
# ═══════════════════════════════════════════════════════════════

def parse_sinp_table(html_content, cycle=None):
    """Парсинг HTML-таблицы каталога СИНП МГУ."""
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Найти таблицу данных (НЕ навигационную)
    # Таблица данных имеет класс MuiTable-root или находится внутри MuiTableContainer-root
    data_table = None
    
    # Способ 1: найти контейнер MuiTableContainer-root
    container = soup.find('div', class_=re.compile(r'MuiTableContainer'))
    if container:
        data_table = container.find('table')
    
    # Способ 2: найти таблицу с классом MuiTable-root
    if not data_table:
        for t in soup.find_all('table'):
            classes = t.get('class', [])
            if any('MuiTable' in c for c in classes):
                data_table = t
                break
    
    # Способ 3: найти таблицу с aria-label="sticky table"
    if not data_table:
        data_table = soup.find('table', attrs={'aria-label': 'sticky table'})
    
    # Способ 4: последняя таблица на странице (fallback)
    if not data_table:
        tables = soup.find_all('table')
        if tables:
            data_table = tables[-1]
    
    if not data_table:
        print("Ошибка: таблица данных не найдена в HTML")
        return []
    
    tbody = data_table.find('tbody')
    if not tbody:
        print("Ошибка: tbody не найден")
        return []
    
    rows = tbody.find_all('tr')
    print(f"Найдено строк в таблице: {len(rows)}")
    
    records = []
    
    for row_idx, row in enumerate(rows):
        cells = row.find_all('td')
        if len(cells) < 15:
            continue
        
        def cell_text(idx):
            if idx >= len(cells):
                return ''
            cell = cells[idx]
            btn = cell.find('button')
            if btn:
                span = btn.find('span', class_='MuiButton-label')
                return span.get_text(strip=True) if span else btn.get_text(strip=True)
            return cell.get_text(strip=True)
        
        # Колонка 0: Event Date — "2020.11.30-335" или "2010.08.03-215"
        event_raw = cell_text(0)
        m_event = re.match(r'(\d{4})\.(\d{2})\.(\d{2})-(\d+)', event_raw)
        if not m_event:
            print(f"  Строка {row_idx}: не удалось распарсить '{event_raw}'")
            continue
        
        year = int(m_event.group(1))
        month = int(m_event.group(2))
        day = int(m_event.group(3))
        doy = int(m_event.group(4))
        
        event_date = f"{year}-{month:02d}-{day:02d}"
        event_doy = f"{year}-{doy:03d}"
        
        # Определение цикла по году если не задан явно
        event_cycle = cycle
        if event_cycle is None:
            if year <= 2019:
                event_cycle = 24
            else:
                event_cycle = 25
        
        # Колонка 1: Start — "01h20m", "10h", "18h00m"
        start_str = cell_text(1)
        to_dt = resolve_datetime_sinp(event_date, start_str)
        
        # Колонка 2: Tmax1 — "01d06h55m", "19h", "08d05h", "17d07"
        tmax_str = cell_text(2)
        tmax_dt = resolve_datetime_sinp(event_date, tmax_str)
        
        # Колонка 3: Jmax1
        jmax_str = cell_text(3)
        try:
            jmax = float(jmax_str.replace(',', '.')) if jmax_str else None
        except ValueError:
            jmax = None
        
        # Колонка 4: γ1
        gamma_str = cell_text(4)
        try:
            gamma = float(gamma_str.replace(',', '.')) if gamma_str else None
        except ValueError:
            gamma = None
        
        # Колонка 5: Eqm
        eqm_str = cell_text(5)
        try:
            eqm = float(eqm_str) if eqm_str else None
        except ValueError:
            eqm = None
        
        # Колонка 6: GLE
        gle_str = cell_text(6)
        is_gle = bool(gle_str and 'GLE' in gle_str.upper())
        gle_num = None
        if is_gle:
            m_gle = re.search(r'(\d+)', gle_str)
            gle_num = int(m_gle.group(1)) if m_gle else None
        
        # Колонка 7: Source type
        source_type = cell_text(7)
        
        # Колонка 8: Confidence
        confidence_str = cell_text(8)
        try:
            confidence = int(confidence_str) if confidence_str else None
        except ValueError:
            confidence = None
        
        # Колонка 9: T0 FL
        flare_time_raw = cell_text(9)
        flare_times = re.findall(r'<?(\d+d\d+h\s*\d*m?)', flare_time_raw)
        flare_dt = None
        if flare_times:
            flare_dt = resolve_datetime_sinp(event_date, flare_times[0], forward=False)
        
        # Колонка 10: Importance
        importance_raw = cell_text(10)
        xray_class, flare_power = parse_xray_class(importance_raw)
        
        optical_class = None
        if importance_raw:
            m_opt = re.search(r'/(\d?[A-Z]{1,3})', importance_raw.split(';')[0])
            if m_opt:
                optical_class = m_opt.group(1)
        
        # Колонка 11: Localization
        loc_raw = cell_text(11)
        coordinates = parse_coordinates(loc_raw)
        
        # Колонка 12: T0 CME
        cme_time_raw = cell_text(12)
        
        # Колонка 13: CME data
        cme_data_raw = cell_text(13)
        cme_velocity, cme_width, cme_pa = parse_cme_data(cme_data_raw)
        
        # Колонка 14: AR
        ar_raw = cell_text(14)
        m_ar = re.search(r'(\d{4,5})', ar_raw)
        active_region = m_ar.group(1) if m_ar else None
        
        # Вычисления
        t_delta_spe = hours_between(to_dt, tmax_dt)
        t_delta_flare = hours_between(flare_dt, to_dt)
        
        record = {
            'Event_date': event_date,
            'Event_DOY': event_doy,
            'GLE': is_gle,
            'GLE_num': gle_num,
            'Max_N': 1,
            'N_maxima': 1,
            'To': to_dt.strftime('%Y-%m-%d %H:%M:%S') if to_dt else '',
            'Tmax_parsed': tmax_dt.strftime('%Y-%m-%d %H:%M:%S') if tmax_dt else '',
            'Jmax_parsed': jmax,
            'T_delta_SPE': t_delta_spe,
            'T_delta_flare': t_delta_flare,
            'Gamma': gamma,
            'Eqm_MeV': eqm,
            'Duration_days': None,
            'Flare_Xray_class': xray_class,
            'Flare_power': flare_power,
            'Flare_optical': optical_class,
            'Flare_coordinates': coordinates,
            'Active_region': active_region,
            'Source_type': source_type,
            'Source_confidence': confidence,
            'Flare_time_raw': flare_time_raw,
            'CME_velocity_km_s': cme_velocity,
            'CME_width_deg': cme_width,
            'CME_PA_deg': cme_pa,
            'Cycle': event_cycle,
        }
        
        records.append(record)
    
    return records


# ═══════════════════════════════════════════════════════════════
#  Главная функция
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Парсер каталога СПС (СИНП МГУ) — циклы 24 и 25'
    )
    parser.add_argument('--html', default=None,
                        help='Путь к сохранённому HTML-файлу')
    parser.add_argument('--output', default='spe_catalog_sinp.csv',
                        help='Выходной CSV-файл')
    parser.add_argument('--cycle', type=int, default=None, choices=[24, 25],
                        help='Номер цикла (если не указан — определяется по году)')
    parser.add_argument('--compat', default=None,
                        help='Дополнительный совместимый CSV')
    args = parser.parse_args()
    
    if args.html:
        print(f"Чтение из файла: {args.html}")
        with open(args.html, 'r', encoding='utf-8') as f:
            html_content = f.read()
    else:
        try:
            import requests
        except ImportError:
            print("Ошибка: pip install requests (или используйте --html)")
            sys.exit(1)
        
        url = "https://swx.sinp.msu.ru/apps/sep_events_cat/index.php?gcm=1&lang=ru"
        print(f"Скачиваю: {url}")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        html_content = resp.text
        
        with open('sinp_catalog_cached.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"HTML сохранён: sinp_catalog_cached.html")
    
    records = parse_sinp_table(html_content, cycle=args.cycle)
    
    if not records:
        print("Не удалось извлечь данные!")
        sys.exit(1)
    
    # Статистика
    n_gle = sum(1 for r in records if r['GLE'])
    n_flare = sum(1 for r in records if r['Flare_Xray_class'])
    n_cme = sum(1 for r in records if r['CME_velocity_km_s'])
    cycles = set(r['Cycle'] for r in records)
    
    print(f"\n{'═' * 55}")
    print(f"  Результаты парсинга")
    print(f"{'═' * 55}")
    print(f"  Событий:       {len(records)}")
    print(f"  Циклы:         {sorted(cycles)}")
    print(f"  GLE:           {n_gle}")
    print(f"  С вспышкой:    {n_flare}")
    print(f"  С CME:         {n_cme}")
    
    # Заполненность
    print(f"\n  Заполненность полей:")
    for field in ['Jmax_parsed', 'Eqm_MeV', 'Flare_Xray_class',
                  'CME_velocity_km_s', 'T_delta_flare', 'T_delta_SPE']:
        filled = sum(1 for r in records if r.get(field) not in (None, '', 'None'))
        pct = 100 * filled / len(records)
        print(f"    {field:<25s}: {filled:>3d}/{len(records)} ({pct:.0f}%)")
    
    # Запись CSV
    columns = [
        'Event_date', 'Event_DOY', 'GLE', 'GLE_num', 'Max_N', 'N_maxima',
        'To', 'Tmax_parsed', 'Jmax_parsed', 'T_delta_SPE', 'T_delta_flare',
        'Gamma', 'Eqm_MeV', 'Duration_days',
        'Flare_Xray_class', 'Flare_power', 'Flare_optical',
        'Flare_coordinates', 'Active_region',
        'Source_type', 'Source_confidence', 'Flare_time_raw',
        'CME_velocity_km_s', 'CME_width_deg', 'CME_PA_deg',
        'Cycle',
    ]
    
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    
    print(f"\n  CSV: {args.output}")
    
    if args.compat:
        compat_cols = ['Event_date', 'Tmax_parsed', 'Jmax_parsed',
                       'Flare_power', 'T_delta_flare', 'T_delta_SPE', 'Cycle']
        with open(args.compat, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=compat_cols, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(records)
        print(f"  Compat CSV: {args.compat}")
    
    # Примеры
    print(f"\nПервые 5 записей:")
    for r in records[:5]:
        print(f"  {r['Event_date']}: Jmax={r['Jmax_parsed']}, "
              f"Class={r['Flare_Xray_class']}, CME_V={r['CME_velocity_km_s']}, "
              f"Eqm={r['Eqm_MeV']}, Cycle={r['Cycle']}")


if __name__ == '__main__':
    main()