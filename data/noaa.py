#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════
  Обогащение каталога СПС данными NOAA о вспышках
  Enriches SPE catalog with accurate flare onset times from NOAA
═══════════════════════════════════════════════════════════════════

Для каждого события запрашивает файл NOAA Solar Event Reports
за дату вспышки, находит соответствующую XRA-запись по номеру AR
и/или рентгеновскому классу, и добавляет точные времена
Begin / Max / End вспышки.

Использование:
    python enrich_with_noaa.py spe_catalog_23cycle_full.csv

    # С задержкой между запросами (по умолчанию 1 сек):
    python enrich_with_noaa.py spe_catalog_23cycle_full.csv --delay 0.5

    # Кеширование скачанных файлов:
    python enrich_with_noaa.py spe_catalog_23cycle_full.csv --cache-dir ./noaa_cache

Зависимости:
    pip install requests

Источники данных (пробуются по порядку):
    1. solarmonitor.org  — быстрый доступ
    2. NCEI NOAA archive — официальный архив (1996–present)
"""

import csv
import re
import os
import sys
import time
import json
from datetime import datetime, timedelta
from collections import defaultdict

try:
    import requests
except ImportError:
    print("Ошибка: библиотека requests не установлена.")
    print("Установите её командой:  pip install requests")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════
#  Конфигурация источников данных
# ═══════════════════════════════════════════════════════════════

SOURCES = [
    # solarmonitor.org
    "https://solarmonitor.org/data/{year}/{month}/{day}/meta/noaa_events_raw_{year}{month}{day}.txt",
    # NCEI NOAA official archive
    "https://www.ngdc.noaa.gov/stp/space-weather/swpc-products/daily_reports/solar_event_reports/{year}/{year}{month}{day}events.txt",
]

HEADERS = {
    'User-Agent': 'SPE-Catalog-Research/1.0 (solar proton event analysis)'
}


# ═══════════════════════════════════════════════════════════════
#  Загрузка NOAA Event Reports
# ═══════════════════════════════════════════════════════════════

def fetch_noaa_events(date_str, cache_dir=None, delay=1.0):
    """Скачать файл NOAA events для указанной даты (YYYY-MM-DD).
    Возвращает текст файла или None."""

    dt = datetime.strptime(date_str, '%Y-%m-%d')
    year = dt.strftime('%Y')
    month = dt.strftime('%m')
    day = dt.strftime('%d')
    datekey = f"{year}{month}{day}"

    # Проверка кеша
    if cache_dir:
        cache_file = os.path.join(cache_dir, f"{datekey}events.txt")
        if os.path.isfile(cache_file):
            with open(cache_file, 'r', encoding='utf-8', errors='replace') as f:
                return f.read()

    # Попробовать каждый источник
    for url_template in SOURCES:
        url = url_template.format(year=year, month=month, day=day)
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code == 200:
                text = resp.text
                # Проверка что это действительно NOAA events файл
                if 'Event' in text and ('XRA' in text or 'FLA' in text or 'No events' in text.lower() or '#' in text):
                    # Сохранить в кеш
                    if cache_dir:
                        os.makedirs(cache_dir, exist_ok=True)
                        with open(cache_file, 'w', encoding='utf-8') as f:
                            f.write(text)
                    time.sleep(delay)
                    return text
        except requests.RequestException:
            continue

    time.sleep(delay)
    return None


# ═══════════════════════════════════════════════════════════════
#  Парсинг NOAA Event Reports
# ═══════════════════════════════════════════════════════════════

def parse_noaa_events(text, date_str):
    """Парсинг текста NOAA events файла.
    Возвращает список словарей с XRA-событиями.

    Формат строки:
     420 +     0056   0104      0116  G18  5   XRA  1-8A      C1.1    1.3E-03   4389
    Колонки (фиксированная ширина, но проще парсить по паттерну):
      Event#  Begin  Max  End  Obs  Q  Type  Loc/Frq  Particulars  Reg#
    """
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    events = []

    for line in text.split('\n'):
        line = line.rstrip()
        if not line or line.startswith('#') or line.startswith(':'):
            continue

        # XRA-события (рентгеновские вспышки)
        # Паттерн: число, возможно +, затем 3 времени (HHMM или ////), обсерватория, качество, тип
        m = re.match(
            r'\s*(\d+)\s*\+?\s+'         # Event number (+ = continuation)
            r'(\d{4}|/{4})\s+'            # Begin time
            r'(\d{4}|/{4})\s+'            # Max time
            r'(\d{4}|/{4})\s+'            # End time
            r'(\w+)\s+'                   # Observatory
            r'(\w)\s+'                    # Quality
            r'(XRA|FLA)\s+'               # Type — нас интересуют XRA и FLA
            r'(.+)',                       # Rest: Loc/Frq + Particulars + Reg#
            line
        )
        if not m:
            continue

        evt_type = m.group(7)
        rest = m.group(8).strip()

        evt = {
            'event_num': m.group(1),
            'begin':     m.group(2),
            'max':       m.group(3),
            'end':       m.group(4),
            'obs':       m.group(5),
            'quality':   m.group(6),
            'type':      evt_type,
            'date':      date_str,
        }

        if evt_type == 'XRA':
            # Rest: "1-8A      C1.1    1.3E-03   4389"
            # или:  "1-8A      X2.1    2.2E-01   9393"
            m_xra = re.match(
                r'1-8A\s+([ABCMX]\d+\.?\d*)\s+'     # X-ray class
                r'([\d.eE+-]+)\s*'                    # Integrated flux
                r'(\d+)?\s*$',                        # AR number (может отсутствовать)
                rest
            )
            if m_xra:
                evt['xray_class'] = m_xra.group(1)
                evt['integrated_flux'] = m_xra.group(2)
                evt['ar_number'] = m_xra.group(3)
            else:
                # Попытка более свободного парсинга
                m_xra2 = re.search(r'([ABCMX]\d+\.?\d*)', rest)
                evt['xray_class'] = m_xra2.group(1) if m_xra2 else None
                m_ar = re.search(r'(\d{4,5})\s*$', rest)
                evt['ar_number'] = m_ar.group(1) if m_ar else None
                evt['integrated_flux'] = None

        elif evt_type == 'FLA':
            # Rest: "S14W34    3B ERU  9393" или "N16W12    SF        9393"
            m_fla = re.match(
                r'([NS]\d+[EW]\d+)\s+'     # Координаты
                r'(\S+)\s*'                 # Оптический класс
                r'(?:\w+\s+)?'             # Тип (ERU, DSF, ...)
                r'(\d+)?\s*$',             # AR
                rest
            )
            if m_fla:
                evt['coordinates'] = m_fla.group(1)
                evt['optical_class'] = m_fla.group(2)
                evt['ar_number'] = m_fla.group(3)
            else:
                m_ar = re.search(r'(\d{4,5})\s*$', rest)
                evt['ar_number'] = m_ar.group(1) if m_ar else None
                evt['coordinates'] = None
                evt['optical_class'] = None

        # Конвертация времён в datetime
        for tfield in ['begin', 'max', 'end']:
            tval = evt[tfield]
            if tval and tval != '////':
                try:
                    h, m_val = int(tval[:2]), int(tval[2:])
                    evt[f'{tfield}_dt'] = dt.replace(hour=h, minute=m_val, second=0)
                except (ValueError, IndexError):
                    evt[f'{tfield}_dt'] = None
            else:
                evt[f'{tfield}_dt'] = None

        events.append(evt)

    return events


# ═══════════════════════════════════════════════════════════════
#  Матчинг вспышки с СПС
# ═══════════════════════════════════════════════════════════════

def normalize_ar(ar_str):
    """Нормализация AR: '9393' -> '9393', '10652' -> '10652', '08100' -> '8100'"""
    if not ar_str:
        return None
    ar_str = ar_str.strip().lstrip('0')
    return ar_str if ar_str else None


def normalize_xray(cls_str):
    """Нормализация: 'X2.1' -> ('X', 2.1), 'M1.4' -> ('M', 1.4)"""
    if not cls_str:
        return None, None
    m = re.match(r'([ABCMX])(\d+\.?\d*)', cls_str.strip())
    if m:
        return m.group(1), float(m.group(2))
    return None, None


def match_flare(spe_row, noaa_events):
    """Найти наиболее подходящее XRA-событие из NOAA для данного СПС.

    Стратегия матчинга (по приоритету):
    1. AR совпадает И X-ray класс совпадает — отличный матч
    2. X-ray класс совпадает (в пределах ±20%) И время близко — хороший матч
    3. AR совпадает И тип XRA — допустимый матч
    4. Самая сильная XRA-вспышка за день — fallback

    Возвращает лучший матч или None.
    """

    xra_events = [e for e in noaa_events if e['type'] == 'XRA']
    if not xra_events:
        return None

    spe_ar = normalize_ar(spe_row.get('Active_region', ''))
    spe_letter, spe_num = normalize_xray(spe_row.get('Flare_Xray_class', ''))

    candidates = []

    for evt in xra_events:
        score = 0
        evt_ar = normalize_ar(evt.get('ar_number', ''))
        evt_letter, evt_num = normalize_xray(evt.get('xray_class', ''))

        # AR совпадение
        ar_match = (spe_ar and evt_ar and spe_ar == evt_ar)
        if ar_match:
            score += 10

        # X-ray класс совпадение
        xray_match = False
        if spe_letter and evt_letter:
            if spe_letter == evt_letter:
                if spe_num and evt_num:
                    ratio = evt_num / spe_num if spe_num > 0 else 0
                    if 0.8 <= ratio <= 1.2:
                        xray_match = True
                        score += 20  # точное совпадение класса
                    elif 0.5 <= ratio <= 2.0:
                        score += 5   # приблизительное
                else:
                    xray_match = True
                    score += 15

        # Если ничего не совпало — пропускаем
        if score == 0:
            continue

        candidates.append((score, evt))

    if not candidates:
        # Fallback: если есть AR и нет совпадений по классу,
        # берём самую мощную вспышку этого AR
        for evt in xra_events:
            evt_ar = normalize_ar(evt.get('ar_number', ''))
            if spe_ar and evt_ar and spe_ar == evt_ar:
                candidates.append((5, evt))

    if not candidates:
        return None

    # Сортировка по score (убывание), затем по рентгеновскому классу (убывание)
    def sort_key(item):
        sc, evt = item
        letter, num = normalize_xray(evt.get('xray_class', ''))
        xray_rank = {'A': 1, 'B': 2, 'C': 3, 'M': 4, 'X': 5}.get(letter, 0)
        return (sc, xray_rank, num or 0)

    candidates.sort(key=sort_key, reverse=True)
    return candidates[0][1]


# ═══════════════════════════════════════════════════════════════
#  Определение даты вспышки из данных каталога
# ═══════════════════════════════════════════════════════════════

def get_flare_date(spe_row):
    """Определить дату вспышки для запроса NOAA.
    Используем поле To (onset СПС) и вычитаем T_delta_flare,
    либо берём дату события минус 1 день (вспышка могла быть накануне).

    Возвращает список дат (YYYY-MM-DD) для проверки.
    """
    dates_to_check = []

    event_date = spe_row.get('Event_date', '')
    if not event_date:
        return dates_to_check

    dt_event = datetime.strptime(event_date, '%Y-%m-%d')

    # Основная дата — день события
    dates_to_check.append(event_date)

    # Предыдущий день (вспышка могла быть накануне)
    prev_day = (dt_event - timedelta(days=1)).strftime('%Y-%m-%d')
    dates_to_check.append(prev_day)

    # Если T_delta_flare > 24ч, проверить ещё раньше
    t_delta = spe_row.get('T_delta_flare', '')
    if t_delta and t_delta not in ('', 'None'):
        try:
            delta_hours = float(t_delta)
            if delta_hours > 24:
                days_back = int(delta_hours / 24) + 1
                for d in range(2, days_back + 1):
                    extra_day = (dt_event - timedelta(days=d)).strftime('%Y-%m-%d')
                    if extra_day not in dates_to_check:
                        dates_to_check.append(extra_day)
        except ValueError:
            pass

    return dates_to_check


# ═══════════════════════════════════════════════════════════════
#  Вычисление обновлённых T_delta
# ═══════════════════════════════════════════════════════════════

def compute_t_delta(to_str, flare_begin_dt):
    """Вычислить T_delta_flare = (To - Flare_begin) в часах."""
    if not to_str or not flare_begin_dt:
        return None
    try:
        to_dt = datetime.strptime(to_str, '%Y-%m-%d %H:%M:%S')
        delta = (to_dt - flare_begin_dt).total_seconds() / 3600
        return round(delta, 4)
    except (ValueError, TypeError):
        return None


# ═══════════════════════════════════════════════════════════════
#  Главная функция
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Обогащение каталога СПС данными NOAA о вспышках'
    )
    parser.add_argument('input_csv', help='Путь к spe_catalog_23cycle_full.csv')
    parser.add_argument('--delay', type=float, default=1.0,
                        help='Задержка между запросами (сек, по умолчанию 1.0)')
    parser.add_argument('--cache-dir', default='./noaa_cache',
                        help='Папка для кеша (по умолчанию ./noaa_cache)')
    parser.add_argument('--output', default=None,
                        help='Выходной файл (по умолчанию: *_enriched.csv)')
    args = parser.parse_args()

    if not os.path.isfile(args.input_csv):
        print(f"Ошибка: файл не найден: {args.input_csv}")
        sys.exit(1)

    # Выходной файл
    if args.output:
        output_path = args.output
    else:
        base, ext = os.path.splitext(args.input_csv)
        output_path = f"{base}_enriched{ext}"

    # Загрузка входных данных
    with open(args.input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        input_fields = reader.fieldnames
        rows = list(reader)

    print(f"Загружено {len(rows)} записей из {args.input_csv}")
    print(f"Кеш: {args.cache_dir}")
    print(f"Задержка: {args.delay}с")
    print()

    # Новые поля
    new_fields = [
        'NOAA_Flare_Begin',       # Время начала вспышки (HHMM)
        'NOAA_Flare_Max',         # Время максимума (HHMM)
        'NOAA_Flare_End',         # Время конца (HHMM)
        'NOAA_Flare_Begin_dt',    # Полный datetime начала
        'NOAA_Flare_Xray_class',  # Класс из NOAA (для верификации)
        'NOAA_Flare_AR',          # AR из NOAA (для верификации)
        'NOAA_Flare_date',        # Дата файла NOAA откуда взято
        'T_delta_flare_onset',    # To(SPE) - Begin(flare) в часах
        'Match_score',            # Качество матчинга
    ]

    output_fields = input_fields + new_fields

    # Определить уникальные даты для загрузки
    # (группировка по Max_N=1, чтобы не дублировать запросы)
    unique_events = {}
    for row in rows:
        key = row['Event_date']
        if key not in unique_events:
            unique_events[key] = row

    print(f"Уникальных событий для обработки: {len(unique_events)}")

    # Кеш NOAA-событий по дате
    noaa_cache = {}  # date_str -> list of events

    def get_noaa_for_date(date_str):
        if date_str in noaa_cache:
            return noaa_cache[date_str]
        text = fetch_noaa_events(date_str, cache_dir=args.cache_dir, delay=args.delay)
        if text:
            events = parse_noaa_events(text, date_str)
            noaa_cache[date_str] = events
            return events
        noaa_cache[date_str] = []
        return []

    # Обработка каждого события
    matches = {}  # event_date -> best match dict
    stats = {'matched': 0, 'no_data': 0, 'no_match': 0}

    for i, (event_date, spe_row) in enumerate(sorted(unique_events.items())):
        print(f"  [{i+1}/{len(unique_events)}] {event_date} "
              f"(AR={spe_row.get('Active_region','?')}, "
              f"Class={spe_row.get('Flare_Xray_class','?')})...", end=' ')

        dates_to_check = get_flare_date(spe_row)
        best_match = None

        for check_date in dates_to_check:
            noaa_events = get_noaa_for_date(check_date)
            if noaa_events:
                match = match_flare(spe_row, noaa_events)
                if match:
                    match['_source_date'] = check_date
                    best_match = match
                    break

        if best_match:
            matches[event_date] = best_match
            stats['matched'] += 1
            print(f"✓ {best_match.get('xray_class','?')} "
                  f"AR{best_match.get('ar_number','?')} "
                  f"Begin={best_match.get('begin','?')}")
        elif not any(get_noaa_for_date(d) for d in dates_to_check):
            stats['no_data'] += 1
            print("— нет данных NOAA")
        else:
            stats['no_match'] += 1
            print("✗ не найдено совпадение")

    # Применить результаты ко всем строкам
    enriched_rows = []
    for row in rows:
        new_row = dict(row)
        event_date = row['Event_date']
        match = matches.get(event_date)

        if match:
            new_row['NOAA_Flare_Begin'] = match.get('begin', '')
            new_row['NOAA_Flare_Max'] = match.get('max', '')
            new_row['NOAA_Flare_End'] = match.get('end', '')
            begin_dt = match.get('begin_dt')
            new_row['NOAA_Flare_Begin_dt'] = (
                begin_dt.strftime('%Y-%m-%d %H:%M:%S') if begin_dt else ''
            )
            new_row['NOAA_Flare_Xray_class'] = match.get('xray_class', '')
            new_row['NOAA_Flare_AR'] = match.get('ar_number', '')
            new_row['NOAA_Flare_date'] = match.get('_source_date', '')
            t_delta_noaa = compute_t_delta(row.get('To', ''), begin_dt)
            new_row['T_delta_flare_onset'] = t_delta_noaa
            # Обновляем T_delta_flare уточнёнными данными NOAA (Begin вспышки),
            # если новое значение валидно (≥ 0). Отрицательное означало бы,
            # что Begin вспышки оказался позже To СПС — неверный матч.
            if t_delta_noaa is not None and t_delta_noaa >= 0:
                new_row['T_delta_flare'] = t_delta_noaa
            # Score: сравнение AR и класса
            score_parts = []
            if (normalize_ar(row.get('Active_region', '')) ==
                normalize_ar(match.get('ar_number', ''))):
                score_parts.append('AR_match')
            if row.get('Flare_Xray_class', '') == match.get('xray_class', ''):
                score_parts.append('Class_exact')
            new_row['Match_score'] = '+'.join(score_parts) if score_parts else 'weak'
        else:
            for f in new_fields:
                new_row[f] = ''

        enriched_rows.append(new_row)

    # Запись результата
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(enriched_rows)

    # Итоги
    print(f"\n{'═' * 55}")
    print(f"  Результаты обогащения")
    print(f"{'═' * 55}")
    print(f"  Найдено совпадений:    {stats['matched']}/{len(unique_events)}")
    print(f"  Нет данных NOAA:       {stats['no_data']}")
    print(f"  Нет совпадения:        {stats['no_match']}")
    print(f"  Файл сохранён:         {output_path}")
    print()

    # Показать примеры T_delta сравнения
    print("Примеры (старый T_delta_flare vs новый T_delta_flare_onset):")
    shown = 0
    for row in enriched_rows:
        if row['T_delta_flare_onset'] and row['T_delta_flare']:
            old = row['T_delta_flare']
            new = row['T_delta_flare_onset']
            if old not in ('', 'None') and new not in ('', 'None'):
                print(f"  {row['Event_date']}: "
                      f"old={float(old):.2f}h (от макс), "
                      f"new={float(new):.2f}h (от onset), "
                      f"Δ={float(old)-float(new):.2f}h")
                shown += 1
                if shown >= 10:
                    break


if __name__ == '__main__':
    main()