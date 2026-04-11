#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════
  Парсер каталога солнечных протонных событий (23-й цикл СА)
  Parser for the Catalog of Solar Proton Events (23rd Cycle)
═══════════════════════════════════════════════════════════════════

Извлекает данные из PDF-каталога Логачёва и др. (2016) в CSV.
Каждый максимум в многопиковых событиях — отдельная строка,
связанная общим Event_ID и различающаяся полем Max_N.

Использование:
    python parse_spe_standalone.py Catalog_SPE_23_cycle_SA_en.pdf

    # Или указать выходную папку:
    python parse_spe_standalone.py Catalog_SPE_23_cycle_SA_en.pdf --outdir ./results

Зависимости:
    pip install pdfplumber

Выходные файлы:
    spe_catalog_23cycle_full.csv   — полная таблица (209 строк, 25 полей)
    spe_catalog_23cycle_compat.csv — совместимый формат (1 строка на событие)

Результаты валидации (на 66 пересекающихся событиях):
    Jmax:         97% совпадение
    Flare_power:  92% совпадение
    T_delta_flare: 83% совпадение
"""

import re
import csv
import sys
import os
from datetime import datetime
from collections import defaultdict

try:
    import pdfplumber
except ImportError:
    print("Ошибка: библиотека pdfplumber не установлена.")
    print("Установите её командой:  pip install pdfplumber")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════
#  Вспомогательные функции
# ═══════════════════════════════════════════════════════════════

def parse_xray_class(class_str):
    """Конвертация рентгеновского класса вспышки в пиковый поток (Вт/м²).
    A=1e-8, B=1e-7, C=1e-6, M=1e-5, X=1e-4"""
    if not class_str:
        return None
    class_str = class_str.strip()
    scale = {'A': 1e-8, 'B': 1e-7, 'C': 1e-6, 'M': 1e-5, 'X': 1e-4}
    letter = class_str[0].upper()
    if letter not in scale:
        return None
    try:
        num = float(class_str[1:])
        return scale[letter] * num
    except (ValueError, IndexError):
        return None


def resolve_datetime(event_date_str, day_hour_str, forward=True):
    """Преобразование '29d19h' в полный datetime, зная дату события '2001.03.29'.
    Число после 'd' — день месяца; обрабатывает переход через конец месяца.

    forward=True  (T0, Tmax): время не раньше события → если day_val < event_day, это следующий месяц.
    forward=False (вспышка): время до события → если day_val > event_day, это предыдущий месяц.
    """
    if not day_hour_str:
        return None

    parts = event_date_str.split('.')
    year, month, event_day = int(parts[0]), int(parts[1]), int(parts[2])

    m = re.match(r'(\d+)d\s*(\d+)h?\s*(?:(\d+)m)?', day_hour_str.strip())
    if not m:
        return None

    day_val = int(m.group(1))
    hour_val = int(m.group(2))
    min_val = int(m.group(3)) if m.group(3) else 0

    next_month = (year, month + 1) if month < 12 else (year + 1, 1)
    prev_month = (year, month - 1) if month > 1 else (year - 1, 12)

    if forward:
        # T0 / Tmax: если day_val < дня события — перешли в следующий месяц
        pairs = [next_month, (year, month)] if day_val < event_day else [(year, month), next_month]
    else:
        # Вспышка: если day_val > дня события — вспышка была в предыдущем месяце
        pairs = [prev_month, (year, month)] if day_val > event_day else [(year, month), prev_month]

    for yr, mo in pairs:
        try:
            return datetime(yr, mo, day_val, hour_val, min_val)
        except ValueError:
            continue
    return None


def parse_flare_class_from_source(source_line):
    """Извлечение рентгеновского класса из строки источника.
    Примеры: 'X2.1/3B', '2B/X9.4', 'M1.1/1F'"""
    if not source_line:
        return None
    m = re.search(r'([ABCMX]\d+\.?\d*)', source_line)
    return m.group(1) if m else None


def parse_coordinates(source_line):
    """Извлечение гелиографических координат: N16W12, S14W34"""
    if not source_line:
        return None
    m = re.search(r'([NS]\d+[EW]\d+)', source_line)
    return m.group(1) if m else None


def parse_active_region(source_line):
    """Извлечение номера активной области (AR)"""
    if not source_line:
        return None
    # А (кириллица) и A (латиница)
    m = re.search(r'[АA]R\s*(\d+)', source_line)
    return m.group(1) if m else None


def extract_flare_time(source_line):
    """Извлечение времени вспышки: '29d09h55m', '04d05h52m'"""
    if not source_line:
        return None
    m = re.search(
        r'(?:flare|event)\s*(?:<\s*)?(\d+d\s*\d+h\s*\d*m?)',
        source_line, re.IGNORECASE
    )
    return m.group(1).replace(' ', '') if m else None


def hours_between(dt1, dt2):
    """Разница в часах между двумя datetime"""
    if dt1 is None or dt2 is None:
        return None
    return round((dt2 - dt1).total_seconds() / 3600, 4)


# ═══════════════════════════════════════════════════════════════
#  Парсер одного события
# ═══════════════════════════════════════════════════════════════

def parse_event(page_num, text):
    """Парсинг текста легенды одного СПС в список словарей
    (по одному на каждый максимум)."""

    full_text = text

    # ── Заголовок события ──
    # Варианты: "Event:", "−"/"–"/"-", пробелы в DOY, суффикс-буква,
    # GLE-59, GLE 64, "и №" (рус.)
    m_header = re.search(
        r'Event:?\s+(\d{4}\.\d{2}\.\d{2})\w?\s*[–−-]\s*'
        r'\((\d{4}\s*-\s*\d{3}\w?)\)\s*'
        r'(?:[–−-]\s*GLE[\s-]*\d*\s*)?'
        r'(?:и\s*)?[№N][ºo°]?\s*(\d+)',
        full_text
    )
    if not m_header:
        m_header = re.search(
            r'Event:?\s+(\d{4}\.\d{2}\.\d{2})\w?\s*[–−-]\s*'
            r'\((\d{4}\s*-\s*\d{3}\w?)\)',
            full_text
        )

    if not m_header:
        return [], f"Cannot parse header: {full_text[:120]}"

    event_date_str = m_header.group(1)
    event_doy = m_header.group(2).replace(' ', '')
    event_num = m_header.group(3) if m_header.lastindex >= 3 else None
    is_gle = bool(re.search(r'GLE', full_text[:200]))

    # ── Onset To ──
    m_to = re.search(
        r'To\s*\(?Ep\s*>?\s*10\s*MeV\)?\s*[–−-]\s*(\d+d\s*\d+h)',
        full_text
    )
    to_str = m_to.group(1).replace(' ', '') if m_to else None
    to_dt = resolve_datetime(event_date_str, to_str)

    # ── Длительность ──
    m_dur = re.search(r'Duration.*?[–−-]\s*([\d.]+)\s*day', full_text)
    if m_dur:
        duration_days = float(m_dur.group(1))
    else:
        m_dur_h = re.search(r'Duration.*?[–−-]\s*([\d.]+)\s*hour', full_text)
        duration_days = round(float(m_dur_h.group(1)) / 24, 4) if m_dur_h else None

    # ── Максимумы (Tmax / Jmax) ──
    # Обрабатывает обычные и научную нотацию (·10³)
    maxima = []
    tmax_jmax_pattern = re.compile(
        r'Tmax\s*(?:\d\s*)?\(?\s*Ep\s*>?\s*10\s*MeV\)?\s*[–−-]\s*(\d+d\s*\d+h)\s*'
        r'[,;]\s*Jmax\s*(?:\d\s*)?\(?\s*Ep\s*>?\s*10\s*MeV\)?\s*[–−-]\s*'
        r'([\d.]+)(?:\s*[·∙×]\s*10(\d+))?\s*/?cm',
        re.IGNORECASE
    )
    for m in tmax_jmax_pattern.finditer(full_text):
        jval = float(m.group(2))
        if m.group(3):
            jval *= 10 ** int(m.group(3))
        maxima.append({'tmax_str': m.group(1).replace(' ', ''), 'jmax': jval})

    if not maxima:
        m_single = re.search(
            r'Tmax.*?[–−-]\s*(\d+d\s*\d+h).*?Jmax.*?[–−-]\s*'
            r'([\d.]+)(?:\s*[·∙×]\s*10(\d+))?\s*/?cm',
            full_text, re.IGNORECASE
        )
        if m_single:
            jval = float(m_single.group(2))
            if m_single.group(3):
                jval *= 10 ** int(m_single.group(3))
            maxima.append({
                'tmax_str': m_single.group(1).replace(' ', ''),
                'jmax': jval
            })

    # ── Квазимаксимальная энергия (Eqm) ──
    eqm_values = []
    for m_eqm in re.finditer(r'Eqm\s*(?:\d\s*)?[=≥≤>]\s*([\d.]+)\s*MeV', full_text):
        eqm_values.append(float(m_eqm.group(1)))

    # ── Источники: информация о вспышке ──
    source_section = ''
    m_src = re.search(
        r'(?:Sources?|Sun source)\s*:?\s*(.*?)(?:Particle fluxes|$)',
        full_text, re.DOTALL
    )
    if m_src:
        source_section = m_src.group(1)

    flare_line = ''
    for line in source_section.split('\n'):
        if re.search(r'solar flare|flare event', line, re.IGNORECASE):
            flare_line = line
            break

    xray_class = parse_flare_class_from_source(flare_line)
    flare_power = parse_xray_class(xray_class)
    coordinates = parse_coordinates(flare_line)
    active_region = parse_active_region(flare_line)

    # Оптический класс (3B, 1N, 2F, SF, EPL…)
    optical_class = None
    if flare_line:
        m_opt = re.search(r'[/,]\s*(\d?[A-Z]{1,3})\s*[,;]', flare_line)
        if m_opt:
            optical_class = m_opt.group(1)

    # ── Рентгеновский всплеск ──
    xray_onset_str = xray_max_str = None
    fluence = None

    m_xray = re.search(
        r'[Xx]-ray.*?onset\s*:?\s*[–−-]\s*(\d+d\s*\d+h\s*\d*m?)\s*[,;]\s*'
        r'max\s*[–−-]\s*(\d+d\s*\d+h\s*\d*m?)\s*[,;]\s*'
        r'(?:[ФΦJ](?:max)?\s*[=≈]\s*)([\d.eE+-]+)',
        full_text
    )
    if m_xray:
        xray_onset_str = m_xray.group(1).replace(' ', '')
        xray_max_str = m_xray.group(2).replace(' ', '')
        fluence = float(m_xray.group(3))
    else:
        m_xray2 = re.search(
            r'[Xx]-ray.*?(\d+d\s*\d+h\s*\d*m?)\s*[,;]\s*'
            r'max\s*[–−-]\s*(\d+d\s*\d+h\s*\d*m?)\s*[,;]\s*'
            r'(?:[ФΦJ](?:max)?\s*[=≈]\s*)([\d.eE+-]+)',
            full_text
        )
        if m_xray2:
            xray_onset_str = m_xray2.group(1).replace(' ', '')
            xray_max_str = m_xray2.group(2).replace(' ', '')
            fluence = float(m_xray2.group(3))

    xray_onset_dt = resolve_datetime(event_date_str, xray_onset_str, forward=False)

    # ── CME ──
    cme_velocity = cme_width = cme_pa = None

    m_cme = re.search(
        r'CME\s*:\s*(\d+d\s*\d+h\s*\d*m?)\s*[,;]\s*V\s*=?\s*(\d+)\s*km/s',
        full_text
    )
    if m_cme:
        cme_velocity = int(m_cme.group(2))

    m_cme_w = re.search(r'[Δ∆]φ\s*=?\s*(\d+)[ºo°]', full_text)
    if m_cme_w:
        cme_width = int(m_cme_w.group(1))

    m_cme_pa = re.search(r'dA\s*=?\s*(\d+)[ºo°˚]', full_text)
    if m_cme_pa:
        cme_pa = int(m_cme_pa.group(1))

    # ── SC (внезапное начало бури) ──
    sc_times = []
    for m_sc in re.finditer(r'[▲∆△]\s*(?:SC\s+)?(\d+d\s*\d+h\s*\d*m?)', full_text):
        sc_times.append(m_sc.group(1).replace(' ', ''))

    # ── Формирование строк (по одной на максимум) ──
    event_date_fmt = event_date_str.replace('.', '-')
    results = []

    for idx in range(max(len(maxima), 1)):
        row = {
            'Event_ID':    event_num,
            'Event_date':  event_date_fmt,
            'Event_DOY':   event_doy,
            'GLE':         is_gle,
            'Max_N':       idx + 1,
            'N_maxima':    len(maxima),
            'To':          to_dt.strftime('%Y-%m-%d %H:%M:%S') if to_dt else '',
        }

        if idx < len(maxima):
            tmax_dt = resolve_datetime(event_date_str, maxima[idx]['tmax_str'])
            row['Tmax_parsed'] = tmax_dt.strftime('%Y-%m-%d %H:%M:%S') if tmax_dt else ''
            row['Jmax_parsed'] = maxima[idx]['jmax']
            row['T_delta_SPE'] = hours_between(to_dt, tmax_dt)
        else:
            row['Tmax_parsed'] = ''
            row['Jmax_parsed'] = ''
            row['T_delta_SPE'] = None

        row['Eqm_MeV'] = (eqm_values[idx] if idx < len(eqm_values)
                          else (eqm_values[0] if eqm_values else None))
        row['Duration_days'] = duration_days

        row['Flare_Xray_class'] = xray_class
        row['Flare_power'] = flare_power
        row['Flare_optical'] = optical_class
        row['Flare_coordinates'] = coordinates
        row['Active_region'] = active_region
        row['Xray_fluence_J_m2'] = fluence
        row['T_delta_flare'] = hours_between(xray_onset_dt, to_dt)

        row['CME_velocity_km_s'] = cme_velocity
        row['CME_width_deg'] = cme_width
        row['CME_PA_deg'] = cme_pa
        row['SC_times'] = '; '.join(sc_times) if sc_times else ''
        row['Cycle'] = 23
        row['Page_in_PDF'] = page_num + 1

        results.append(row)

    return results, None


# ═══════════════════════════════════════════════════════════════
#  Главная функция
# ═══════════════════════════════════════════════════════════════

def main():
    # ── Аргументы командной строки ──
    if len(sys.argv) < 2:
        print(__doc__)
        print("Использование: python parse_spe_standalone.py <путь_к_PDF> [--outdir <папка>]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    outdir = '.'
    if '--outdir' in sys.argv:
        idx = sys.argv.index('--outdir')
        if idx + 1 < len(sys.argv):
            outdir = sys.argv[idx + 1]

    if not os.path.isfile(pdf_path):
        print(f"Ошибка: файл не найден: {pdf_path}")
        sys.exit(1)

    os.makedirs(outdir, exist_ok=True)

    # ── Шаг 1: Чтение PDF и поиск страниц с легендами событий ──
    print(f"Открываю PDF: {pdf_path}")
    pdf = pdfplumber.open(pdf_path)
    total_pages = len(pdf.pages)
    print(f"Всего страниц: {total_pages}")

    print("Ищу страницы с легендами событий...")
    event_data = []  # [(page_index, text), ...]

    # События начинаются примерно с 34-й страницы (после описания каталога)
    start_page = 30
    for i in range(start_page, total_pages):
        text = pdf.pages[i].extract_text()
        if text and text.strip().startswith("Event") and "Particle event" in text[:600]:
            event_data.append((i, text))

        # Прогресс
        if (i - start_page) % 100 == 0 and i > start_page:
            print(f"  ...просканировано {i - start_page}/{total_pages - start_page} страниц, "
                  f"найдено {len(event_data)} событий")

    pdf.close()
    print(f"Найдено страниц с легендами: {len(event_data)}")

    # ── Шаг 2: Парсинг каждого события ──
    print("Парсинг событий...")
    all_records = []
    errors = []

    for page_idx, text in event_data:
        result, error = parse_event(page_idx, text)
        if error:
            errors.append((page_idx + 1, error))
        if result:
            all_records.extend(result)

    n_events = len(set(r['Event_ID'] for r in all_records if r['Event_ID']))
    n_with_multi = sum(1 for r in all_records if int(r['N_maxima']) > 1 and r['Max_N'] == 1)
    n_gle = sum(1 for r in all_records if r['GLE'] and r['Max_N'] == 1)

    print(f"\n{'═' * 55}")
    print(f"  Результаты парсинга")
    print(f"{'═' * 55}")
    print(f"  Событий распознано:    {n_events}")
    print(f"  Строк в таблице:       {len(all_records)} (с учётом мультимаксимумов)")
    print(f"  Многопиковых событий:  {n_with_multi}")
    print(f"  GLE-событий:           {n_gle}")
    print(f"  Ошибок парсинга:       {len(errors)}")
    for pg, err in errors:
        print(f"    Стр. {pg}: {err[:80]}")

    # Статистика заполненности полей
    print(f"\n  Заполненность ключевых полей:")
    for field in ['Jmax_parsed', 'Eqm_MeV', 'Flare_Xray_class',
                  'CME_velocity_km_s', 'Xray_fluence_J_m2',
                  'T_delta_flare', 'T_delta_SPE']:
        filled = sum(1 for r in all_records
                     if r.get(field) not in (None, '', 'None'))
        pct = 100 * filled / len(all_records) if all_records else 0
        print(f"    {field:<25s}: {filled:>4d}/{len(all_records)} ({pct:.0f}%)")

    # ── Шаг 3: Запись полного CSV ──
    columns = [
        'Event_ID', 'Event_date', 'Event_DOY', 'GLE', 'Max_N', 'N_maxima',
        'To', 'Tmax_parsed', 'Jmax_parsed', 'T_delta_SPE', 'T_delta_flare',
        'Eqm_MeV', 'Duration_days',
        'Flare_Xray_class', 'Flare_power', 'Flare_optical',
        'Flare_coordinates', 'Active_region',
        'Xray_fluence_J_m2',
        'CME_velocity_km_s', 'CME_width_deg', 'CME_PA_deg',
        'SC_times', 'Cycle', 'Page_in_PDF'
    ]

    full_path = os.path.join(outdir, 'spe_catalog_24cycle_full.csv')
    with open(full_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(all_records)

    # ── Шаг 4: Запись совместимого CSV (1 строка на событие) ──
    events_grouped = defaultdict(list)
    for r in all_records:
        events_grouped[r['Event_date']].append(r)

    compat_cols = ['Event_date', 'Tmax_parsed', 'Jmax_parsed',
                   'Flare_power', 'T_delta_flare', 'T_delta_SPE', 'Cycle']
    compat_records = []
    for eid, maxima_list in sorted(events_grouped.items()):
        first_max = min(maxima_list, key=lambda x: int(x['Max_N']))
        compat_records.append({c: first_max.get(c, '') for c in compat_cols})

    compat_path = os.path.join(outdir, 'spe_catalog_24cycle_compat.csv')
    with open(compat_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=compat_cols)
        writer.writeheader()
        writer.writerows(compat_records)

    print(f"\n{'═' * 55}")
    print(f"  Файлы сохранены:")
    print(f"{'═' * 55}")
    print(f"  Полная таблица:     {full_path}")
    print(f"  Совместимая (1/ev): {compat_path}")
    print()

    # Первые 5 записей для проверки
    print("Первые 5 записей:")
    for r in all_records[:5]:
        print(f"  {r['Event_date']} Max{r['Max_N']}: "
              f"Jmax={r['Jmax_parsed']}, Eqm={r['Eqm_MeV']}, "
              f"Flare={r['Flare_Xray_class']}, CME_V={r['CME_velocity_km_s']}")


if __name__ == '__main__':
    main()