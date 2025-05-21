import pandas as pd
from bs4 import BeautifulSoup
import re

# Чтение HTML-кода из файла
with open('data/25 цикл.txt', 'r', encoding='utf-8') as file:
    html_content = file.read()

# Создание объекта BeautifulSoup для парсинга HTML
soup = BeautifulSoup(html_content, 'html.parser')

# Извлечение заголовков таблицы
headers = []
header_rows = soup.select('thead tr')

# Обработка сложной структуры заголовков с colspan и rowspan
all_headers = []
for row in header_rows:
    row_headers = []
    for cell in row.select('th'):
        text = cell.text.strip()
        colspan = int(cell.get('colspan', 1))
        rowspan = int(cell.get('rowspan', 1))
        
        # Добавление заголовка с учетом colspan
        for _ in range(colspan):
            row_headers.append({
                'text': text,
                'rowspan': rowspan
            })
    all_headers.append(row_headers)

# Формирование окончательных заголовков с учетом rowspan
max_cols = max(len(row) for row in all_headers)
final_headers = [''] * max_cols

for col_idx in range(max_cols):
    header_texts = []
    for row_idx, row in enumerate(all_headers):
        if col_idx < len(row) and row[col_idx]['rowspan'] > row_idx:
            header_texts.append(row[col_idx]['text'])
    
    final_headers[col_idx] = ' '.join(header_texts).strip()

# Извлечение данных из строк таблицы
rows_data = []
for row in soup.select('tbody tr'):
    row_data = []
    for cell in row.select('td'):
        # Извлечение текста из ячейки, включая текст из вложенных элементов
        cell_text = cell.get_text(strip=True)
        row_data.append(cell_text)
    
    rows_data.append(row_data)

# Создание DataFrame из данных
df = pd.DataFrame(rows_data, columns=final_headers)

# Сохранение в Excel-файл
df.to_excel('solar_proton_events_table.xlsx', index=False)

print("Таблица успешно сохранена в файл 'solar_proton_events_table.xlsx'")
