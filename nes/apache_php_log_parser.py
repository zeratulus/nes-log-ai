import re
import os
import hashlib
import json

def parse_log_file(file_path):
    """
    Парсить лог-файл, групує помилки та збирає статистику.

    Args:
        file_path (str): Шлях до лог-файлу.

    Returns:
        dict: Словник з агрегованою інформацією про помилки.
    """
    # Регулярний вираз для виявлення рядка з новою помилкою
    # Він захоплює дату, тип помилки та основне повідомлення.
    log_entry_regex = re.compile(
        r'\[(.*?)\]\s+'  # 1. Часова мітка (напр. 29-Jun-2025 18:35:49 UTC)
        r'(PHP (?:Fatal error|Warning|Notice|Parse error|Core error|Core warning|Compile error|Compile warning|User error|User warning|User notice|Strict Standards|Deprecated|User deprecated)):\s+'  # 2. Тип помилки (напр. PHP Fatal error)
        r'(.*)',  # 3. Повідомлення про помилку
        re.DOTALL
    )

    aggregated_errors = {}
    current_entry_lines = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Перевіряємо, чи є поточний рядок початком нового запису в лозі
                if log_entry_regex.match(line) and current_entry_lines:
                    # Якщо так, обробляємо попередній запис
                    process_log_entry(current_entry_lines, log_entry_regex, aggregated_errors)
                    current_entry_lines = []

                current_entry_lines.append(line)

            # Обробляємо останній запис у файлі
            if current_entry_lines:
                process_log_entry(current_entry_lines, log_entry_regex, aggregated_errors)

    except FileNotFoundError:
        print(f"Помилка: Файл не знайдено за шляхом '{file_path}'")
        return None
    except Exception as e:
        print(f"Виникла несподівана помилка: {e}")
        return None

    return aggregated_errors


def process_log_entry(lines, regex, aggregated_errors):
    """
    Обробляє окремий запис логу (який може бути багаторядковим).

    Args:
        lines (list): Список рядків, що належать до одного запису.
        regex (re.Pattern): Скомпільований регулярний вираз для парсингу.
        aggregated_errors (dict): Словник для зберігання результатів.
    """
    full_entry_text = "".join(lines)
    match = regex.match(full_entry_text)

    if not match:
        return

    timestamp = match.group(1).strip()
    error_type = match.group(2).strip()
    # Повідомлення та Stack Trace
    message_and_trace = match.group(3).strip()

    # Розділяємо основне повідомлення та stack trace
    if 'Stack trace:' in message_and_trace:
        parts = message_and_trace.split('Stack trace:', 1)
        error_message = parts[0].strip()
        stack_trace = 'Stack trace:' + parts[1].strip()
    else:
        error_message = message_and_trace
        stack_trace = None

    # Створюємо унікальний ключ для групування.
    # Помилки вважаються однаковими, якщо збігається їхній тип, повідомлення та stack trace.
    # Шляхи до файлів у stack trace можуть містити динамічні частини, тому ми їх нормалізуємо.
    normalized_trace = re.sub(r'#\d+\s+.*?\(\d+\)', '#N file(line)', stack_trace) if stack_trace else ''
    unique_key_string = f"{error_type}|{error_message}|{normalized_trace}"

    # Використовуємо хеш як ключ для ефективності
    unique_key = hashlib.md5(unique_key_string.encode('utf-8')).hexdigest()

    if unique_key not in aggregated_errors:
        aggregated_errors[unique_key] = {
            'type': error_type,
            'message': error_message,
            'count': 0,
            'timestamps': [],
            'stack_trace': stack_trace
        }

    # Оновлюємо статистику
    aggregated_errors[unique_key]['count'] += 1
    aggregated_errors[unique_key]['timestamps'].append(timestamp)


def print_summary(aggregated_errors, language_iso2: str = 'en'):
    """
    Виводить звіт про проаналізовані помилки у читабельному форматі.

    Args:
        :param aggregated_errors: Словник з агрегованою інформацією.
        :param language_iso2: мова, на котрій буде вивод ф-ції
    """
    if not aggregated_errors:
        print("Не знайдено жодних помилок для аналізу.")
        return

    print("=" * 80)
    print("Аналіз лог-файлу завершено. Результати:")
    print("=" * 80)

    # Сортуємо помилки за кількістю повторень (від найбільшої до найменшої)
    sorted_errors = sorted(aggregated_errors.values(), key=lambda x: x['count'], reverse=True)

    total_unique_errors = len(sorted_errors)
    total_errors = sum(item['count'] for item in sorted_errors)

    print(f"Всього знайдено помилок: {total_errors}")
    print(f"Кількість унікальних типів помилок: {total_unique_errors}\n")

    for i, error_data in enumerate(sorted_errors, 1):
        print("-" * 80)
        print(format_error_item_to_str(i, error_data, language_iso2))

def format_error_item_to_str(index, error_data, language):
    with open(f"{os.environ.get('DIR_ROOT')}nes/i18n/{language}/format-error-item-to-str.json") as f:
        translations = json.load(f)

    result = f"{index} | {translations['text_error_type']}: {error_data['type']}\n"
    result += f"{translations['text_count']}: {error_data['count']}\n"
    result += f"{translations['text_message']}: {error_data['message']}\n"
    if error_data['timestamps']:
        result += f"{translations['text_first_timestamp']}: {error_data['timestamps'][0]}\n"
        result += f"{translations['text_last_timestamp']}: {error_data['timestamps'][-1]}\n"

    if error_data['stack_trace']:
        result += "<StackTrace>\n"
        result += error_data['stack_trace']
        result += "\n</StackTrace>\n"

    return result

def save_json_file(content, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(content, indent=4))