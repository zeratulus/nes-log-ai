import re
import os
import hashlib
import json
from nes.i18n.language import Language

def parse_log_file(file_path):
    """
    Parses the log file, groups errors, and collects statistics.

    Args:
        file_path (str): Path to log file.

    Returns:
        dict: Dictionary with aggregated error information.
    """
    # A regular expression to detect a line with a new error. It captures the date, error type, and main message.
    log_entry_regex = re.compile(
        r'\[(.*?)\]\s+'  # 1. Timestamp
        r'(PHP (?:Fatal error|Warning|Notice|Parse error|Core error|Core warning|Compile error|Compile warning|User error|User warning|User notice|Strict Standards|Deprecated|User deprecated)):\s+'  # 2. Error type (PHP Fatal error)
        r'(.*)',  # Error message
        re.DOTALL
    )

    aggregated_errors = {}
    current_entry_lines = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Checking whether the current line is the beginning of a new record in the vine
                if log_entry_regex.match(line) and current_entry_lines:
                    # If so, we process the previous record
                    process_log_entry(current_entry_lines, log_entry_regex, aggregated_errors)
                    current_entry_lines = []

                current_entry_lines.append(line)

            # Processing the last record in the file
            if current_entry_lines:
                process_log_entry(current_entry_lines, log_entry_regex, aggregated_errors)

    except FileNotFoundError:
        print(f"Error: File not found at path '{file_path}'")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

    return aggregated_errors


def process_log_entry(lines, regex, aggregated_errors):
    """
    Processes a single log entry (which may be multi-line).

    Args:
        lines (list): List of rows belonging to a single record.
        regex (re.Pattern): Compiled regular expression for parsing.
        aggregated_errors (dict): Dictionary for storing results.
    """
    full_entry_text = "".join(lines)
    match = regex.match(full_entry_text)

    if not match:
        return

    timestamp = match.group(1).strip()
    error_type = match.group(2).strip()
    # Message and Stack Trace
    message_and_trace = match.group(3).strip()

    # Split message and Stack Trace
    if 'Stack trace:' in message_and_trace:
        parts = message_and_trace.split('Stack trace:', 1)
        error_message = parts[0].strip()
        stack_trace = 'Stack trace:' + parts[1].strip()
    else:
        error_message = message_and_trace
        stack_trace = None

    # Create a unique grouping key.
    # Errors are considered the same if their type, message, and stack trace match.
    # File paths in the stack trace can contain dynamic parts, so we normalize them.
    normalized_trace = re.sub(r'#\d+\s+.*?\(\d+\)', '#N file(line)', stack_trace) if stack_trace else ''
    unique_key_string = f"{error_type}|{error_message}|{normalized_trace}"

    # We use hash as a key for efficiency
    unique_key = hashlib.md5(unique_key_string.encode('utf-8')).hexdigest()

    if unique_key not in aggregated_errors:
        aggregated_errors[unique_key] = {
            'type': error_type,
            'message': error_message,
            'count': 0,
            'timestamps': [],
            'stack_trace': stack_trace
        }

    # Update statistics
    aggregated_errors[unique_key]['count'] += 1
    aggregated_errors[unique_key]['timestamps'].append(timestamp)


def print_summary(aggregated_errors, language_iso2: str = 'en'):
    """
    Outputs a report of analyzed errors in a readable format.

    Args:
        :param aggregated_errors: Dictionary with aggregated information.
        :param language_iso2: the language ISO2 code in which the function output will be
    """
    translations = Language(language_iso2=language_iso2)
    translations.load('print-summary')
    if not aggregated_errors:
        print(translations.get('text_empty_errors'))
        return

    print("=" * 80)
    print(translations.get('text_anal_finished_results'))
    print("=" * 80)

    # Сортуємо помилки за кількістю повторень (від найбільшої до найменшої)
    sorted_errors = sorted(aggregated_errors.values(), key=lambda x: x['count'], reverse=True)

    total_unique_errors = len(sorted_errors)
    total_errors = sum(item['count'] for item in sorted_errors)

    print(f"{translations.get('text_errors_total')} {total_errors}")
    print(f"{translations.get('text_unique_errors_count')} {total_unique_errors}\n")

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