import os
import json
from typing import Dict

class Language:

    translations: Dict = {}

    def __init__(self, language_iso2: str = 'en'):
        self.language_code = language_iso2

    def load(self, language_file_name: str):
        path = f"{os.environ.get('DIR_ROOT')}/nes/i18n/{self.language_code}/{language_file_name}.json"

        if not os.path.isfile(path):
            raise FileNotFoundError(f"Class Language Error -> i18n file not found: {path}")

        with open(path) as f:
            self.translations.update(json.load(f))

    def get(self, key: str):
        if key in self.translations:
            return self.translations[key]
        else:
            return key

    def print_translations(self):
        return print(self.translations)
