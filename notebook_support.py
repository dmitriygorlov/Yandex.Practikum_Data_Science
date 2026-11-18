"""Portable notebook input and Russian morphology, without shell installation."""
import os
import re
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

DATA_DIR = Path(os.environ.get('DATA_DIR', Path(__file__).parent / 'data')).resolve()


def read_dataset(url, **kwargs):
    """Prefer explicitly supplied local data; never silently download a dataset."""
    local = DATA_DIR / Path(urlparse(url).path).name
    if local.is_file():
        return pd.read_csv(local, **kwargs)
    if os.environ.get('ALLOW_DATA_DOWNLOAD') == '1':
        return pd.read_csv(url, **kwargs)
    raise FileNotFoundError(f'Place {local.name} in {DATA_DIR}, or explicitly set ALLOW_DATA_DOWNLOAD=1. Source: {url}')


class RussianLemmatizer:
    """Dictionary-based token lemmas; not bit-identical to historical Mystem."""
    def __init__(self):
        from pymorphy3 import MorphAnalyzer
        self.analyzer = MorphAnalyzer()

    @lru_cache(maxsize=16384)
    def _lemma(self, token):
        return self.analyzer.parse(token)[0].normal_form

    def lemmatize(self, text):
        return [self._lemma(token) for token in re.findall(r'[а-яё]+', str(text).lower())]
