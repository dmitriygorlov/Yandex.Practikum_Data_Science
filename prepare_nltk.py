"""Explicit one-time preparation of public NLTK resources inside this checkout."""
from pathlib import Path

import nltk

if __name__ == '__main__':
    target = Path(__file__).parent / 'data' / 'nltk_data'
    target.mkdir(parents=True, exist_ok=True)
    for resource in ('stopwords', 'wordnet', 'punkt_tab', 'averaged_perceptron_tagger_eng'):
        if not nltk.download(resource, download_dir=str(target), raise_on_error=True):
            raise RuntimeError(f'Failed to prepare {resource}')
