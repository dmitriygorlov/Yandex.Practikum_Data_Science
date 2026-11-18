"""Execute the entire text notebook on synthetic comments, not benchmark data."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

root = Path(__file__).parent.resolve()
with tempfile.TemporaryDirectory(prefix='text-smoke-') as temp:
    data = Path(temp)
    pd.DataFrame({'text': ['You are a rude fool', 'Thank you for your kind help',
                          'Awful stupid comment', 'Great clear explanation'] * 20,
                  'toxic': [1, 0, 1, 0] * 20}).to_csv(data / 'toxic_comments.csv', index=False)
    env = dict(os.environ, DATA_DIR=str(data), NLTK_DATA=str(root / 'data' / 'nltk_data'))
    notebook = next(root.glob('Module*/*text-analysis.ipynb'))
    print('SYNTHETIC text smoke; no benchmark metric claim', flush=True)
    subprocess.run([sys.executable, 'validate_notebooks.py', str(notebook.relative_to(root))], cwd=root, env=env, check=True)
