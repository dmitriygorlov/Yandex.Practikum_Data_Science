"""Validate notebook syntax or execute into a separate artifact, never overwrite history."""
import argparse
import ast
from pathlib import Path

import nbformat
from IPython.core.inputtransformer2 import TransformerManager
from nbclient import NotebookClient


def check(root):
    paths = sorted(root.rglob('*.ipynb'))
    paths = [p for p in paths if not any(x in p.parts for x in ('.venv', 'artifacts', '.ipynb_checkpoints'))]
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == 'code':
                ast.parse(TransformerManager().transform_cell(cell.source), filename=f'{path}:cell{index}')
    print(f'Validated {len(paths)} notebooks')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('notebook', nargs='?')
    parser.add_argument('--timeout', type=int, default=300)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if not args.notebook:
        check(root)
    else:
        path = (root / args.notebook).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Notebook must be inside the repository')
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(notebook, timeout=args.timeout, kernel_name='python3',
                                resources={'metadata': {'path': str(root)}})
        output = root / 'artifacts' / path.name
        output.parent.mkdir(exist_ok=True)
        try:
            client.execute()
        finally:
            nbformat.write(notebook, output)
        print(f'Executed {path.name}; fresh output: {output}')
