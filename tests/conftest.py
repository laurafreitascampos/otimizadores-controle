"""Permite importar os scripts de scripts/ nos testes como modulos 'scripts_<nome>'."""

import importlib.util
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for _name in ('train', 'sweep', 'compare_stats'):
    _path = os.path.join(_ROOT, 'scripts', f'{_name}.py')
    if not os.path.exists(_path):
        continue
    _spec = importlib.util.spec_from_file_location(f'scripts_{_name}', _path)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    sys.modules[f'scripts_{_name}'] = _mod
