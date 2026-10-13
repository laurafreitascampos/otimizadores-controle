"""Permite importar scripts/train.py nos testes como modulo 'scripts_train'."""

import importlib.util
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location('scripts_train',
                                               os.path.join(_ROOT, 'scripts', 'train.py'))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
sys.modules['scripts_train'] = _mod
