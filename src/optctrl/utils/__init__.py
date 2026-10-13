"""Utilitarios de rastreabilidade: todo resultado salvo carrega o config,
o commit do git e as versoes de software que o geraram."""

import datetime as _dt
import json
import os
import platform
import subprocess

import torch


def git_info():
    """Hash do commit atual e se ha alteracoes nao commitadas."""
    def run(*args):
        return subprocess.check_output(['git', *args], stderr=subprocess.DEVNULL,
                                       text=True).strip()
    try:
        return {'commit': run('rev-parse', 'HEAD'),
                'dirty': bool(run('status', '--porcelain'))}
    except Exception:  # noqa: BLE001
        return {'commit': None, 'dirty': None}


def environment_info(device):
    info = {'python': platform.python_version(), 'torch': torch.__version__,
            'platform': platform.platform(), 'device': str(device)}
    if torch.cuda.is_available():
        info['gpu'] = torch.cuda.get_device_name(0)
    return info


def timestamp():
    return _dt.datetime.now().strftime('%Y%m%d-%H%M%S')


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
