"""Registro de otimizadores.

Para adicionar um otimizador novo: implemente a classe (ex.: em um arquivo
novo nesta pasta) e acrescente uma entrada em REGISTRY. Os configs YAML
referenciam o otimizador pelo nome da chave.
"""

import torch

from .pid import SGD_PID, Adam_PID, SGD_PID_Classic, SGD_PID_Paper

REGISTRY = {
    'sgd': torch.optim.SGD,              # use momentum=0.9 para SGD-Momentum
    'adam': torch.optim.Adam,
    'sgd_pid': SGD_PID,                  # forma separada (codigo tensorboy)
    'sgd_pid_paper': SGD_PID_Paper,      # Eq. (10) de An et al. (2018)
    'adam_pid': Adam_PID,                # Dai et al. (2023)
}


def build_optimizer(name, params, **hparams):
    """Instancia um otimizador do REGISTRY pelo nome."""
    key = name.lower()
    if key not in REGISTRY:
        raise ValueError(f"Otimizador desconhecido '{name}'. Opcoes: {sorted(REGISTRY)}")
    return REGISTRY[key](params, **hparams)


__all__ = ['REGISTRY', 'build_optimizer', 'SGD_PID', 'SGD_PID_Paper',
           'SGD_PID_Classic', 'Adam_PID']
