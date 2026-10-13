"""Registro de modelos. Cada fabrica recebe input_dim, num_classes e kwargs do config."""

import torch.nn as nn


def make_mlp(input_dim, num_classes, hidden=1000):
    """input_dim -> hidden (ReLU) -> num_classes. Softmax fica na CrossEntropyLoss.
    Com hidden=1000 e o MLP de An et al. (2018) e Dai et al. (2023)."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, num_classes),
    )


REGISTRY = {
    'mlp': make_mlp,
}


def build_model(name, input_dim, num_classes, **kwargs):
    key = name.lower()
    if key not in REGISTRY:
        raise ValueError(f"Modelo desconhecido '{name}'. Opcoes: {sorted(REGISTRY)}")
    return REGISTRY[key](input_dim, num_classes, **kwargs)
