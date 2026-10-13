"""Maior autovalor da Hessiana da perda (curvatura maxima), por iteracao de potencia.

Usa produtos Hessiana-vetor (H v) calculados pelo autograd, sem nunca montar
a Hessiana (o MLP 784-1000-10 tem ~795 mil parametros: a Hessiana teria
~6e11 entradas).

O valor obtido e a curvatura da direcao mais "ingreme" naquele ponto. E o
'a' que, no modelo linear (optctrl.analysis.linear), decide primeiro a
estabilidade. Atencao: e medido num ponto (ex.: na inicializacao) e muda
ao longo do treino; serve como previsao de primeira ordem.
"""

import math

import torch
from torch import nn


def top_hessian_eigenvalue(model, X, Y, criterion=None, iters=100, tol=1e-4, seed=0):
    """Retorna (lambda_max, n_iteracoes). Converge para o autovalor de maior
    modulo; o sinal vem do quociente de Rayleigh v^T H v."""
    criterion = criterion or nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    loss = criterion(model(X), Y)
    grads = torch.autograd.grad(loss, params, create_graph=True)

    gen = torch.Generator().manual_seed(seed)
    v = [torch.randn(p.shape, generator=gen, dtype=p.dtype).to(p.device) for p in params]
    norm = math.sqrt(sum(float((x * x).sum()) for x in v))
    v = [x / norm for x in v]

    lam_prev, lam = None, 0.0
    for k in range(1, iters + 1):
        Hv = torch.autograd.grad(grads, params, grad_outputs=v, retain_graph=True)
        lam = sum(float((hv * vv).sum()) for hv, vv in zip(Hv, v))     # Rayleigh
        norm = math.sqrt(sum(float((hv * hv).sum()) for hv in Hv))
        if norm == 0:
            return 0.0, k
        v = [hv / norm for hv in Hv]
        if lam_prev is not None and abs(lam - lam_prev) <= tol * max(abs(lam), 1e-12):
            return lam, k
        lam_prev = lam
    return lam, iters
