"""Equivalencias exatas entre os otimizadores PID e os baselines do PyTorch.

Sao as afirmacoes das Secoes 3.2-3.3 de An et al. (SGD = controlador P,
SGD-Momentum = controlador PI) e da reducao Adam-PID -> Adam de Dai et al.,
transformadas em testes. Se alguma falhar, a implementacao nao e a lei de
aprendizado que diz ser.
"""

import copy

import pytest
import torch
import torch.nn as nn

from optctrl.optimizers import SGD_PID, Adam_PID, SGD_PID_Paper

STEPS = 25


def _problem(seed=0):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(6, 8), nn.Tanh(), nn.Linear(8, 3)).double()
    X = torch.randn(64, 6, dtype=torch.float64)
    y = torch.randint(0, 3, (64,))
    return model, X, y


def _train(model, opt, X, y, steps=STEPS):
    crit = nn.CrossEntropyLoss()
    for k in range(steps):
        idx = slice((k * 16) % 64, (k * 16) % 64 + 16)     # mini-lotes diferentes
        opt.zero_grad()
        crit(model(X[idx]), y[idx]).backward()
        opt.step()
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def _compare(make_a, make_b, atol=1e-10):
    model, X, y = _problem()
    model_b = copy.deepcopy(model)
    pa = _train(model, make_a(model.parameters()), X, y)
    pb = _train(model_b, make_b(model_b.parameters()), X, y)
    assert torch.allclose(pa, pb, atol=atol, rtol=0), (pa - pb).abs().max().item()


@pytest.mark.parametrize('wd', [0.0, 1e-3])
def test_sgd_pid_P_only_is_sgd(wd):
    """Secao 3.2: SGD e um controlador P (Kp = 1, sem I nem D)."""
    _compare(lambda p: SGD_PID(p, lr=0.05, momentum=0.9, weight_decay=wd, Kp=1, Ki=0, Kd=0),
             lambda p: torch.optim.SGD(p, lr=0.05, weight_decay=wd))


@pytest.mark.parametrize('wd', [0.0, 1e-3])
def test_sgd_pid_I_only_is_momentum(wd):
    """Secao 3.3 (forma do PyTorch): Kp=0, Ki=1, Kd=0 e o SGD-Momentum."""
    _compare(lambda p: SGD_PID(p, lr=0.05, momentum=0.9, weight_decay=wd, Kp=0, Ki=1, Kd=0),
             lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9, weight_decay=wd))


def test_paper_form_without_D_is_momentum():
    """Eq. (10) com Kd = 0 reduz-se a Eq. (3), o SGD-Momentum."""
    _compare(lambda p: SGD_PID_Paper(p, lr=0.05, alpha=0.9, Kd=0.0),
             lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9))


@pytest.mark.parametrize('k', [0.5, 3.0, 100.0])
def test_separated_and_paper_forms_differ_by_sign_of_D(k):
    """Forma separada (codigo publico) com Kd = k equivale a forma da Eq. (10)
    com Kd = -lr*k: o termo D tem sinais OPOSTOS no codigo e no texto."""
    lr = 0.05
    _compare(lambda p: SGD_PID(p, lr=lr, momentum=0.9, Kp=0, Ki=1, Kd=k),
             lambda p: SGD_PID_Paper(p, lr=lr, alpha=0.9, Kd=-lr * k),
             atol=1e-9)


@pytest.mark.parametrize('wd', [0.0, 1e-3])
def test_adam_pid_I_only_is_adam(wd):
    """Dai et al.: Adam e um controlador I adaptativo (Kp=0, Ki=1, Kd=0)."""
    _compare(lambda p: Adam_PID(p, lr=1e-3, weight_decay=wd, Kp=0, Ki=1, Kd=0),
             lambda p: torch.optim.Adam(p, lr=1e-3, weight_decay=wd),
             atol=1e-9)


def test_D_term_changes_the_update():
    """Sanidade: com Kd != 0 o resultado tem que ser diferente do momentum."""
    with pytest.raises(AssertionError):
        _compare(lambda p: SGD_PID_Paper(p, lr=0.05, alpha=0.9, Kd=0.5),
                 lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9))
