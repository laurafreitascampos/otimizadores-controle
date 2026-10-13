"""O modelo linear (matriz de malha fechada) e EXATO para quadraticas.

Compara A^t x_0 com a trajetoria produzida pelas classes de otimizador reais
(as mesmas usadas no MNIST). Se estes testes passam, o mapa de estabilidade
analitico descreve o codigo que roda nos experimentos, e nao uma idealizacao.
"""

import numpy as np
import pytest
import torch

from optctrl.analysis import (
    closed_loop_matrix,
    initial_state,
    momentum_boundaries,
    momentum_matrix,
    poles,
    simulate,
    spectral_radius,
)
from optctrl.optimizers import build_optimizer

STEPS = 60


def _run_optimizer(opt_name, hparams, a, theta0=3.0, steps=STEPS):
    theta = torch.tensor([theta0], dtype=torch.float64, requires_grad=True)
    opt = build_optimizer(opt_name, [theta], **hparams)
    traj = [theta0]
    for _ in range(steps):
        opt.zero_grad()
        (0.5 * a * theta ** 2).sum().backward()
        opt.step()
        traj.append(theta.item())
    return np.array(traj)


CASES = [
    ('sgd', {'lr': 0.1}, 2.0),
    ('sgd', {'lr': 0.05, 'momentum': 0.9}, 4.0),
    ('sgd', {'lr': 0.05, 'momentum': 0.9, 'weight_decay': 0.1}, 4.0),
    ('sgd_pid', {'lr': 0.01, 'momentum': 0.9, 'Kp': 1.0, 'Ki': 3.0, 'Kd': 100.0}, 0.5),
    ('sgd_pid', {'lr': 0.02, 'momentum': 0.8, 'Kp': 0.5, 'Ki': 1.0, 'Kd': -5.0}, 3.0),
    ('sgd_pid_paper', {'lr': 0.01, 'alpha': 0.9, 'Kd': 0.5}, 4.0),
    ('sgd_pid_paper', {'lr': 0.01, 'alpha': 0.9, 'Kd': -0.93}, 0.2),
]


@pytest.mark.parametrize('opt_name,hparams,a', CASES)
def test_matrix_reproduces_optimizer_trajectory(opt_name, hparams, a):
    A = closed_loop_matrix(opt_name, hparams, a)
    x = simulate(A, initial_state(3.0, A.shape[0]), STEPS)[:, 0]
    real = _run_optimizer(opt_name, hparams, a)
    assert np.allclose(x, real, rtol=1e-9, atol=1e-9), np.abs(x - real).max()


def test_momentum_characteristic_polynomial():
    h, m = 0.04, 0.9
    z = np.sort_complex(poles(momentum_matrix(h, m)))
    expected = np.sort_complex(np.roots([1, -(1 + m - h), m]))
    assert np.allclose(z, expected)


def test_momentum_regimes():
    m = 0.9
    b = momentum_boundaries(m)
    # abaixo de (1-sqrt m)^2: polos reais; dentro da faixa: complexos
    assert np.all(np.isreal(poles(momentum_matrix(0.5 * b['complexo_inicio'], m))))
    assert np.any(np.iscomplex(poles(momentum_matrix(0.04, m))))
    # fronteira de estabilidade h = 2(1+m)
    assert spectral_radius(momentum_matrix(0.99 * b['instavel'], m)) < 1
    assert spectral_radius(momentum_matrix(1.01 * b['instavel'], m)) > 1


def test_largest_stable_recovers_momentum_boundary():
    from optctrl.analysis import largest_stable
    h = largest_stable(lambda x: spectral_radius(momentum_matrix(x, 0.9)))
    assert abs(h - 2 * (1 + 0.9)) < 1e-5


def test_adaptive_methods_are_not_linear():
    assert closed_loop_matrix('adam', {'lr': 1e-3}, 1.0) is None
