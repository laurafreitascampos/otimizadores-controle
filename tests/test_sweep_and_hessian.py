"""Classificacao dos treinos, iteracao de potencia da Hessiana e a varredura de ponta a ponta."""

import json
import math
import os
import textwrap

import numpy as np
import torch
from scripts_sweep import classify, main, worst
from torch import nn

from optctrl.analysis.hessian import top_hessian_eigenvalue


def _hist(losses, diverged=False):
    return {'train_loss': losses, 'diverged': diverged}


def test_classify():
    C = 10
    assert classify(_hist([0.5, 0.3, 0.2]), C) == 'convergiu'
    assert classify(_hist([0.5, 0.3, 0.4]), C) == 'instavel'          # sobe 33 %
    assert classify(_hist([3.0, 2.9, 2.8]), C) == 'nao_convergiu'     # >= ln(10)
    assert classify(_hist([0.5], diverged=True), C) == 'divergiu'
    assert classify(_hist([]), C) == 'divergiu'
    assert worst(['convergiu', 'instavel', 'convergiu']) == 'instavel'


def test_hessian_power_iteration_on_quadratic():
    """Regressao linear com perda 0.5*||Xw - y||^2: a Hessiana e X^T X."""
    torch.manual_seed(0)
    X = torch.randn(50, 6, dtype=torch.float64)
    y = torch.randn(50, 1, dtype=torch.float64)
    model = nn.Linear(6, 1, bias=False).double()
    lam, _ = top_hessian_eigenvalue(model, X, y, criterion=lambda o, t: 0.5 * ((o - t) ** 2).sum(),
                                    iters=500, tol=1e-12)
    expected = np.linalg.eigvalsh((X.T @ X).numpy()).max()
    assert math.isclose(lam, expected, rel_tol=1e-6)


def test_sweep_end_to_end(tmp_path):
    cfg = tmp_path / 'sweep.yaml'
    cfg.write_text(textwrap.dedent("""
        name: sweep_teste
        dataset: {name: synthetic, val_size: 400}
        eval_on: val
        model: {name: mlp, hidden: 16}
        train: {epochs: 2, batch_size: 64}
        seeds: [0, 1]
        hessian: {n_samples: 300, iters: 30}
        sweep:
          optimizer: sgd
          fixed: {weight_decay: 1.0e-4}
          grid:
            momentum: [0.0, 0.9]
            lr: [0.01, 1000.0]
    """), encoding='utf-8')
    out = main([str(cfg), '--device', 'cpu', '--out-root', str(tmp_path / 'r')])
    with open(os.path.join(out, 'sweep.json'), encoding='utf-8') as f:
        res = json.load(f)
    assert len(res['runs']) == 8 and len(res['summary']) == 4
    assert res['lambda_max'] > 0
    by_lr = {(s['params']['momentum'], s['params']['lr']): s for s in res['summary']}
    assert by_lr[(0.0, 0.01)]['status'] == 'convergiu'
    assert by_lr[(0.0, 1000.0)]['status'] in ('divergiu', 'nao_convergiu')
    assert by_lr[(0.0, 1000.0)]['rho_previsto'] > 1
