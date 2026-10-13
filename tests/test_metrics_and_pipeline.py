"""Metricas e um teste de ponta a ponta do pipeline (dataset sintetico, sem download)."""

import json
import os
import textwrap

from optctrl.metrics import epochs_to_target, overshoot, settling_step


def test_epochs_to_target():
    assert epochs_to_target([0.5, 0.9, 0.97, 0.99], 0.97) == 3
    assert epochs_to_target([0.5, 0.6], 0.97) is None


def test_overshoot_step_response():
    # degrau de -10 para 0: passa 2 unidades do alvo -> 20 %
    assert overshoot([-10, -3, 2, -0.5, 0.1, 0], 0.0) == 0.2
    assert overshoot([-10, -5, -1, 0], 0.0) == 0.0          # sem overshoot
    assert overshoot([0, 6, 4, 5], 5.0) == 0.2              # = Eq. (8) quando theta0 = 0


def test_settling_step():
    assert settling_step([-10, -5, 1, -0.1, 0.05, 0.0], 0.0, tol=0.02) == 3
    assert settling_step([-10, -5, -1], 0.0) is None


def test_train_script_end_to_end(tmp_path):
    from scripts_train import main  # importado via conftest

    cfg = tmp_path / 'smoke.yaml'
    cfg.write_text(textwrap.dedent("""
        name: smoke
        dataset: {name: synthetic, val_size: 400}
        eval_on: val
        model: {name: mlp, hidden: 16}
        train: {epochs: 3, batch_size: 64, target: 0.9}
        seeds: [0, 1]
        optimizers:
          momentum: {name: sgd, lr: 0.05, momentum: 0.9}
          pid: {name: sgd_pid, lr: 0.05, momentum: 0.9, Kp: 1.0, Ki: 1.0, Kd: 1.0}
          adam_pid: {name: adam_pid, lr: 0.001, Kp: 0.5, Ki: 1.0, Kd: 0.3}
    """), encoding='utf-8')

    out_dir = main([str(cfg), '--out-root', str(tmp_path / 'results'), '--quiet',
                    '--device', 'cpu'])
    with open(os.path.join(out_dir, 'results.json'), encoding='utf-8') as f:
        res = json.load(f)

    assert set(res['aggregate']) == {'momentum', 'pid', 'adam_pid'}
    for agg in res['aggregate'].values():
        assert agg['n_completed'] == 2
        assert len(agg['curves']['eval_acc']['mean']) == 3
        assert agg['final_acc'][0] > 0.8          # gaussianas separaveis: tem que aprender
    assert res['config']['seeds'] == [0, 1]
    assert 'commit' in res['git'] and 'torch' in res['environment']
