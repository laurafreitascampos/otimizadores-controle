"""A auditoria aprova um resultado real e reprova um resultado adulterado."""

import json
import textwrap

from scripts_audit_results import main as audit
from scripts_train import main as train


def test_audit_accepts_real_and_rejects_tampered(tmp_path):
    cfg = tmp_path / 'c.yaml'
    cfg.write_text(textwrap.dedent("""
        name: auditoria
        dataset: {name: synthetic, val_size: 0}
        eval_on: test
        model: {name: mlp, hidden: 8}
        train: {epochs: 2, batch_size: 64}
        seeds: [0, 1]
        optimizers:
          momentum: {name: sgd, lr: 0.01, momentum: 0.9}
    """), encoding='utf-8')
    out = train([str(cfg), '--out-root', str(tmp_path / 'r'), '--quiet', '--device', 'cpu'])
    path = f'{out}/results.json'
    assert audit([path, '--rerun', '--epochs', '2', '--device', 'cpu']) == 0

    with open(path, encoding='utf-8') as f:
        res = json.load(f)
    res['aggregate']['momentum']['final_acc'][0] = 0.123          # "maquiado"
    bad = tmp_path / 'bad.json'
    bad.write_text(json.dumps(res), encoding='utf-8')
    assert audit([str(bad)]) == 1
