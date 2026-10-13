"""compare_stats.py com resultados fabricados (numeros conhecidos)."""

import json
import math

from scripts_compare_stats import main


def _hist(seed, acc, ett=None, diverged=False):
    return {'seed': seed, 'eval_acc': [acc - 0.01, acc], 'epochs_to_target': ett,
            'diverged': diverged}


def _write(path, per_seed):
    cfg = {'dataset': {'name': 'mnist'}, 'model': {'name': 'mlp'},
           'train': {'epochs': 2, 'batch_size': 100}, 'eval_on': 'test'}
    path.write_text(json.dumps({'config': cfg, 'eval_on': 'test', 'per_seed': per_seed}),
                    encoding='utf-8')
    return str(path)


def test_compare_stats_paired_and_merge(tmp_path):
    base = [0.980, 0.982, 0.979, 0.981, 0.983]
    a = _write(tmp_path / 'a.json', {
        'ref': [_hist(s, b, 7) for s, b in enumerate(base)],
        'melhor': [_hist(s, b + 0.002, 4) for s, b in enumerate(base)],
    })
    b = _write(tmp_path / 'b.json', {   # segundo arquivo: um rotulo novo, uma semente divergiu
        'outro': [_hist(s, b, 7) for s, b in enumerate(base[:4])] + [_hist(4, 0.1, None, True)],
    })
    tex = tmp_path / 't.tex'
    blocks = main([a, b, '--ref', 'ref', '--latex', str(tex)])
    rows = {r['label']: r for r in blocks[0][1]}

    assert math.isclose(rows['melhor']['diff_paired'], 0.2, abs_tol=1e-9)    # em pontos %
    assert rows['melhor']['p_paired'] < 1e-6
    assert rows['outro']['n'] == 4 and rows['outro']['n_total'] == 4          # divergiu: fora
    assert rows['outro']['n_paired'] == 4
    ett = {r['label']: r for r in blocks[1][1]}
    assert math.isclose(ett['melhor']['diff_paired'], -3.0)

    latex = tex.read_text(encoding='utf-8')
    assert r'acurácia final (\%)' in latex and '{,}' in latex and r'\bottomrule' in latex
