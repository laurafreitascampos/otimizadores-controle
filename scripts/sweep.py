"""Varredura de hiperparametros: mapa EMPIRICO de estabilidade numa rede real.

Para cada ponto da grade (e cada semente) treina o modelo e classifica o
comportamento do treinamento:

    convergiu      perda de treino cai epoca a epoca
    instavel       perda de treino SOBE em alguma epoca (mais que rise_tol)
    nao_convergiu  perda final >= ln(C): nao melhor que chutar a classe
    divergiu       perda nao finita (NaN/inf)

Se o config tiver a secao `hessian`, mede tambem a curvatura maxima da perda
na inicializacao (lambda_max, maior autovalor da Hessiana) e, para cada ponto
da grade, o raio espectral PREVISTO pelo modelo linear em tempo discreto
(optctrl.analysis.linear) naquela curvatura. Assim o mapa empirico pode ser
comparado com a previsao teorica.

Uso:
    python scripts/sweep.py experiments/estabilidade_discreta/configs/mnist_momentum_lr.yaml
    python scripts/sweep.py <config> --seeds 0 --epochs 2      # teste rapido
Depois:
    python scripts/plot_sweep.py results/<nome>/<timestamp>/sweep.json
"""

import argparse
import copy
import itertools
import math
import os
import sys
import time

import torch
import yaml

from optctrl.analysis import predicted_rho
from optctrl.analysis.hessian import top_hessian_eigenvalue
from optctrl.data import load_splits
from optctrl.models import build_model
from optctrl.optimizers import build_optimizer
from optctrl.training import set_seed, train_eval
from optctrl.utils import environment_info, git_info, save_json, timestamp

SEVERITY = ['convergiu', 'instavel', 'nao_convergiu', 'divergiu']


def classify(hist, num_classes, rise_tol=0.10):
    """Classe do comportamento de UMA execucao (ver docstring do modulo)."""
    tl = hist['train_loss']
    if hist['diverged'] or not tl:
        return 'divergiu'
    if tl[-1] >= math.log(num_classes):
        return 'nao_convergiu'
    rises = [tl[i] / tl[i - 1] - 1.0 for i in range(1, len(tl)) if tl[i - 1] > 0]
    if rises and max(rises) > rise_tol:
        return 'instavel'
    return 'convergiu'


def worst(statuses):
    return max(statuses, key=SEVERITY.index)


def expand_grid(grid):
    keys = list(grid)
    for values in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('config')
    p.add_argument('--seeds', type=int, nargs='+')
    p.add_argument('--epochs', type=int)
    p.add_argument('--device', default=None)
    p.add_argument('--data-dir', default='./data')
    p.add_argument('--out-root', default='./results')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    with open(args.config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    if args.seeds:
        cfg['seeds'] = args.seeds
    if args.epochs:
        cfg['train']['epochs'] = args.epochs

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    ds, tr, sw = cfg['dataset'], cfg['train'], cfg['sweep']
    eval_on = cfg.get('eval_on', 'val')
    rise_tol = cfg.get('rise_tol', 0.10)

    data = load_splits(ds['name'], args.data_dir, device,
                       val_size=ds.get('val_size', 10000), split_seed=ds.get('split_seed', 0))
    eval_set = data[eval_on]
    if eval_set is None:
        sys.exit("eval_on='val' exige dataset.val_size > 0")
    C = data['num_classes']

    m_cfg = dict(cfg['model'])
    m_name = m_cfg.pop('name')

    def model_fn():
        return build_model(m_name, data['input_dim'], C, **m_cfg)

    points = list(expand_grid(sw['grid']))
    n_runs = len(points) * len(cfg['seeds'])
    print(f"Varredura: {cfg['name']} | {sw['optimizer']} | {len(points)} pontos x "
          f"{len(cfg['seeds'])} sementes = {n_runs} treinos | {device} | avaliacao em {eval_on}")

    # ---- curvatura maxima na inicializacao (opcional) ----
    lambda_max = None
    if cfg.get('hessian'):
        hc = cfg['hessian']
        set_seed(cfg['seeds'][0])
        model = model_fn().to(device)
        n = hc.get('n_samples', 2000)
        X, Y = data['train'][0][:n], data['train'][1][:n]
        t0 = time.perf_counter()
        lambda_max, its = top_hessian_eigenvalue(model, X, Y, iters=hc.get('iters', 100))
        print(f"lambda_max (Hessiana na inicializacao, {n} amostras): {lambda_max:.4g} "
              f"[{its} iteracoes, {time.perf_counter() - t0:.1f}s]")
        del model

    # ---- treinos ----
    runs, k = [], 0
    for params in points:
        hp = {**sw.get('fixed', {}), **params}
        rho = predicted_rho(sw['optimizer'], hp, lambda_max) if lambda_max is not None else None
        for seed in cfg['seeds']:
            k += 1
            t0 = time.perf_counter()
            h = train_eval(model_fn, lambda p, hp=hp: build_optimizer(sw['optimizer'], p, **hp),
                           data['train'], eval_set, epochs=tr['epochs'],
                           batch_size=tr['batch_size'], seed=seed, target=tr.get('target'),
                           clip=tr.get('clip', 0.0), device=device, verbose=False)
            status = classify(h, C, rise_tol)
            acc = h['eval_acc'][-1] if h['eval_acc'] else None
            runs.append({'params': params, 'seed': seed, 'status': status,
                         'final_eval_acc': acc,
                         'final_train_loss': h['train_loss'][-1] if h['train_loss'] else None,
                         'rho_previsto': rho, 'hist': h})
            acc_s = f"{acc:.4f}" if (acc is not None and status != 'divergiu') else '  —   '
            rho_s = f" | rho previsto {rho:.4f}" if rho is not None else ''
            print(f"[{k:3d}/{n_runs}] {params} s{seed}: {status:13s} acc {acc_s}{rho_s} "
                  f"({time.perf_counter() - t0:.1f}s)", flush=True)

    # ---- resumo por ponto da grade ----
    summary = []
    for params in points:
        rs = [r for r in runs if r['params'] == params]
        accs = [r['final_eval_acc'] for r in rs if r['status'] != 'divergiu'
                and r['final_eval_acc'] is not None]
        counts = {s: sum(r['status'] == s for r in rs) for s in SEVERITY}
        summary.append({'params': params, 'status': worst([r['status'] for r in rs]),
                        'contagem': counts,
                        'acc_media': sum(accs) / len(accs) if accs else None,
                        'acc_min': min(accs) if accs else None,
                        'rho_previsto': rs[0]['rho_previsto']})

    out_dir = os.path.join(args.out_root, cfg['name'], timestamp())
    save_json({'config': cfg, 'config_path': args.config, 'git': git_info(),
               'environment': environment_info(device), 'eval_on': eval_on,
               'num_classes': C, 'lambda_max': lambda_max, 'summary': summary, 'runs': runs},
              os.path.join(out_dir, 'sweep.json'))

    hits = [s for s in summary if s['rho_previsto'] is not None]
    if hits:
        agree = sum((s['rho_previsto'] >= 1) == (s['status'] == 'divergiu') for s in hits)
        print(f"\nPrevisao linear (rho >= 1 -> diverge) concorda com o observado em "
              f"{agree}/{len(hits)} pontos da grade.")
    print(f"Resultados em {out_dir}")
    print(f"Figura: python scripts/plot_sweep.py {os.path.join(out_dir, 'sweep.json')}")
    return out_dir


if __name__ == '__main__':
    main()
