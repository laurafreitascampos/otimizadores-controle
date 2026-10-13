"""Executa um experimento descrito em um config YAML.

Uso:
    python scripts/train.py experiments/an2018_pid/configs/fig4_mnist_paper.yaml
    python scripts/train.py <config> --seeds 0 1 --epochs 2      # teste rapido
    python scripts/train.py <config> --only pid_repo             # um otimizador so

Cada execucao gera results/<nome>/<timestamp>/results.json com o config, o
commit do git, o ambiente, as curvas por semente e a agregacao.
"""

import argparse
import copy
import os
import sys

import torch
import yaml

from optctrl.data import load_splits
from optctrl.metrics.energy import track_energy
from optctrl.models import build_model
from optctrl.optimizers import build_optimizer
from optctrl.training import aggregate, train_eval
from optctrl.utils import environment_info, git_info, save_json, timestamp


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('config')
    p.add_argument('--seeds', type=int, nargs='+', help='sobrescreve as sementes do config')
    p.add_argument('--epochs', type=int, help='sobrescreve train.epochs')
    p.add_argument('--only', nargs='+', help='roda so estes rotulos de otimizador')
    p.add_argument('--device', default=None, help="'cuda' ou 'cpu' (padrao: auto)")
    p.add_argument('--data-dir', default='./data')
    p.add_argument('--out-root', default='./results')
    p.add_argument('--energy', action='store_true', help='mede energia com CodeCarbon')
    p.add_argument('--quiet', action='store_true')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    with open(args.config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg_used = copy.deepcopy(cfg)

    if args.seeds:
        cfg_used['seeds'] = args.seeds
    if args.epochs:
        cfg_used['train']['epochs'] = args.epochs
    optimizers = cfg_used['optimizers']
    if args.only:
        optimizers = {k: v for k, v in optimizers.items() if k in args.only}
        if not optimizers:
            sys.exit(f"Nenhum otimizador com rotulo em {args.only}")

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    tr_cfg = cfg_used['train']
    ds_cfg = cfg_used['dataset']
    eval_on = cfg_used.get('eval_on', 'val')

    data = load_splits(ds_cfg['name'], args.data_dir, device,
                       val_size=ds_cfg.get('val_size', 10000),
                       split_seed=ds_cfg.get('split_seed', 0))
    eval_set = data[eval_on]
    if eval_set is None:
        sys.exit("eval_on='val' exige dataset.val_size > 0")

    print(f"Experimento: {cfg_used['name']} | dispositivo: {device} | avaliacao em: {eval_on}")
    print(f"Treino {tuple(data['train'][0].shape)} | avaliacao {tuple(eval_set[0].shape)}")

    m_cfg = dict(cfg_used['model'])
    m_name = m_cfg.pop('name')

    def model_fn():
        return build_model(m_name, data['input_dim'], data['num_classes'], **m_cfg)

    per_seed, agg = {}, {}
    for label, o_cfg in optimizers.items():
        o_cfg = dict(o_cfg)
        o_name = o_cfg.pop('name')
        print(f"\n=== {label} ({o_name}) {o_cfg} ===")

        def opt_fn(params, o_name=o_name, o_cfg=o_cfg):
            return build_optimizer(o_name, params, **o_cfg)

        hists = []
        for seed in cfg_used['seeds']:
            with track_energy(enabled=args.energy) as energy:
                h = train_eval(model_fn, opt_fn, data['train'], eval_set,
                               epochs=tr_cfg['epochs'], batch_size=tr_cfg['batch_size'],
                               seed=seed, target=tr_cfg.get('target'),
                               clip=tr_cfg.get('clip', 0.0), device=device,
                               verbose=not args.quiet, label=label)
            h.update(energy)
            hists.append(h)
        per_seed[label] = hists
        agg[label] = aggregate(hists)

    out_dir = os.path.join(args.out_root, cfg_used['name'], timestamp())
    save_json({'config': cfg_used, 'config_path': args.config, 'git': git_info(),
               'environment': environment_info(device), 'eval_on': eval_on,
               'aggregate': agg, 'per_seed': per_seed},
              os.path.join(out_dir, 'results.json'))

    print("\n================ resumo (media +/- desvio entre sementes) ================")
    for label, a in agg.items():
        if not a['final_acc']:
            print(f"{label:>16} | divergiu em todas as sementes")
            continue
        fa = torch.tensor(a['final_acc'])
        reached = [e for e in a['epochs_to_target'] if e is not None]
        ett = f"{len(reached)}/{len(a['epochs_to_target'])} atingiram" + \
              (f" (media {sum(reached) / len(reached):.1f} ep)" if reached else "")
        acc = f"{fa.mean():.4f} +/- {fa.std(unbiased=False):.4f}"
        print(f"{label:>16} | acc final {acc} | alvo: {ett}")
    print(f"\nResultados em {out_dir}")
    return out_dir


if __name__ == '__main__':
    main()
