"""Auditoria de um results.json: os numeros sao reais e vem deste codigo?

Quatro verificacoes, da mais barata a mais forte:

  1. PROCEDENCIA   commit do git, alteracoes nao commitadas, ambiente, config.
                   Um resultado sem commit (ou com dirty=True) nao pode ser
                   reproduzido exatamente: refaca-o com o codigo commitado.
  2. INTEGRIDADE   recalcula media/desvio a partir das curvas de cada semente
                   e compara com o resumo salvo. Detecta arquivo editado a mao
                   ou resumo que nao corresponde aos dados brutos.
  3. SANIDADE      acuracias entre 0 e 1, sem NaN, numero de epocas certo,
                   sementes distintas, aprendizado acima do acaso (1/C).
  4. REEXECUCAO    (--rerun) treina de novo UMA semente de UM otimizador com o
                   config salvo e compara epoca a epoca. E a prova mais forte:
                   o numero sai de novo do codigo, na sua frente.
                   Em GPU ha nao determinismo numerico pequeno (ordem de 1e-3
                   na acuracia); em CPU o resultado costuma ser identico.

Uso:
    python scripts/audit_results.py results/<nome>/<data-hora>/results.json
    python scripts/audit_results.py <results.json> --rerun --label pid_repo --seed 0 --epochs 3
Codigo de saida: 0 se tudo passou, 1 se alguma verificacao falhou.
"""

import argparse
import json
import math
import sys

import numpy as np

OK, WARN, FAIL = 'OK  ', 'AVISO', 'FALHA'


class Report:
    def __init__(self):
        self.failed = False

    def line(self, status, msg):
        if status == FAIL:
            self.failed = True
        print(f'  [{status}] {msg}')


def check_provenance(res, rep, current_commit):
    print('\n1. Procedência')
    git = res.get('git', {})
    commit = git.get('commit')
    if not commit:
        rep.line(FAIL, 'sem commit registrado: não dá para saber qual código gerou o arquivo')
    else:
        rep.line(OK, f'gerado pelo commit {commit[:10]}')
        if git.get('dirty'):
            rep.line(WARN, 'havia alterações NÃO commitadas no código quando rodou '
                           '(reprodução exata não garantida)')
        else:
            rep.line(OK, 'código sem alterações locais no momento da execução')
        if current_commit and current_commit != commit:
            rep.line(WARN, f'o código atual está no commit {current_commit[:10]}; '
                           f'para reexecutar igual: git checkout {commit[:10]}')
    env = res.get('environment', {})
    if env:
        rep.line(OK, f"ambiente: Python {env.get('python')}, torch {env.get('torch')}, "
                     f"{env.get('gpu', env.get('device'))}")
    cfg = res.get('config', {})
    rep.line(OK, f"experimento '{cfg.get('name')}', config {res.get('config_path')}, "
                 f"sementes {cfg.get('seeds')}")


def check_integrity(res, rep):
    print('\n2. Integridade (resumo recalculado a partir das curvas de cada semente)')
    for label, hists in res['per_seed'].items():
        agg = res['aggregate'][label]
        ok_h = [h for h in hists if not h['diverged'] and h['eval_acc']]
        final = [h['eval_acc'][-1] for h in ok_h]
        if sorted(final) != sorted(agg['final_acc']):
            rep.line(FAIL, f'{label}: acurácias finais do resumo não batem com as curvas')
            continue
        if ok_h:
            n_ep = min(len(h['eval_acc']) for h in ok_h)
            mean = np.mean([h['eval_acc'][:n_ep] for h in ok_h], axis=0)
            stored = np.array(agg['curves']['eval_acc']['mean'])
            if not np.allclose(mean, stored, atol=1e-12):
                rep.line(FAIL, f'{label}: curva média salva difere da recalculada')
                continue
        rep.line(OK, f'{label}: {len(final)}/{len(hists)} sementes, média final '
                     f'{100 * np.mean(final):.2f} % (recalculada = salva)' if final else
                 f'{label}: todas as sementes divergiram (registrado)')


def check_sanity(res, rep):
    print('\n3. Sanidade')
    cfg = res['config']
    epochs = cfg['train']['epochs']
    chance = 1.0 / res.get('num_classes', 10)
    for label, hists in res['per_seed'].items():
        seeds = [h['seed'] for h in hists]
        if len(set(seeds)) != len(seeds):
            rep.line(FAIL, f'{label}: sementes repetidas {seeds}')
        problems = []
        for h in hists:
            for key in ('train_loss', 'eval_loss', 'train_acc', 'eval_acc'):
                vals = h[key]
                if any(not math.isfinite(v) for v in vals):
                    problems.append(f's{h["seed"]}: {key} com NaN/inf')
            if any(not 0.0 <= a <= 1.0 for a in h['eval_acc'] + h['train_acc']):
                problems.append(f's{h["seed"]}: acurácia fora de [0, 1]')
            if not h['diverged'] and len(h['eval_acc']) != epochs:
                problems.append(f's{h["seed"]}: {len(h["eval_acc"])} épocas, esperado {epochs}')
        if problems:
            for p in problems:
                rep.line(FAIL, f'{label}: {p}')
            continue
        learned = [h['eval_acc'][-1] for h in hists if h['eval_acc'] and not h['diverged']]
        if learned and min(learned) < chance + 0.05:
            rep.line(WARN, f'{label}: alguma semente terminou perto do acaso ({chance:.0%})')
        else:
            rep.line(OK, f'{label}: valores válidos, {epochs} épocas por semente')


def rerun(res, rep, label, seed, epochs, device):
    print(f'\n4. Reexecução: {label}, semente {seed}, {epochs} época(s)')
    import torch

    from optctrl.data import load_splits
    from optctrl.models import build_model
    from optctrl.optimizers import build_optimizer
    from optctrl.training import train_eval

    cfg = res['config']
    stored = next((h for h in res['per_seed'][label] if h['seed'] == seed), None)
    if stored is None:
        rep.line(FAIL, f'semente {seed} não existe em {label}')
        return
    dev = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    ds = cfg['dataset']
    data = load_splits(ds['name'], './data', dev, val_size=ds.get('val_size', 10000),
                       split_seed=ds.get('split_seed', 0))
    m_cfg = dict(cfg['model'])
    m_name = m_cfg.pop('name')
    o_cfg = dict(cfg['optimizers'][label])
    o_name = o_cfg.pop('name')
    h = train_eval(lambda: build_model(m_name, data['input_dim'], data['num_classes'], **m_cfg),
                   lambda p: build_optimizer(o_name, p, **o_cfg),
                   data['train'], data[res.get('eval_on', 'val')],
                   epochs=epochs, batch_size=cfg['train']['batch_size'], seed=seed,
                   device=dev, verbose=False)
    tol = 0.003 if dev.type == 'cuda' else 1e-6
    for e in range(min(epochs, len(stored['eval_acc']))):
        a, b = stored['eval_acc'][e], h['eval_acc'][e]
        status = OK if abs(a - b) <= tol else FAIL
        rep.line(status, f'época {e + 1}: salvo {100 * a:.2f} %  |  agora {100 * b:.2f} %  '
                         f'(diferença {100 * abs(a - b):.3f} pp, tolerância {100 * tol:.3f})')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('results')
    p.add_argument('--rerun', action='store_true')
    p.add_argument('--label', default=None, help='otimizador a reexecutar (padrão: o primeiro)')
    p.add_argument('--seed', type=int, default=None, help='semente (padrão: a primeira)')
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--device', default=None)
    args = p.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass

    with open(args.results, encoding='utf-8') as f:
        res = json.load(f)
    from optctrl.utils import git_info
    rep = Report()
    print(f'Auditoria de {args.results}')
    check_provenance(res, rep, git_info().get('commit'))
    check_integrity(res, rep)
    check_sanity(res, rep)
    if args.rerun:
        label = args.label or next(iter(res['per_seed']))
        seed = args.seed if args.seed is not None else res['config']['seeds'][0]
        rerun(res, rep, label, seed, args.epochs, args.device)

    print('\nResultado: ' + ('FALHOU em alguma verificação' if rep.failed
                             else 'todas as verificações passaram'))
    return 1 if rep.failed else 0


if __name__ == '__main__':
    sys.exit(main())
