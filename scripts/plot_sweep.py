"""Figura de uma varredura (saida do scripts/sweep.py).

  - grade com 2 hiperparametros -> mapa (linhas = 1o, colunas = 2o) com a
    acuracia em cada celula, o status do treino e, se houver lambda_max,
    contorno vermelho nas celulas que a teoria linear preve instaveis;
  - grade com 1 hiperparametro -> dois paineis: acuracia x parametro (com o
    status de cada semente) e raio espectral previsto x parametro.

Uso:
    python scripts/plot_sweep.py results/<nome>/<timestamp>/sweep.json
Salva mapa_varredura.png ao lado do sweep.json.
"""

import argparse
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

INK, MUTED, GRAY = '#0b0b0b', '#52514e', '#d9d8d4'
STATUS_STYLE = {   # forma + cor: o status nunca depende so da cor
    'convergiu':     {'marker': 'o', 'color': '#2a78d6', 'label': 'convergiu', 'sym': ''},
    'instavel':      {'marker': '^', 'color': '#eb6834', 'label': 'instável (perda sobe)',
                      'sym': '△'},
    'nao_convergiu': {'marker': 's', 'color': '#6250d6', 'label': 'não convergiu', 'sym': '□'},
    'divergiu':      {'marker': 'X', 'color': '#e34948', 'label': 'divergiu', 'sym': '×'},
}


def fmt(v):
    return f'{v:g}'


def style_axes():
    ax = plt.gca()
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def plot_2d(res, keys):
    ky, kx = keys
    ys = sorted({s['params'][ky] for s in res['summary']})
    xs = sorted({s['params'][kx] for s in res['summary']})
    acc = np.full((len(ys), len(xs)), np.nan)
    cell = {}
    for s in res['summary']:
        i, j = ys.index(s['params'][ky]), xs.index(s['params'][kx])
        cell[i, j] = s
        if s['status'] != 'divergiu' and s['acc_media'] is not None:
            acc[i, j] = s['acc_media']

    finite = acc[np.isfinite(acc)]
    vmin = max(0.5, float(finite.min())) if finite.size else 0.5
    vmax = float(finite.max()) if finite.size else 1.0
    if vmax - vmin < 0.005:                       # todas iguais: evita escala degenerada
        vmin = vmax - 0.01

    plt.figure(figsize=(1.15 * len(xs) + 3, 0.75 * len(ys) + 2.6))
    plt.subplot(1, 1, 1)
    masked = np.ma.masked_invalid(acc)
    plt.imshow(np.ones_like(acc), cmap=matplotlib.colors.ListedColormap([GRAY]),
               aspect='auto', origin='lower')
    im = plt.imshow(masked, cmap='Blues', vmin=vmin, vmax=vmax, aspect='auto', origin='lower')
    mid = (vmin + vmax) / 2
    unstable_pred = False
    for (i, j), s in cell.items():
        st = STATUS_STYLE[s['status']]
        a = s['acc_media']
        txt = st['sym'] if s['status'] == 'divergiu' else f"{100 * a:.1f}{st['sym']}"
        color = 'white' if (a is not None and a > mid and s['status'] != 'divergiu') else INK
        plt.text(j, i, txt, ha='center', va='center', fontsize=9, color=color)
        if s['rho_previsto'] is not None and s['rho_previsto'] >= 1:
            unstable_pred = True
            plt.gca().add_patch(Rectangle((j - 0.47, i - 0.47), 0.94, 0.94, fill=False,
                                          ec='#e34948', lw=2.2))
    plt.xticks(range(len(xs)), [fmt(x) for x in xs])
    plt.yticks(range(len(ys)), [fmt(y) for y in ys])
    plt.xlabel(kx)
    plt.ylabel(ky)
    cb = plt.colorbar(im)
    cb.set_label(f"acurácia média ({res['eval_on']}), %")
    cb.ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{100*v:.0f}'))

    lam = res.get('lambda_max')
    sub = f"  |  λmax(init) = {lam:.3g}" if lam is not None else ''
    plt.title(f"{res['config']['name']}{sub}", fontsize=10)
    notes = ['△ instável   □ não convergiu   × divergiu (cinza)']
    if unstable_pred:
        notes.append('contorno vermelho: teoria linear prevê ρ ≥ 1')
    plt.figtext(0.01, 0.01, '   |   '.join(notes), fontsize=8, color=MUTED)
    plt.tight_layout(rect=(0, 0.04, 1, 1))


def plot_1d(res, key):
    xs = sorted({s['params'][key] for s in res['summary']})
    symlog = any(x <= 0 for x in xs)
    has_rho = any(s['rho_previsto'] is not None for s in res['summary'])

    plt.figure(figsize=(9, 6.5 if has_rho else 4.2))
    ax_top = plt.subplot(2 if has_rho else 1, 1, 1)
    accs = [r['final_eval_acc'] for r in res['runs']
            if r['final_eval_acc'] is not None and r['status'] != 'divergiu']
    floor = (min(accs) if accs else 0.5) - 0.02     # onde desenhar quem divergiu
    summ = sorted(res['summary'], key=lambda s: s['params'][key])
    xm = [s['params'][key] for s in summ if s['acc_media'] is not None]
    ym = [s['acc_media'] for s in summ if s['acc_media'] is not None]
    plt.plot(xm, ym, '-', color=MUTED, lw=1.2, zorder=1)
    seen = set()
    for r in res['runs']:
        st = STATUS_STYLE[r['status']]
        y = floor if (r['status'] == 'divergiu' or r['final_eval_acc'] is None) \
            else r['final_eval_acc']
        plt.scatter(r['params'][key], y, marker=st['marker'], s=60, color=st['color'],
                    edgecolor='white', linewidth=0.8, zorder=3,
                    label=st['label'] if r['status'] not in seen else None)
        seen.add(r['status'])
    if symlog:
        plt.xscale('symlog', linthresh=max(1e-3, min(abs(x) for x in xs if x != 0)))
    else:
        plt.xscale('log')
    plt.ylabel(f"acurácia final ({res['eval_on']})")
    plt.ylim(floor - 0.02, None)
    if 'divergiu' in seen:
        plt.axhline(floor, color='#e34948', lw=0.8, ls=':')
        plt.text(1.01, floor, 'divergiu\n(sem acurácia)', color='#e34948', fontsize=8,
                 va='center', transform=plt.gca().get_yaxis_transform())
    plt.legend(fontsize=8, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=4,
               frameon=False)
    plt.title(res['config']['name'], fontsize=10, pad=24)
    plt.grid(alpha=0.25)
    style_axes()

    if has_rho:
        plt.subplot(2, 1, 2, sharex=ax_top)
        xr = [s['params'][key] for s in summ]
        yr = [s['rho_previsto'] for s in summ]
        plt.plot(xr, yr, 'o-', color=INK, ms=5)
        plt.axhline(1.0, color='#e34948', lw=1.4, ls='--')
        plt.yscale('log')
        plt.text(1.01, 1.0, 'ρ = 1\n(fronteira)', color='#e34948', fontsize=8, va='center',
                 transform=plt.gca().get_yaxis_transform())
        if symlog:
            plt.xscale('symlog', linthresh=max(1e-3, min(abs(x) for x in xs if x != 0)))
        else:
            plt.xscale('log')
        plt.ylabel(f"ρ previsto (λmax = {res['lambda_max']:.3g})")
        plt.grid(alpha=0.25)
        style_axes()
    plt.xlabel(key)
    plt.tight_layout()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('sweep')
    p.add_argument('--out', default=None)
    args = p.parse_args()
    with open(args.sweep, encoding='utf-8') as f:
        res = json.load(f)
    keys = list(res['config']['sweep']['grid'])
    if len(keys) == 1:
        plot_1d(res, keys[0])
    elif len(keys) == 2:
        plot_2d(res, keys)
    else:
        raise SystemExit('plot_sweep.py desenha grades com 1 ou 2 hiperparametros')
    out = args.out or os.path.join(os.path.dirname(args.sweep), 'mapa_varredura.png')
    plt.savefig(out, dpi=200)
    print(f'Figura salva em {out}')


if __name__ == '__main__':
    main()
