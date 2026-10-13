"""Mapas de estabilidade ANALITICOS em tempo discreto + validacao no toy.

Gera tres figuras e uma tabela de validacao:

  1. regimes_momentum.png   SGD-Momentum no plano (h = lr*a, momento):
                            superamortecido / oscilatorio / zigue-zague / instavel
  2. mapa_pid_artigo.png    PID da Eq. (10) de An et al. no plano (curvatura a, Kd),
                            com os casos do toy de De Jong sobrepostos
  3. mapa_pid_codigo.png    PID na forma do codigo publico (Kp=1, Ki=3) no plano
                            (h = lr*a, Kd): a margem de estabilidade que o Kd consome
  validacao_toy.md/.json    rho previsto x comportamento observado no toy

Toda a analise e feita em tempo discreto, sobre as matrizes de
optctrl.analysis.linear (exatas para quadraticas: tests/test_linear_model.py).

Uso:
    python experiments/estabilidade_discreta/stability_map.py
    python experiments/estabilidade_discreta/stability_map.py --lambda-max 12.3
      (--lambda-max: curvatura maxima medida no MNIST pelo scripts/sweep.py;
       marca onde a configuracao do MNIST cai no mapa 3)
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import ListedColormap, TwoSlopeNorm  # noqa: E402

from optctrl.analysis import (  # noqa: E402
    largest_stable,
    momentum_boundaries,
    momentum_matrix,
    pid_paper_matrix,
    pid_separated_matrix,
    poles,
    spectral_radius,
)
from optctrl.metrics import overshoot, settling_step  # noqa: E402
from optctrl.optimizers import SGD_PID_Paper  # noqa: E402
from optctrl.utils import git_info, save_json, timestamp  # noqa: E402

# Paleta (categorica slot 1 / status critico) e tinta de texto
BLUE, RED, INK, MUTED = '#2a78d6', '#e34948', '#0b0b0b', '#52514e'
UNSTABLE_GRAY = '#d9d8d4'
CMAP_RHO = plt.get_cmap('Blues_r')      # sequencial: mais escuro = converge mais rapido

TOY = {'r': 0.01, 'alpha': 0.9, 'curvaturas': {'x₁': 0.2, 'x₂': 4.0},
       'Kd': [0.0, 0.1, 0.5, 0.93, -0.1, -0.5, -0.93], 'steps': 300}


def rho_grid(builder, xs, ys):
    """rho(builder(x, y)) numa grade; linhas = ys, colunas = xs."""
    mats = np.array([[builder(x, y) for x in xs] for y in ys])
    eig = np.linalg.eigvals(mats.reshape(-1, *mats.shape[-2:]))
    return np.abs(eig).max(axis=-1).reshape(len(ys), len(xs))


def draw_rho(X, Y, R, vmin=0.0):
    """Fundo: rho < 1 em azul sequencial, regiao instavel em cinza."""
    stable = np.ma.masked_where(R >= 1, R)
    unstable = np.ma.masked_where(R < 1, np.ones_like(R))
    mesh = plt.pcolormesh(X, Y, stable, cmap=CMAP_RHO, vmin=vmin, vmax=1.0, shading='auto')
    plt.pcolormesh(X, Y, unstable, cmap=ListedColormap([UNSTABLE_GRAY]), shading='auto')
    plt.contour(X, Y, R, levels=[1.0], colors=INK, linewidths=1.6)
    return mesh


def style_axes():
    ax = plt.gca()
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED)


# ----------------------------------------------------------------------------
# Figura 1: regimes do SGD-Momentum
# ----------------------------------------------------------------------------
def fig_momentum(out_dir):
    hs = np.logspace(-4, 1, 400)
    ms = np.linspace(0.0, 0.99, 300)
    R = rho_grid(lambda h, m: momentum_matrix(h, m), hs, ms)

    plt.figure(figsize=(7.5, 5))
    plt.subplot(1, 1, 1)
    mesh = draw_rho(hs, ms, R, vmin=0.0)
    b = [momentum_boundaries(m) for m in ms]
    plt.plot([x['complexo_inicio'] for x in b], ms, '--', color=INK, lw=1)
    plt.plot([x['complexo_fim'] for x in b], ms, '--', color=INK, lw=1)
    plt.xscale('log')
    plt.xlim(hs[0], hs[-1])

    plt.text(1.5e-4, 0.3, 'polos reais positivos\n(monótono, sem sobressinal)',
             color=INK, fontsize=9)
    plt.text(0.06, 0.62, 'polos complexos\n(oscila: sobressinal)', color=INK, fontsize=9)
    plt.annotate('polo real negativo\n(zigue-zague)', xy=(1.75, 0.05), xytext=(0.012, 0.1),
                 fontsize=8, color=INK,
                 arrowprops={'arrowstyle': '->', 'color': INK, 'lw': 1})
    plt.text(4.6, 0.45, 'instável', color=INK, fontsize=9, rotation=90)

    for name, a in TOY['curvaturas'].items():
        h = TOY['r'] * a
        plt.plot(h, TOY['alpha'], 'o', ms=8, mfc='white', mec=RED, mew=2)
        plt.annotate(f'toy {name}\n(a = {a:g})', (h, TOY['alpha']), textcoords='offset points',
                     xytext=(-10, -34), fontsize=9, color=INK, ha='center')

    cb = plt.colorbar(mesh)
    cb.set_label('raio espectral ρ (menor = converge mais rápido)')
    plt.xlabel('h = taxa de aprendizado × curvatura')
    plt.ylabel('coeficiente de momento α')
    plt.title('SGD-Momentum em tempo discreto: regimes de amortecimento', fontsize=11)
    style_axes()
    plt.tight_layout()
    path = os.path.join(out_dir, 'regimes_momentum.png')
    plt.savefig(path, dpi=200)
    plt.close()
    return path


# ----------------------------------------------------------------------------
# Toy: comportamento observado (classe de otimizador real, coordenada escalar)
# ----------------------------------------------------------------------------
def run_toy_coordinate(a, Kd, theta0=-10.0):
    """O toy de De Jong e separavel: cada coordenada e uma quadratica escalar
    com curvatura a = 2*coef (0.2 para x1 e 4 para x2)."""
    theta = torch.tensor([theta0], dtype=torch.float64, requires_grad=True)
    opt = SGD_PID_Paper([theta], lr=TOY['r'], alpha=TOY['alpha'], Kd=Kd)
    traj = [theta0]
    for _ in range(TOY['steps']):
        opt.zero_grad()
        (0.5 * a * theta ** 2).sum().backward()
        opt.step()
        traj.append(theta.item())
        if not np.isfinite(traj[-1]) or abs(traj[-1]) > 1e6:
            break
    traj = np.array(traj)
    diverged = (not np.isfinite(traj[-1])) or abs(traj[-1]) > 1e6
    return {'divergiu': bool(diverged),
            'sobressinal': None if diverged else overshoot(traj, 0.0),
            'acomodacao_2pct': None if diverged else settling_step(traj, 0.0)}


def toy_validation():
    rows = []
    for name, a in TOY['curvaturas'].items():
        for Kd in TOY['Kd']:
            A = pid_paper_matrix(TOY['r'], a, TOY['alpha'], Kd)
            z = poles(A)
            rows.append({'coordenada': name, 'a': a, 'Kd': Kd,
                         'rho_previsto': spectral_radius(A),
                         'polos_complexos': bool(np.any(np.abs(z.imag) > 1e-12)),
                         'previsao': 'diverge' if spectral_radius(A) > 1 else 'converge',
                         **run_toy_coordinate(a, Kd)})
    for r in rows:
        r['acerto'] = (r['previsao'] == 'diverge') == r['divergiu']
    return rows


# ----------------------------------------------------------------------------
# Figura 2: PID da Eq. (10) do artigo
# ----------------------------------------------------------------------------
def fig_pid_paper(out_dir, rows):
    r, alpha = TOY['r'], TOY['alpha']
    a_s = np.logspace(-2, 3, 350)
    kds = np.linspace(-1.5, 1.5, 301)
    R = rho_grid(lambda a, kd: pid_paper_matrix(r, a, alpha, kd), a_s, kds)
    R0 = rho_grid(lambda a, kd: pid_paper_matrix(r, a, alpha, 0.0), a_s, kds)
    delta = np.ma.masked_where((R >= 1) | (R0 >= 1), R - R0)

    plt.figure(figsize=(13, 5))

    plt.subplot(1, 2, 1)
    mesh = draw_rho(a_s, kds, R, vmin=0.5)
    plt.xscale('log')
    for row in rows:
        diverged = row['divergiu']
        plt.plot(row['a'], row['Kd'], 'x' if diverged else 'o', ms=8,
                 color=RED if diverged else 'white', mec=RED if diverged else INK,
                 mew=2 if diverged else 1.2)
    plt.axhline(0, color=MUTED, lw=0.8, ls=':')
    plt.plot([], [], 'o', color='white', mec=INK, label='toy: convergiu')
    plt.plot([], [], 'x', color=RED, mew=2, label='toy: divergiu')
    plt.plot([], [], '-', color=INK, label='fronteira ρ = 1 (teoria)')
    plt.legend(loc='lower left', fontsize=8, framealpha=0.9)
    cb = plt.colorbar(mesh)
    cb.set_label('raio espectral ρ')
    plt.xlabel('curvatura a da direção')
    plt.ylabel('Kd (sinal como na Eq. 10: +Kd·D)')
    plt.title(f'(a) Estabilidade do PID da Eq. (10), r = {r}, α = {alpha}', fontsize=11)
    style_axes()

    plt.subplot(1, 2, 2)
    lim = 0.05                                   # escala saturada para ver variacoes pequenas
    mesh2 = plt.pcolormesh(a_s, kds, delta, cmap='RdBu_r',
                           norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim), shading='auto')
    plt.pcolormesh(a_s, kds, np.ma.masked_where(~((R >= 1) | (R0 >= 1)), np.ones_like(R)),
                   cmap=ListedColormap([UNSTABLE_GRAY]), shading='auto')
    plt.contour(a_s, kds, R, levels=[1.0], colors=INK, linewidths=1.6)
    plt.xscale('log')
    for name, a in TOY['curvaturas'].items():
        plt.axvline(a, color=MUTED, lw=0.8, ls=':')
        plt.text(a * 1.08, 1.35, f'toy {name}', fontsize=9, color=INK)
    plt.axhline(0, color=MUTED, lw=0.8, ls=':')
    cb2 = plt.colorbar(mesh2, extend='both')
    cb2.set_label('ρ(PID) − ρ(momentum)   (azul: o termo D acelera)')
    plt.xlabel('curvatura a da direção')
    plt.ylabel('Kd')
    plt.title('(b) Onde o termo derivativo ajuda ou atrapalha', fontsize=11)
    style_axes()

    plt.tight_layout()
    path = os.path.join(out_dir, 'mapa_pid_artigo.png')
    plt.savefig(path, dpi=200)
    plt.close()
    return path


# ----------------------------------------------------------------------------
# Figura 3: PID na forma do codigo publico (a usada no MNIST)
# ----------------------------------------------------------------------------
def fig_pid_code(out_dir, lambda_max=None, lr=0.01, wd=1e-4, Kp=1.0, Ki=3.0, m=0.9):
    hs = np.logspace(-4, 1, 350)
    kds = np.concatenate([-np.logspace(4, -1, 160), [0.0], np.logspace(-1, 4, 160)])
    R = rho_grid(lambda h, kd: pid_separated_matrix(h, m, Kp, Ki, kd), hs, kds)

    plt.figure(figsize=(7.5, 5))
    plt.subplot(1, 1, 1)
    mesh = draw_rho(hs, kds, R, vmin=0.5)
    plt.xscale('log')
    plt.yscale('symlog', linthresh=1.0)
    plt.axhline(100, color=RED, lw=1.2, ls='--')
    plt.text(1.3e-4, 140, 'Kd = 100 (código dos autores)', color=RED, fontsize=9)
    plt.axhline(0, color=MUTED, lw=0.8, ls=':')
    plt.axvline(2 * (1 + m), color=MUTED, lw=1, ls='-.')
    plt.text(2 * (1 + m) * 0.75, -3000, 'limite do\nmomentum\nh = 2(1+α)', fontsize=8,
             color=INK, ha='right')
    if lambda_max:
        h_mnist = lr * (lambda_max + wd)
        plt.axvline(h_mnist, color=INK, lw=1.4)
        plt.text(h_mnist * 1.1, 3000, f'MNIST: lr·λmax = {h_mnist:.3g}', fontsize=9, color=INK,
                 va='top')
    cb = plt.colorbar(mesh)
    cb.set_label('raio espectral ρ')
    plt.xlabel('h = taxa de aprendizado × curvatura')
    plt.ylabel(f'Kd  (forma do código; Kp = {Kp:g}, Ki = {Ki:g}, m = {m:g})')
    plt.title('PID na forma do código público: margem de estabilidade', fontsize=11)
    style_axes()
    plt.tight_layout()
    path = os.path.join(out_dir, 'mapa_pid_codigo.png')
    plt.savefig(path, dpi=200)
    plt.close()

    # margem: maior h estavel em funcao de Kd (vira tabela no README/poster)
    margin = {}
    for kd in [0.0, 30.0, 100.0, 300.0, -30.0, -100.0]:
        margin[kd] = largest_stable(
            lambda h, kd=kd: spectral_radius(pid_separated_matrix(h, m, Kp, Ki, kd)))
    return path, margin


def to_markdown(rows):
    head = ('| coord. | a | Kd | ρ previsto | polos | previsão | observado | sobressinal '
            '| acomodação (passos) |\n|---|---|---|---|---|---|---|---|---|\n')
    lines = []
    for r in rows:
        obs = 'divergiu' if r['divergiu'] else 'convergiu'
        os_ = '—' if r['sobressinal'] is None else f"{r['sobressinal']:.3f}"
        st = '—' if r['acomodacao_2pct'] is None else str(r['acomodacao_2pct'])
        lines.append(f"| {r['coordenada']} | {r['a']:g} | {r['Kd']:+g} | {r['rho_previsto']:.4f} "
                     f"| {'complexos' if r['polos_complexos'] else 'reais'} | {r['previsao']} "
                     f"| {obs} {'✔' if r['acerto'] else '✘'} | {os_} | {st} |")
    return head + '\n'.join(lines) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--lambda-max', type=float, default=None,
                   help='curvatura maxima medida no MNIST (saida do scripts/sweep.py)')
    p.add_argument('--out-root', default='./results')
    args = p.parse_args()
    try:                                   # simbolos (ρ, ✔, x₁) no console do Windows
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass

    out_dir = os.path.join(args.out_root, 'estabilidade_discreta', timestamp())
    os.makedirs(out_dir, exist_ok=True)

    rows = toy_validation()
    figs = [fig_momentum(out_dir), fig_pid_paper(out_dir, rows)]
    path3, margin = fig_pid_code(out_dir, args.lambda_max)
    figs.append(path3)

    md = to_markdown(rows)
    with open(os.path.join(out_dir, 'validacao_toy.md'), 'w', encoding='utf-8') as f:
        f.write(md)
    save_json({'args': vars(args), 'git': git_info(), 'toy': TOY, 'validacao': rows,
               'margem_h_max_por_Kd_forma_codigo': {str(k): v for k, v in margin.items()}},
              os.path.join(out_dir, 'results.json'))

    hits = sum(r['acerto'] for r in rows)
    print('Validação no toy (teoria linear em tempo discreto x otimizador real):\n')
    print(md)
    print(f'Previsão de convergir/divergir correta em {hits}/{len(rows)} casos.\n')
    print('Forma do código (Kp=1, Ki=3, m=0.9): maior h = lr·a estável para cada Kd')
    for kd, h in margin.items():
        txt = f'h_max = {h:.4g}' if h else 'instável'
        print(f'  Kd = {kd:+7.1f}  ->  {txt}')
    print(f'  (SGD-Momentum, m=0.9: h_max = {2 * 1.9:.4g})')
    print('\nFiguras:\n  ' + '\n  '.join(figs))


if __name__ == '__main__':
    main()
