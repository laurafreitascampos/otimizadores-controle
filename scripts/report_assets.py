"""Gera as figuras (PDF + PNG) e tabelas (LaTeX) do relatorio a partir dos resultados.

Nada e digitado a mao: as figuras analiticas sao recalculadas a partir das
matrizes de malha fechada, e as experimentais saem dos results.json /
sweep.json gerados pelo treino. Cada tabela leva um comentario com o arquivo
e o commit de origem.

Uso (todas as entradas experimentais sao opcionais):
    python scripts/report_assets.py
    python scripts/report_assets.py --fig4 results/an2018_fig4_mnist_repo/<ts>/results.json \
        results/an2018_h2_passo_equivalente/<ts>/results.json \
        --sweep-lr results/mapa_mnist_momentum_lr/<ts>/sweep.json \
        --sweep-kd results/mapa_mnist_pid_kd/<ts>/sweep.json
Saida: docs/relatorio/figuras/*.pdf|png e docs/relatorio/tabelas/*.tex
"""

import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch

from optctrl import plotting as P
from optctrl.analysis import (
    largest_stable,
    momentum_boundaries,
    momentum_matrix,
    pid_paper_matrix,
    pid_separated_matrix,
    predicted_rho,
    spectral_radius,
)
from optctrl.optimizers import SGD_PID_Paper

plt = P.plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_module(name, rel_path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


stab = _load_module('stability_map', 'experiments/estabilidade_discreta/stability_map.py')
cstats = _load_module('compare_stats', 'scripts/compare_stats.py')


def rho_grid(builder, xs, ys):
    mats = np.array([[builder(x, y) for x in xs] for y in ys])
    eig = np.linalg.eigvals(mats.reshape(-1, *mats.shape[-2:]))
    return np.abs(eig).max(axis=-1).reshape(len(ys), len(xs))


def fmt_br(v, nd=2):
    return f'{v:.{nd}f}'.replace('.', ',')


# ============================================================================
# Figuras analiticas (deterministicas)
# ============================================================================
def scalar_run(a, Kd, steps, r=0.01, alpha=0.9, theta0=-10.0):
    theta = torch.tensor([theta0], dtype=torch.float64, requires_grad=True)
    opt = SGD_PID_Paper([theta], lr=r, alpha=alpha, Kd=Kd)
    traj = [theta0]
    for _ in range(steps):
        opt.zero_grad()
        (0.5 * a * theta ** 2).sum().backward()
        opt.step()
        traj.append(theta.item())
        if abs(traj[-1]) > 1e3:
            break
    return np.array(traj)


def fig_toy(out):
    """Resposta no tempo da coordenada de alta curvatura do toy (x2, a = 4)."""
    steps, a = 150, 4.0
    mom = scalar_run(a, 0.0, steps)
    plt.figure(figsize=(P.WIDTH_FULL, 2.5))
    for k, (sign, title) in enumerate([(+1, 'sinal do artigo: $+K_d D$'),
                                       (-1, 'sinal invertido: $-K_d D$')]):
        plt.subplot(1, 2, k + 1)
        plt.axhline(0, color=P.MUTED, lw=0.8)
        plt.plot(mom, color=P.BASE, lw=1.8, label='momentum ($K_d$ = 0)')
        for kd, color in zip([0.1, 0.5, 0.93], P.BLUES[1:]):
            tr = scalar_run(a, sign * kd, steps)
            diverged = np.abs(tr).max() > 50 or len(tr) < steps + 1
            tr = tr[: (np.nonzero(np.abs(tr) > 60)[0][:1].tolist() or [len(tr)])[0]]
            lab = '$K_d$ = ' + f'{sign * kd:+.2f}'.replace('.', ',')
            plt.plot(tr, color=color, label=lab + (' (diverge)' if diverged else ''))
        plt.ylim(-16, 12)
        plt.xlim(0, steps)
        plt.xlabel('passo $t$')
        if k == 0:
            plt.ylabel('$x_2$ (curvatura alta)')
        plt.title(title, fontsize=9, color=P.INK, loc='left')
        plt.legend(loc='lower right', fontsize=7, handlelength=1.6)
        P.panel_tag('ab'[k])
    plt.tight_layout(w_pad=2)
    return P.save(out, 'toy_resposta_x2')


def fig_regimes(out):
    hs = np.logspace(-4, 1, 500)
    ms = np.linspace(0.0, 0.99, 300)
    R = rho_grid(lambda h, m: momentum_matrix(h, m), hs, ms)
    H, M = np.meshgrid(hs, ms)
    s = np.sqrt(M)
    regime = np.select([H < (1 - s) ** 2, H < (1 + s) ** 2, H < 2 * (1 + M)], [0, 1, 2], 3)

    plt.figure(figsize=(P.WIDTH_FULL, 3.0))
    plt.subplot(1, 1, 1)
    colors = ['#ffffff', '#eaf1fb', '#fbf0e6', P.FILL_UNSTABLE]
    plt.contourf(H, M, regime, levels=[-0.5, 0.5, 1.5, 2.5, 3.5],
                 colors=colors)
    b = [momentum_boundaries(m) for m in ms]
    plt.plot([x['complexo_inicio'] for x in b], ms, color=P.INK, lw=0.9)
    plt.plot([x['complexo_fim'] for x in b], ms, color=P.INK, lw=0.9)
    plt.plot(2 * (1 + ms), ms, color=P.INK, lw=1.4)
    plt.contour(H, M, np.ma.masked_where(R >= 1, R), levels=[0.9],
                colors=P.MUTED, linewidths=0.7, linestyles=':')
    plt.text(2.2e-3, 0.30, '$\\rho$ = 0,9', fontsize=7, color=P.MUTED, rotation=-62)
    plt.xscale('log')
    plt.xlim(hs[0], hs[-1])
    plt.ylim(0, 0.99)
    plt.grid(False)
    plt.text(1.5e-4, 0.35, 'monótono\n(polos reais positivos)', fontsize=8, color=P.INK)
    plt.text(0.05, 0.55, 'oscilatório\n(polos complexos)', fontsize=8, color=P.INK)
    plt.annotate('zigue-zague\n(polo negativo)', xy=(1.7, 0.04), xytext=(0.03, 0.08),
                 fontsize=8, color=P.INK, arrowprops={'arrowstyle': '-', 'color': P.MUTED,
                                                      'lw': 0.7})
    plt.text(4.8, 0.5, 'instável', fontsize=8, color=P.INK, rotation=90, va='center')
    for name, a in stab.TOY['curvaturas'].items():
        h = stab.TOY['r'] * a
        plt.plot(h, stab.TOY['alpha'], 'o', ms=6, mfc='white', mec=P.ACCENT, mew=1.6)
        plt.annotate(f'toy ${name[0]}_{name[1:].translate(str.maketrans("₁₂", "12"))}$',
                     (h, stab.TOY['alpha']), xytext=(0, -12), textcoords='offset points',
                     ha='center', fontsize=8, color=P.ACCENT)
    plt.xlabel('$h$ = taxa de aprendizado × curvatura')
    plt.ylabel('momento $\\alpha$')
    plt.tight_layout()
    return P.save(out, 'regimes_momentum')


def fig_pid_paper(out, rows):
    r, alpha = stab.TOY['r'], stab.TOY['alpha']
    a_s = np.logspace(-2, 3, 400)
    kds = np.linspace(-1.5, 1.5, 361)
    R = rho_grid(lambda a, kd: pid_paper_matrix(r, a, alpha, kd), a_s, kds)
    R0 = rho_grid(lambda a, kd: pid_paper_matrix(r, a, alpha, 0.0), a_s, kds)
    stable = (R < 1) & (R0 < 1)
    delta = np.where(stable, R - R0, np.nan)

    plt.figure(figsize=(P.WIDTH_FULL, 3.0))
    plt.subplot(1, 1, 1)
    plt.contourf(a_s, kds, (R >= 1).astype(float), levels=[0.5, 1.5], colors=[P.FILL_UNSTABLE])
    plt.contour(a_s, kds, R, levels=[1.0], colors=P.INK, linewidths=1.3)
    plt.contour(a_s, kds, delta, levels=[0.0], colors=P.ACCENT, linewidths=1.0,
                linestyles='--')
    plt.axhline(0, color=P.MUTED, lw=0.6)
    plt.xscale('log')
    plt.grid(False)
    for row in rows:
        if row['divergiu']:
            plt.plot(row['a'], row['Kd'], 'x', color=P.BAD, ms=7, mew=2)
        else:
            plt.plot(row['a'], row['Kd'], 'o', ms=5.5, mfc='white', mec=P.INK, mew=1.1)
    plt.text(60, 0.6, 'instável', fontsize=8, color=P.INK)
    plt.text(0.6, -0.33, 'D acelera', fontsize=8, color=P.ACCENT)
    plt.text(0.5, 0.33, 'D atrasa', fontsize=8, color=P.ACCENT)
    plt.text(0.012, -1.3, 'D atrasa', fontsize=8, color=P.ACCENT)
    plt.plot([], [], 'o', mfc='white', mec=P.INK, label='toy: convergiu')
    plt.plot([], [], 'x', color=P.BAD, mew=2, label='toy: divergiu')
    plt.plot([], [], color=P.INK, lw=1.3, label='fronteira $\\rho = 1$')
    plt.plot([], [], color=P.ACCENT, lw=1.0, ls='--', label='$\\rho_{PID} = \\rho_{momentum}$')
    plt.text(0.012, 0.9, 'D acelera', fontsize=8, color=P.ACCENT)
    plt.legend(loc='upper left', fontsize=7.5, bbox_to_anchor=(1.01, 1.0))
    plt.xlabel('curvatura $a$ da direção')
    plt.ylabel('$K_d$ (sinal da Eq. 10)')
    plt.tight_layout()
    return P.save(out, 'mapa_pid_artigo')


def h_max_curve(kds, m=0.9, Kp=1.0, Ki=3.0):
    out = []
    for kd in kds:
        h = largest_stable(lambda x, kd=kd: spectral_radius(pid_separated_matrix(x, m, Kp, Ki, kd)),
                           n=300)
        out.append(np.nan if h is None else h)
    return np.array(out)


def fig_margin(out, lambda_max=None, lr=0.01, wd=1e-4):
    kds = np.concatenate([-np.logspace(3, -1, 70), [0.0], np.logspace(-1, np.log10(3000), 90)])
    hmax = h_max_curve(kds)
    plt.figure(figsize=(P.WIDTH_FULL, 2.6))
    plt.subplot(1, 1, 1)
    plt.plot(kds, hmax, color=P.ACCENT, lw=1.8)
    plt.axhline(3.8, color=P.BASE, lw=1.2, ls='--')
    plt.text(-900, 4.6, 'SGD-Momentum ($\\alpha$ = 0,9): 3,8', fontsize=8, color=P.BASE)
    plt.ylim(8e-4, 12)
    for kd, txt, dy in [(0.0, '$K_d$ = 0', 8), (100.0, '$K_d$ = 100 (autores)', 8)]:
        hk = h_max_curve([kd])[0]
        plt.plot(kd, hk, 'o', color=P.ACCENT, ms=5)
        plt.annotate(f'{txt}: {fmt_br(hk, 3)}', (kd, hk), xytext=(5, dy),
                     textcoords='offset points', fontsize=8, color=P.INK)
    if lambda_max:
        h_mnist = lr * (lambda_max + wd)
        plt.axhline(h_mnist, color=P.BAD, lw=1.0, ls=':')
        plt.text(0.3, h_mnist * 0.55, f'MNIST, lr = 0,01: lr·λmax = {fmt_br(h_mnist, 3)}',
                 fontsize=8, color=P.BAD)
    plt.xscale('symlog', linthresh=1.0)
    plt.yscale('log')
    plt.xlabel('$K_d$ (forma do código; $K_p$ = 1, $K_i$ = 3, $m$ = 0,9)')
    plt.ylabel('maior $h$ estável')
    plt.tight_layout()
    return P.save(out, 'margem_pid_codigo')


# ============================================================================
# Figuras e tabelas experimentais
# ============================================================================
LABELS = {'sgd_momentum': 'SGD-Momentum', 'pid_repo': 'PID ($K_d$ = 100)',
          'pid_repo_kd0': 'PID sem D ($K_d$ = 0)',
          'sgd_momentum_lr_eq': 'Momentum, mesmo ganho DC do PID',
          'pid_paper_eq14': 'PID, $K_d$ da Eq. (14)'}
STYLE = {'sgd_momentum': {'color': P.BASE}, 'pid_repo': {'color': P.ACCENT},
         'pid_repo_kd0': {'color': P.ACCENT, 'ls': '--'},
         'sgd_momentum_lr_eq': {'color': P.ACCENT_2}}


def load_results(paths):
    merged, meta = {}, []
    for p in paths:
        with open(p, encoding='utf-8') as f:
            res = json.load(f)
        meta.append((p, res.get('git', {}).get('commit'), res.get('git', {}).get('dirty')))
        for label, hists in res['per_seed'].items():
            merged.setdefault(label, hists)
    return merged, meta


def fig_fig4(paths, out):
    runs, meta = load_results(paths)
    plt.figure(figsize=(P.WIDTH_FULL, 2.5))
    for k, (key, ylabel, scale) in enumerate([('eval_acc', 'acurácia de teste (%)', 100),
                                              ('train_loss', 'perda de treino', 1)]):
        plt.subplot(1, 2, k + 1)
        for label, hists in runs.items():
            ok = [h[key] for h in hists if not h['diverged'] and h[key]]
            if not ok or label not in STYLE:
                continue
            n = min(len(c) for c in ok)
            arr = np.array([c[:n] for c in ok]) * scale
            mu = arr.mean(0)
            sd = arr.std(0, ddof=1) if len(arr) > 1 else np.zeros_like(mu)
            x = np.arange(1, n + 1)
            st = STYLE[label]
            plt.plot(x, mu, color=st['color'], ls=st.get('ls', '-'),
                     label=f"{LABELS.get(label, label)} (n = {len(arr)})")
            top = mu + sd if key != 'eval_acc' else np.minimum(mu + sd, 100.0)
            plt.fill_between(x, mu - sd, top, color=st['color'], alpha=0.15, lw=0)
        plt.xlabel('época')
        plt.ylabel(ylabel)
        if key == 'train_loss':
            plt.yscale('log')
        else:
            lo, hi = plt.ylim()
            plt.ylim(max(lo, 95.5), min(hi, 100.2))
        P.panel_tag('ab'[k])
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.figlegend(handles, labels, loc='lower center', ncol=len(labels), fontsize=7.5,
                  bbox_to_anchor=(0.5, -0.06))
    plt.tight_layout(w_pad=2)
    path = P.save(out, 'fig4_mnist')
    return path, meta


def table_fig4(paths, out_tab, ref='sgd_momentum'):
    runs, protocol = cstats.load(paths)
    if ref not in runs:
        return None
    blocks = []
    for metric in ('final_acc', 'epochs_to_target'):
        nd = 2 if cstats.METRICS[metric][2] == 100.0 else 1
        name, rows = cstats.compare(runs, ref, metric)
        for r in rows:
            r['label'] = LABELS.get(r['label'], r['label'])
        blocks.append((name, rows, nd))
    tex = cstats.to_latex(blocks, LABELS[ref])
    tex = tex.replace(r'\$', '$').replace(r'\_d', '_d')          # rotulos ja tem math
    _, meta = load_results(paths)
    header = ''.join(f'% origem: {p} (commit {c}, dirty={d})\n' for p, c, d in meta)
    os.makedirs(out_tab, exist_ok=True)
    path = os.path.join(out_tab, 'fig4.tex')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(header + tex)
    return path


def _floor_and_marks(summ, key, value_of):
    accs = [s['acc_media'] for s in summ if s['acc_media'] is not None
            and s['status'] != 'divergiu']
    return (min(accs) if accs else 0.5) * 100 - 2


def fig_sweep_lr(path, out):
    with open(path, encoding='utf-8') as f:
        res = json.load(f)
    grid = res['config']['sweep']['grid']
    wd = res['config']['sweep'].get('fixed', {}).get('weight_decay', 0.0)
    lam = res.get('lambda_max')
    ms = sorted(grid['momentum'])
    floor = _floor_and_marks(res['summary'], 'lr', None)
    plt.figure(figsize=(P.WIDTH_FULL, 2.8))
    plt.subplot(1, 1, 1)
    shades = P.BLUES if len(ms) <= 4 else plt.get_cmap('Blues')(np.linspace(0.35, 0.95, len(ms)))
    for m, color in zip(ms, shades):
        pts = sorted([s for s in res['summary'] if s['params']['momentum'] == m],
                     key=lambda s: s['params']['lr'])
        xs = [s['params']['lr'] for s in pts]
        ys = [100 * s['acc_media'] if (s['acc_media'] is not None and s['status'] != 'divergiu')
              else np.nan for s in pts]
        plt.plot(xs, ys, '-', color=color, lw=1.4, label=f'$\\alpha$ = {fmt_br(m, 2)}')
        for s, y in zip(pts, ys):
            x = s['params']['lr']
            if s['status'] == 'divergiu':
                plt.plot(x, floor, 'x', color=P.BAD, ms=6, mew=1.6)
            elif s['status'] == 'nao_convergiu':
                plt.plot(x, floor, 's', mfc='white', mec=P.BAD, ms=5)
            elif s['status'] == 'instavel':
                plt.plot(x, y, 'o', mfc='white', mec=color, ms=5, mew=1.3)
            else:
                plt.plot(x, y, 'o', color=color, ms=4)
        if lam:
            lr_c = 2 * (1 + m) / (lam + wd)                       # fronteira linear
            plt.plot(lr_c, 1.0, 'v', color=color, ms=7, transform=plt.gca().get_xaxis_transform(),
                     clip_on=False)
    plt.axhline(floor, color=P.BAD, lw=0.6, ls=':')
    plt.text(1.01, floor, 'divergiu (×) /\nnão convergiu (□)', fontsize=7, color=P.BAD,
             va='center', transform=plt.gca().get_yaxis_transform())
    plt.xscale('log')
    plt.xlabel('taxa de aprendizado')
    plt.ylabel('acurácia de validação (%)')
    plt.legend(loc='upper left', fontsize=7, bbox_to_anchor=(1.01, 1.0), title='momento',
               title_fontsize=7)
    if lam:
        plt.figtext(0.01, -0.02, f'▼ fronteira prevista lr = 2(1+α)/λmax, λmax = {fmt_br(lam, 3)} '
                    '(na inicialização)    ○ vazio: perda subiu em alguma época',
                    fontsize=7, color=P.MUTED)
    plt.tight_layout()
    return P.save(out, 'varredura_lr_momentum')


def fig_sweep_kd(path, out):
    with open(path, encoding='utf-8') as f:
        res = json.load(f)
    sw = res['config']['sweep']
    lam = res.get('lambda_max')
    summ = sorted(res['summary'], key=lambda s: s['params']['Kd'])
    floor = _floor_and_marks(summ, 'Kd', None)
    plt.figure(figsize=(P.WIDTH_FULL, 2.6))
    plt.subplot(1, 1, 1)
    xs = [s['params']['Kd'] for s in summ]
    if lam:
        fine = np.concatenate([-np.logspace(np.log10(max(abs(min(xs)), 1)), -1, 200), [0.0],
                               np.logspace(-1, np.log10(max(max(xs), 1)), 200)])
        rho = np.array([predicted_rho(sw['optimizer'], {**sw['fixed'], 'Kd': k}, lam)
                        for k in fine])
        unstable = rho >= 1
        edges = np.diff(np.concatenate([[0], unstable.astype(int), [0]]))
        for i0, i1 in zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]):
            plt.axvspan(fine[i0], fine[i1 - 1], color=P.FILL_UNSTABLE, lw=0, zorder=0)
    for r in res['runs']:
        x = r['params']['Kd']
        if r['status'] in ('divergiu', 'nao_convergiu'):
            plt.plot(x, floor, 'x' if r['status'] == 'divergiu' else 's', color=P.BAD,
                     mfc='white', ms=6, mew=1.6)
        else:
            plt.plot(x, 100 * r['final_eval_acc'], 'o', color=P.ACCENT, alpha=0.35, ms=3.5)
    ym = [100 * s['acc_media'] if (s['acc_media'] is not None and s['status'] != 'divergiu')
          else np.nan for s in summ]
    plt.plot(xs, ym, '-o', color=P.ACCENT, ms=4, lw=1.4)
    plt.axhline(floor, color=P.BAD, lw=0.6, ls=':')
    plt.text(1.01, floor, 'divergiu (×) /\nnão convergiu (□)', fontsize=7, color=P.BAD,
             va='center', transform=plt.gca().get_yaxis_transform())
    plt.xscale('symlog', linthresh=max(1e-3, min(abs(x) for x in xs if x != 0)))
    plt.xlabel('$K_d$ (forma do código)')
    plt.ylabel('acurácia de validação (%)')
    if lam:
        plt.figtext(0.01, -0.02, f'faixa cinza: instável pela teoria linear (λmax = '
                    f'{fmt_br(lam, 3)}, na inicialização)    pontos claros: sementes',
                    fontsize=7, color=P.MUTED)
    plt.tight_layout()
    return P.save(out, 'varredura_kd')


def table_validation(rows, out_tab):
    lines = ['% gerado por scripts/report_assets.py (determinístico)',
             r'\begin{tabular}{ccrcccrr}', r'\toprule',
             r'Coord. & $a$ & $K_d$ & $\rho$ previsto & Polos & Previsão & Sobressinal '
             r'& Acomodação \\', r'\midrule']
    def tex(v, nd, sign=False):
        txt = f'{v:+.{nd}f}' if sign else f'{v:.{nd}f}'
        return '$' + txt.replace('.', '{,}') + '$'

    for r in rows:
        coord = '$x_1$' if '₁' in r['coordenada'] else '$x_2$'
        prev = 'diverge' if r['previsao'] == 'diverge' else 'converge'
        mark = r'\checkmark' if r['acerto'] else r'$\times$'
        os_ = '---' if r['sobressinal'] is None else tex(r['sobressinal'], 2)
        st = '---' if r['acomodacao_2pct'] is None else str(r['acomodacao_2pct'])
        polos = 'complexos' if r['polos_complexos'] else 'reais'
        lines.append(f"{coord} & {tex(r['a'], 1)} & {tex(r['Kd'], 2, True)} & "
                     f"{tex(r['rho_previsto'], 4)} & {polos} & {prev} {mark} & {os_} & {st} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    path = os.path.join(out_tab, 'validacao_toy.tex')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--fig4', nargs='+', default=None, help='results.json da Fig. 4 (e da H2)')
    p.add_argument('--sweep-lr', default=None)
    p.add_argument('--sweep-kd', default=None)
    p.add_argument('--lambda-max', type=float, default=None)
    p.add_argument('--out', default=os.path.join(ROOT, 'docs', 'relatorio'))
    args = p.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass

    P.use_style()
    fig_dir, tab_dir = os.path.join(args.out, 'figuras'), os.path.join(args.out, 'tabelas')
    os.makedirs(tab_dir, exist_ok=True)

    lam = args.lambda_max
    for sw in (args.sweep_lr, args.sweep_kd):
        if lam is None and sw:
            with open(sw, encoding='utf-8') as f:
                lam = json.load(f).get('lambda_max')

    rows = stab.toy_validation()
    made = [fig_toy(fig_dir), fig_regimes(fig_dir), fig_pid_paper(fig_dir, rows),
            fig_margin(fig_dir, lam), table_validation(rows, tab_dir)]
    if args.fig4:
        path, _ = fig_fig4(args.fig4, fig_dir)
        made += [path, table_fig4(args.fig4, tab_dir)]
    if args.sweep_lr:
        made.append(fig_sweep_lr(args.sweep_lr, fig_dir))
    if args.sweep_kd:
        made.append(fig_sweep_kd(args.sweep_kd, fig_dir))
    print('Gerados:\n  ' + '\n  '.join(m for m in made if m))


if __name__ == '__main__':
    main()
