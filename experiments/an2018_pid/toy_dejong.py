"""Figura 2 de An et al. (2018): overshoot do momentum na 1a funcao de De Jong.

    f(x) = 0.1*x1^2 + 2*x2^2,   x0 = (-10, -10),   minimo em (0, 0)

Gradiente deterministico (sem ruido de mini-lote), entao isto e a resposta
"pura" do sistema dinamico otimizador + funcao quadratica.

O artigo NAO informa lr nem alpha do toy; os padroes abaixo foram escolhidos
para deixar o overshoot do momentum visivel e estao expostos como argumentos.

Para cada Kd, roda o SGD-Momentum (Kd = 0) e o PID na forma da Eq. (10)
(SGD_PID_Paper) com o Kd escrito no artigo E com o sinal invertido. Qual
linha reproduz a Figura 2 responde a pergunta do sinal do termo D.

Uso:
    python experiments/an2018_pid/toy_dejong.py
    python experiments/an2018_pid/toy_dejong.py --lr 0.02 --alpha 0.95 --kd 0.1 0.5 0.93
"""

import argparse
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from optctrl.metrics import overshoot, settling_step  # noqa: E402
from optctrl.optimizers import SGD_PID_Paper  # noqa: E402
from optctrl.utils import git_info, save_json, timestamp  # noqa: E402

X0 = (-10.0, -10.0)
COEF = torch.tensor([0.1, 2.0], dtype=torch.float64)


def dejong(x):
    return (COEF * x ** 2).sum()


def run(lr, alpha, kd, steps):
    x = torch.tensor(X0, dtype=torch.float64, requires_grad=True)
    opt = SGD_PID_Paper([x], lr=lr, alpha=alpha, Kd=kd)
    traj = [x.detach().clone().numpy()]
    for _ in range(steps):
        opt.zero_grad()
        dejong(x).backward()
        opt.step()
        traj.append(x.detach().clone().numpy())
        if not np.isfinite(traj[-1]).all() or np.abs(traj[-1]).max() > 1e6:
            break
    return np.array(traj)


def summarize(traj):
    out = {}
    for i, axis in enumerate(('x1', 'x2')):
        out[f'overshoot_{axis}'] = overshoot(traj[:, i], 0.0)
        out[f'settling_{axis}'] = settling_step(traj[:, i], 0.0)
    out['diverged'] = bool(np.abs(traj[-1]).max() > 1e6 or not np.isfinite(traj[-1]).all())
    out['steps_run'] = len(traj) - 1
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--alpha', type=float, default=0.9)
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--kd', type=float, nargs='+', default=[0.1, 0.5, 0.93],
                   help='valores de Kd da Figura 2')
    p.add_argument('--out-root', default='./results')
    args = p.parse_args()

    momentum = run(args.lr, args.alpha, 0.0, args.steps)
    rows = [('Kd como no artigo (+Kd·D)', +1), ('sinal invertido (−Kd·D)', -1)]

    g1, g2 = np.meshgrid(np.linspace(-11, 8, 200), np.linspace(-11, 8, 200))
    fval = 0.1 * g1 ** 2 + 2 * g2 ** 2

    summary = {'momentum': summarize(momentum), 'pid': []}
    print(f"lr={args.lr} alpha={args.alpha} passos={args.steps}")
    print(f"{'caso':>28} | {'OS x1':>7} {'OS x2':>7} | {'assent. x1':>10} {'assent. x2':>10}")
    s = summary['momentum']
    print(f"{'momentum (Kd=0)':>28} | {s['overshoot_x1']:7.3f} {s['overshoot_x2']:7.3f} | "
          f"{str(s['settling_x1']):>10} {str(s['settling_x2']):>10}")

    n_cols = len(args.kd)
    plt.figure(figsize=(4.2 * n_cols, 8))
    for r, (row_title, sign) in enumerate(rows):
        for c, kd in enumerate(args.kd):
            kd_eff = sign * kd
            pid = run(args.lr, args.alpha, kd_eff, args.steps)
            s = summarize(pid)
            summary['pid'].append({'Kd': kd_eff, **s})
            tag = f"Kd = {kd_eff:+.2f}" + (" (divergiu)" if s['diverged'] else "")
            print(f"{tag:>28} | {s['overshoot_x1']:7.3f} {s['overshoot_x2']:7.3f} | "
                  f"{str(s['settling_x1']):>10} {str(s['settling_x2']):>10}")

            plt.subplot(2, n_cols, r * n_cols + c + 1)
            plt.contourf(g1, g2, fval, levels=20, cmap='coolwarm')
            plt.plot(momentum[:, 0], momentum[:, 1], color='gold', lw=2.5, label='SGD-Momentum')
            out_of_view = np.nonzero(np.abs(pid).max(axis=1) >= 12)[0]
            shown = pid[:out_of_view[0]] if len(out_of_view) else pid   # ate sair do quadro
            plt.plot(shown[:, 0], shown[:, 1], color='red', lw=2, label='PID')
            plt.xlim(-11, 8)
            plt.ylim(-11, 8)
            plt.title(f"{row_title}\n{tag}", fontsize=9)
            if r == 0 and c == 0:
                plt.legend(loc='lower right', fontsize=8)
    plt.tight_layout()

    out_dir = os.path.join(args.out_root, 'an2018_toy_dejong', timestamp())
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, 'trajetorias.png'), dpi=150)
    save_json({'args': vars(args), 'git': git_info(), 'summary': summary},
              os.path.join(out_dir, 'results.json'))
    print(f"\nFigura e resultados em {out_dir}")


if __name__ == '__main__':
    main()
