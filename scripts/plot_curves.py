"""Curvas media +/- desvio a partir de um results.json (formato da Figura 4 de An et al.).

Uso:
    python scripts/plot_curves.py results/<nome>/<timestamp>/results.json
Salva curves.png ao lado do results.json.
"""

import argparse
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PANELS = [('train_loss', 'Perda de treino'), ('eval_loss', 'Perda de avaliação'),
          ('train_acc', 'Acurácia de treino'), ('eval_acc', 'Acurácia de avaliação')]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('results')
    p.add_argument('--out', default=None)
    args = p.parse_args()

    with open(args.results, encoding='utf-8') as f:
        res = json.load(f)

    plt.figure(figsize=(11, 8))
    for k, (key, title) in enumerate(PANELS, start=1):
        plt.subplot(2, 2, k)
        for label, agg in res['aggregate'].items():
            curve = agg['curves'].get(key)
            if not curve:
                continue
            mean, std = np.array(curve['mean']), np.array(curve['std'])
            x = np.arange(1, len(mean) + 1)
            line, = plt.plot(x, mean, label=label)
            if std.any():
                plt.fill_between(x, mean - std, mean + std, alpha=0.2, color=line.get_color())
        plt.title(f"{title} ({res.get('eval_on', 'val')})" if 'eval' in key else title)
        plt.xlabel('época')
        plt.grid(True, alpha=0.3)
        plt.legend()
    plt.suptitle(res['config']['name'])
    plt.tight_layout()

    out = args.out or os.path.join(os.path.dirname(args.results), 'curves.png')
    plt.savefig(out, dpi=150)
    print(f"Figura salva em {out}")


if __name__ == '__main__':
    main()
