"""Estilo sobrio e consistente para as figuras de relatorio/poster.

Regras (as mesmas em todas as figuras):
  - fundo branco, sem titulo dentro da figura (quem explica e a legenda do LaTeX);
  - cinza para a referencia (baseline), UMA cor de destaque para o metodo
    estudado; variacoes do mesmo metodo = tons da mesma cor ou tracejado;
  - rotulos diretos no fim das curvas em vez de caixa de legenda, quando cabem;
  - status (divergiu) por forma E cor, nunca so por cor;
  - grade leve, sem molduras superior/direita;
  - PDF vetorial (nitido no LaTeX em qualquer zoom) + PNG para conferir.
"""

import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

INK = '#1f1f1d'          # texto e linhas principais
MUTED = '#6b6a66'        # texto secundario, eixos
GRID = '#e6e5e1'
BASE = '#8a8985'         # serie de referencia (baseline)
ACCENT = '#2a78d6'       # metodo estudado
ACCENT_2 = '#eb6834'     # segunda serie de destaque (usar com parcimonia)
BAD = '#c8383a'          # divergiu / instavel
FILL_UNSTABLE = '#ecebe7'
BLUES = ['#9cc2ef', '#5f9ce3', '#2a78d6', '#1a56a3']   # tons ordenados (claro -> escuro)

WIDTH_FULL = 6.3         # polegadas: largura de texto de um A4 com margens de 2,5 cm
WIDTH_HALF = 3.1


def use_style():
    plt.rcParams.update({
        'font.size': 9, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'legend.fontsize': 8, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
        'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID,
        'grid.linewidth': 0.6, 'lines.linewidth': 1.6, 'legend.frameon': False,
        'figure.dpi': 150, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03,
        'pdf.fonttype': 42,
    })


def panel_tag(letter):
    """(a), (b)... no canto superior esquerdo do painel atual."""
    plt.text(-0.02, 1.02, f'({letter})', transform=plt.gca().transAxes, fontsize=9,
             color=INK, ha='right', va='bottom', fontweight='bold')


def end_label(x, y, text, color, dy=0.0, **kw):
    """Rotulo direto no fim de uma curva."""
    plt.annotate(text, (x, y), xytext=(4, dy), textcoords='offset points', color=color,
                 fontsize=8, va='center', ha='left', **kw)


def save(out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    pdf = os.path.join(out_dir, f'{name}.pdf')
    plt.savefig(pdf)
    plt.savefig(os.path.join(out_dir, f'{name}.png'), dpi=200)
    plt.close()
    return pdf
