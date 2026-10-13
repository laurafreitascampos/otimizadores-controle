"""Compara otimizadores entre sementes com testes estatisticos (results.json do train.py).

Para cada otimizador, contra um de referencia:
  media +/- desvio (amostral), diferenca com IC de 95 %, p do teste PAREADO
  (mesmas sementes = mesma inicializacao e ordem de lotes) e p do teste de Welch.

Aceita varios results.json (ex.: uma execucao com 3 otimizadores e outra so
com o controle de passo equivalente) e junta pelo rotulo, avisando se o
protocolo (dataset, modelo, epocas, lote) for diferente entre eles.

Uso:
    python scripts/compare_stats.py results/<nome>/<data-hora>/results.json --ref sgd_momentum
    python scripts/compare_stats.py A.json B.json --ref pid_repo --latex docs/tabela_fig4.tex
"""

import argparse
import json
import sys

from optctrl.stats import mean, paired, sd, welch

METRICS = {
    'final_acc': ('acurácia final (%)', lambda h: h['eval_acc'][-1] if h['eval_acc'] else None,
                  100.0),
    'best_acc': ('melhor acurácia (%)', lambda h: max(h['eval_acc']) if h['eval_acc'] else None,
                 100.0),
    'epochs_to_target': ('épocas até o alvo', lambda h: h.get('epochs_to_target'), 1.0),
}
PROTOCOL_KEYS = ('dataset', 'model', 'train', 'eval_on')


def load(paths):
    runs, protocol, sources = {}, None, {}
    for path in paths:
        with open(path, encoding='utf-8') as f:
            res = json.load(f)
        proto = {k: res['config'].get(k, res.get(k)) for k in PROTOCOL_KEYS}
        if protocol is None:
            protocol = proto
        elif proto != protocol:
            print(f"AVISO: protocolo de {path} difere do primeiro arquivo:\n"
                  f"  {proto}\n  {protocol}", file=sys.stderr)
        for label, hists in res['per_seed'].items():
            if label in runs:
                print(f"AVISO: rotulo '{label}' repetido em {path}; mantendo o primeiro",
                      file=sys.stderr)
                continue
            runs[label] = {h['seed']: h for h in hists if not h.get('diverged')}
            sources[label] = path
    return runs, protocol


def compare(runs, ref, metric):
    name, getter, scale = METRICS[metric]
    ref_vals = {s: getter(h) for s, h in runs[ref].items()}
    ref_vals = {s: v for s, v in ref_vals.items() if v is not None}
    rows = []
    for label, by_seed in runs.items():
        vals = {s: getter(h) for s, h in by_seed.items()}
        vals = {s: v for s, v in vals.items() if v is not None}
        row = {'label': label, 'n': len(vals), 'n_total': len(by_seed),
               'mean': mean(list(vals.values())) * scale if vals else None,
               'sd': sd(list(vals.values())) * scale if len(vals) > 1 else None}
        if label != ref and len(vals) > 1 and len(ref_vals) > 1:
            w = welch(list(vals.values()), list(ref_vals.values()))
            row.update(diff=w['diff'] * scale, ci=(w['ci'][0] * scale, w['ci'][1] * scale),
                       p_welch=w['p'])
            common = sorted(set(vals) & set(ref_vals))
            if len(common) > 1:
                pr = paired([vals[s] for s in common], [ref_vals[s] for s in common])
                row.update(p_paired=pr['p'], n_paired=len(common),
                           ci_paired=(pr['ci'][0] * scale, pr['ci'][1] * scale),
                           diff_paired=pr['diff'] * scale)
        rows.append(row)
    return name, rows


def fmt_p(p):
    if p is None:
        return '—'
    return '< 0,001' if p < 1e-3 else f'{p:.3f}'.replace('.', ',')


def fmt_num(v, nd):
    return '—' if v is None else f'{v:.{nd}f}'.replace('.', ',')


def to_markdown(name, rows, ref, nd):
    out = [f'**{name}** (referência: `{ref}`)\n',
           '| otimizador | n | média ± dp | Δ vs ref. (pareado) [IC 95 %] | p pareado | p Welch |',
           '|---|---|---|---|---|---|']
    for r in rows:
        if r['label'] == ref:
            delta = '(referência)'
        elif 'diff_paired' in r:
            delta = (f"{fmt_num(r['diff_paired'], nd)} [{fmt_num(r['ci_paired'][0], nd)}; "
                     f"{fmt_num(r['ci_paired'][1], nd)}]")
        elif 'diff' in r:
            lo, hi = fmt_num(r['ci'][0], nd), fmt_num(r['ci'][1], nd)
            delta = f"{fmt_num(r['diff'], nd)} [{lo}; {hi}]"
        else:
            delta = '—'
        n = f"{r['n']}" + (f"/{r['n_total']}" if r['n'] != r['n_total'] else '')
        out.append(f"| {r['label']} | {n} | {fmt_num(r['mean'], nd)} ± {fmt_num(r['sd'], nd)} "
                   f"| {delta} | {fmt_p(r.get('p_paired'))} | {fmt_p(r.get('p_welch'))} |")
    return '\n'.join(out) + '\n'


def to_latex(blocks, ref):
    def esc(s):
        return s.replace('_', r'\_').replace('%', r'\%')

    def num(v, nd):                    # virgula decimal sem espaco extra no modo matematico
        return '---' if v is None else '$' + f'{v:.{nd}f}'.replace('.', '{,}') + '$'

    def pv(p):
        if p is None:
            return '---'
        return '$<0{,}001$' if p < 1e-3 else num(p, 3)

    lines = [r'\begin{tabular}{lrcccc}', r'\toprule',
             r'Otimizador & $n$ & Média $\pm$ dp & $\Delta$ vs.\ ref.\ [IC 95\%] '
             r'& $p$ pareado & $p$ Welch \\', r'\midrule']
    for name, rows, nd in blocks:
        lines.append(rf'\multicolumn{{6}}{{l}}{{\textit{{{esc(name)}}}}} \\')
        for r in rows:
            if r['label'] == ref:
                delta = '(referência)'
            elif 'diff_paired' in r:
                delta = (f"{num(r['diff_paired'], nd)} "
                         f"[{num(r['ci_paired'][0], nd)}; {num(r['ci_paired'][1], nd)}]")
            else:
                delta = '---'
            mean_sd = (num(r['mean'], nd)[:-1] + r' \pm ' + num(r['sd'], nd)[1:]
                       if r['sd'] is not None else num(r['mean'], nd))
            lines.append(f"{esc(r['label'])} & {r['n']} & {mean_sd} & {delta} & "
                         f"{pv(r.get('p_paired'))} & {pv(r.get('p_welch'))} \\\\")
        lines.append(r'\midrule')
    lines[-1] = r'\bottomrule'
    lines.append(r'\end{tabular}')
    return '\n'.join(lines) + '\n'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('results', nargs='+')
    p.add_argument('--ref', required=True, help='rotulo do otimizador de referencia')
    p.add_argument('--metrics', nargs='+', default=['final_acc', 'epochs_to_target'],
                   choices=sorted(METRICS))
    p.add_argument('--latex', default=None, help='salva a tabela em LaTeX (booktabs)')
    p.add_argument('--markdown', default=None, help='salva a tabela em Markdown')
    args = p.parse_args(argv)

    runs, protocol = load(args.results)
    if args.ref not in runs:
        sys.exit(f"Referencia '{args.ref}' nao encontrada. Rotulos: {sorted(runs)}")

    md_parts, blocks = [], []
    print(f"Protocolo: {protocol}\n")
    for metric in args.metrics:
        nd = 2 if METRICS[metric][2] == 100.0 else 1
        name, rows = compare(runs, args.ref, metric)
        md = to_markdown(name, rows, args.ref, nd)
        print(md)
        md_parts.append(md)
        blocks.append((name, rows, nd))
    print('Notas: dp amostral (n-1). "Pareado" usa só as sementes presentes nos dois '
          'otimizadores; execuções que divergiram ficam de fora (n/total).')

    if args.markdown:
        with open(args.markdown, 'w', encoding='utf-8') as f:
            f.write('\n'.join(md_parts))
        print(f'Markdown salvo em {args.markdown}')
    if args.latex:
        with open(args.latex, 'w', encoding='utf-8') as f:
            f.write(to_latex(blocks, args.ref))
        print(f'LaTeX salvo em {args.latex} (requer \\usepackage{{booktabs}})')
    return blocks


if __name__ == '__main__':
    main()
