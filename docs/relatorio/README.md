# Relatório de andamento (LaTeX)

`relatorio.tex` puxa as figuras de `figuras/` e as tabelas de `tabelas/`.
Nada é digitado a partir de gráficos: tudo é gerado pelos scripts a partir
dos arquivos de resultado.

## Atualizar figuras e tabelas

Na raiz do repositório:

```powershell
# figuras analíticas (sempre)
python scripts/report_assets.py

# com os resultados experimentais (substitua <data-hora> pelas pastas reais)
python scripts/report_assets.py --fig4 results/an2018_fig4_mnist_repo/<data-hora>/results.json results/an2018_h2_passo_equivalente/<data-hora>/results.json --sweep-lr results/mapa_mnist_momentum_lr/<data-hora>/sweep.json --sweep-kd results/mapa_mnist_pid_kd/<data-hora>/sweep.json
```

O que ainda não foi gerado aparece no PDF como uma caixa cinza **Pendente**,
dizendo o que falta rodar. Quando `tabelas/fig4.tex` existir, ela substitui
automaticamente a tabela transcrita à mão da Seção 5.

## Compilar

- **Overleaf:** envie `relatorio.tex` e as pastas `figuras/` e `tabelas/`
  (só os `.pdf` de figuras são usados; os `.png` servem para conferir).
- **Local:** `pdflatex relatorio.tex` duas vezes (requer `babel` com português).
