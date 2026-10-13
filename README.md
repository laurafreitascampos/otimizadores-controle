# otimizadores-controle

Leis de aprendizado de redes neurais (SGD, SGD-Momentum, Adam e variações PID)
analisadas como **sistemas de controle em tempo discreto**: replicação de
artigos, comparação justa entre otimizadores e análise de estabilidade.

Projeto Final de Curso (PFC), Engenharia Elétrica, CEFET-MG.
Autora: **Laura Inês Freitas Campos**. Orientação: Prof. Dr. Everthon de Souza Oliveira.

Este repositório evolui o trabalho do Laboratório de IA
([analise-de-otimizadores-ML](https://github.com/laurafreitascampos/analise-de-otimizadores-ML)).
O que veio de lá está listado em [`docs/migracao.md`](docs/migracao.md).

## Organização

Cada artigo ou estudo é **um experimento** em `experiments/`; o código
reaproveitável fica no pacote `optctrl`.

```
├── src/optctrl/
│   ├── optimizers/   SGD_PID (forma separada), SGD_PID_Paper (Eq. 10), Adam_PID + registro
│   ├── data/         MNIST, Fashion-MNIST, EMNIST, sintético (na memória da GPU) + divisão treino/val/teste
│   ├── models/       MLP + registro
│   ├── training/     loop de treino, sementes, agregação entre sementes
│   ├── metrics/      épocas até o alvo, overshoot, tempo de acomodação, energia (CodeCarbon)
│   └── utils/        rastreabilidade (commit do git, ambiente) e E/S
├── scripts/          train.py (roda um config YAML) e plot_curves.py
├── experiments/
│   ├── an2018_pid/        replicação de An et al. (CVPR 2018)
│   └── dai2023_adampid/   Adam-PID (Dai et al., 2023), vindo do Lab de IA
├── reference/numpy/  implementação pedagógica em NumPy puro (Lab de IA)
├── tests/            equivalências exatas entre os otimizadores (P = SGD, PI = momentum, ...)
├── docs/             decisões e migração
└── results/          saídas dos experimentos (fora do git)
```

Para acrescentar algo novo:
- **otimizador** → classe em `src/optctrl/optimizers/` + uma linha no `REGISTRY`;
- **modelo ou dataset** → idem em `models/` ou `data/`;
- **experimento** → pasta em `experiments/` com `README.md` e `configs/*.yaml`.

## Instalação

Requer Python 3.10+ e, para os experimentos acelerados, uma GPU NVIDIA
(desenvolvido em uma GTX 1650). Tudo roda em CPU, mais devagar.

**Windows (PowerShell):**

```powershell
git clone https://github.com/laurafreitascampos/otimizadores-controle.git
cd otimizadores-controle
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# 1) PyTorch com CUDA: copie o comando do seletor em https://pytorch.org/get-started/locally/
#    (o pip padrão no Windows instala só a versão de CPU)
# 2) o restante:
pip install -e ".[dev,energy]"
pytest
```

Se o PowerShell bloquear o `Activate.ps1`, rode uma vez:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

**Linux:**

```bash
python -m venv .venv && source .venv/bin/activate   # fish: source .venv/bin/activate.fish
pip install -e ".[dev,energy]"
pytest
```

## Uso

```bash
python scripts/train.py experiments/an2018_pid/configs/fig4_mnist_repo.yaml
python scripts/plot_curves.py results/an2018_fig4_mnist_repo/<timestamp>/results.json
```

Cada execução grava `results/<nome>/<timestamp>/results.json` com o config
usado, o commit do git (e se havia alterações não commitadas), o ambiente,
as curvas por semente e a média ± desvio entre sementes.

## Protocolo de comparação justa

Herdado do Lab de IA:
- sintonização **só na validação**; o teste fica intocado até a avaliação final;
- todos os otimizadores partem da **mesma inicialização** e veem os
  **mini-lotes na mesma ordem** para uma mesma semente;
- várias sementes, com média ± desvio;
- execuções que divergem são registradas, não descartadas em silêncio.

Exceção: replicações de artigos que reportam direto no teste usam
`eval_on: test` e `val_size: 0`, documentado no config.

## Referências principais

1. W. An *et al.* "A PID Controller Approach for Stochastic Optimization of Deep Networks." *CVPR*, 2018.
2. M. Dai *et al.* "PID controller-based adaptive gradient optimizer for deep neural networks." *IET Control Theory & Applications*, 2023.
3. A. C. Wilson *et al.* "The Marginal Value of Adaptive Gradient Methods in Machine Learning." *NeurIPS*, 2017.
