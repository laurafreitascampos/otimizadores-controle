# Migração do repositório do Lab de IA

Origem: [analise-de-otimizadores-ML](https://github.com/laurafreitascampos/analise-de-otimizadores-ML),
commit `a69717e` (README de reprodução e ajustes da Fase 2). O repositório
antigo fica congelado como registro do trabalho entregue (sugestão: tag `v1.0-lab-ia`).

| Arquivo antigo | Destino | Situação |
|---|---|---|
| `pid_optimizers.py` | `src/optctrl/optimizers/pid.py` | ✅ migrado. `SGD_PID_Classic` virou `SGD_PID_Paper` (alias mantido). Docstrings corrigidas: a forma separada é a do código tensorboy; a "clássica" é a Eq. (10) do artigo |
| `compare_mnist_mlp.py` | `data/`, `models/`, `training/`, `scripts/train.py`, `scripts/plot_curves.py` | ✅ generalizado: registros de dataset/modelo/otimizador, configs YAML |
| `tune_mnist.py`: divisão treino/val/teste | `optctrl.data.load_splits` | ✅ |
| `tune_mnist.py`: `train_eval` | `optctrl.training.train_eval` | ✅ |
| `tune_mnist.py`: energia/CO₂ | `optctrl.metrics.energy` | ✅ (`--energy` no `train.py`) |
| `tune_mnist.py`: fases grid → confirm → dterm → final | `scripts/tune.py` | ⬜ próximo passo: versão genérica dirigida por config |
| `core.py`, `conv.py` | `reference/numpy/` | ✅ copiados sem alteração |
| `make_figs_mlp.py`, `make_figs.py`, `make_efficiency.py` | `scripts/` | ⬜ portar sob demanda, lendo o novo formato de `results.json` |
| `run_experiment.py`, `tune_pid.py`, `confirm_seeds.py` (CIFAR, NumPy) | — | não migrados; o CIFAR será refeito em PyTorch com DataLoader |
| `experiments/tune_*/*.json`, `results.json`, `imagens/` | — | ficam no repositório antigo (resultados do trabalho entregue) |
| `baixar_mnist.py` | — | desnecessário: o torchvision baixa sozinho |

## Diferenças de comportamento em relação ao código antigo

- Nenhuma nas equações dos otimizadores (verificado pelos testes de equivalência).
- O `train.py` grava o commit do git e o ambiente em cada `results.json`.
- Curvas por época agora usam `eval_*` (validação ou teste, conforme `eval_on`)
  no lugar de `test_*`.
