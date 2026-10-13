# Adam-PID: Dai et al. (2023), *PID controller-based adaptive gradient optimizer for deep neural networks* (IET CTA)

Trabalho do Laboratório de IA. O otimizador `Adam_PID` já está em
`src/optctrl/optimizers/pid.py`, com o teste de redução ao Adam
(`test_adam_pid_I_only_is_adam`).

Os resultados entregues (MLP/MNIST, MLP/Fashion-MNIST, CNN/CIFAR-10, com
energia e CO₂) estão no repositório
[analise-de-otimizadores-ML](https://github.com/laurafreitascampos/analise-de-otimizadores-ML).
Principais achados de lá: sem ganho estatisticamente significativo do Adam-PID
sobre o Adam sob comparação justa, e termo D praticamente inerte com mini-lotes.

## Próximos passos
- [ ] portar as fases de sintonização (`grid → confirm → dterm → final`) para um `scripts/tune.py` genérico
- [ ] configs YAML para MNIST e Fashion-MNIST reproduzindo a tabela final do Lab de IA
