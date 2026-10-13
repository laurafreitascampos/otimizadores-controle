# Implementação de referência em NumPy

`core.py` (MLP + SGD, Adam, SGD-PID e Adam-PID) e `conv.py` (CNN com im2col)
vieram sem alteração do repositório do Lab de IA. Cada passo de atualização é
escrito literalmente como a equação do artigo, sem autograd.

Servem como referência pedagógica e como ponto de partida para a análise em
tempo discreto do PFC (o passo do otimizador como sistema dinâmico linear em
uma quadrática). Os experimentos usam a versão em PyTorch, em `src/optctrl/`.

São módulos (sem ponto de entrada próprio): os scripts que os usavam
(`tune_pid.py`, `confirm_seeds.py`, `run_experiment.py`) continuam no
repositório antigo. O `conv.py` faz `import core`, então os dois precisam
estar na mesma pasta.
