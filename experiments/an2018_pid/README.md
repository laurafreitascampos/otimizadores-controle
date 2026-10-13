# Replicação: An et al. (2018), *A PID Controller Approach for Stochastic Optimization of Deep Networks* (CVPR)

Artigo de partida do PFC. O objetivo não é só reproduzir os números, mas
registrar **o que o artigo diz, o que o código público faz e o que observamos**.

- Artigo: CVPR 2018, pp. 8522–8531
- Código dos autores: [tensorboy/PIDOptimizer](https://github.com/tensorboy/PIDOptimizer)
  (só MNIST; sem licença declarada, por isso aqui tudo é reimplementado e citado, não copiado)
- Logs dos autores no repositório deles (`pid.txt`, `momentum.txt`, 20 épocas, batch 100):
  acurácia de teste final de **98,26 %** (PID) e **97,99 %** (SGD-Momentum)

## Escopo e status

| Item do artigo | Script / config | Hardware | Status |
|---|---|---|---|
| Fig. 2: overshoot no toy de De Jong | `toy_dejong.py` | CPU, segundos | ⬜ rodar e interpretar |
| Fig. 4: MLP no MNIST, 10 execuções | `configs/fig4_mnist_paper.yaml`, `configs/fig4_mnist_repo.yaml` | GTX 1650, minutos | ⬜ |
| Tab. 1: ResNet-110 / DenseNet-100-12 no CIFAR | (a fazer: caminho com DataLoader) | GTX 1650, horas por execução | ⬜ |
| Tab. 1: modelos grandes, Fig. 5, Fig. 6 | — | fora do alcance local | ⬜ avaliar Kaggle/Colab ou réplica do fenômeno com ResNet-20/32/56 |

## Como rodar

```bash
# Figura 2
python experiments/an2018_pid/toy_dejong.py

# Figura 4 (teste rápido primeiro, depois completo)
python scripts/train.py experiments/an2018_pid/configs/fig4_mnist_repo.yaml --seeds 0 --epochs 2
python scripts/train.py experiments/an2018_pid/configs/fig4_mnist_repo.yaml
python scripts/plot_curves.py results/an2018_fig4_mnist_repo/<timestamp>/results.json
```

## Registro de discrepâncias (artigo × código × observação)

| # | Onde | Artigo diz | Código público faz | Observado aqui |
|---|---|---|---|---|
| D1 | Forma do update | Eq. (10): θ ← θ + V + Kd·D, com lr dentro de V | θ ← θ − lr·(g + I·I_buf + D·D_buf) | As duas formas são equivalentes a menos do **sinal** de Kd: `SGD_PID(Kd=k)` ≡ `SGD_PID_Paper(Kd=−lr·k)` (teste `test_separated_and_paper_forms_differ_by_sign_of_D`) |
| D2 | Valor de Kd | Eq. (14): com r = 0,01, Kd ≈ 1854,6 | D = 100 multiplicado por lr → Kd efetivo = 1 | ⬜ `pid_paper_eq14` em `fig4_mnist_paper.yaml`. Ver também a issue #11 do repositório original |
| D3 | Sinal do termo D | +Kd·D (Eqs. 9 e 10) | −lr·Kd·D | ⬜ qual linha do `toy_dejong.py` reproduz a Fig. 2? |
| D4 | Batch do MNIST | 128 | 100 | dois configs, um para cada |
| D5 | Weight decay | não mencionado | 1e-4 | usado nos dois configs |
| D6 | Eq. (8) (overshoot) | (θmax − θ*)/θ* | — | indefinida para θ* = 0; usamos a normalização pela distância inicial (`optctrl.metrics.overshoot`) |
| D7 | Material suplementar | citado para as derivações das Eqs. (7) e (12) | — | indisponível (issue #6 do repositório original) |
| D8 | Inicialização do D_buf | — | referência ao tensor do gradiente, alterada in-place nos 1–2 primeiros passos | aqui g₋₁ = 0; diferença só no transiente |

## Hipóteses a testar

- **H1 (sinal):** o termo derivativo só reduz o overshoot com o sinal do código, não com o do texto.
- **H2 (passo efetivo):** com Ki = 3 e m = 0,9, o PID do código dá passos de ordem
  lr·(1 + 3/(1−0,9)) = 31·lr, contra 10·lr do SGD-Momentum de mesmo lr. Parte da
  "aceleração" pode ser só um passo maior. Teste: comparar com um SGD-Momentum de lr maior.
- **H3 (papel do D):** `pid_repo` × `pid_repo_kd0` no `fig4_mnist_repo.yaml` isola a contribuição
  do termo D. No Lab de IA, o termo D se mostrou praticamente inerte com mini-lotes.
- **H4 (métrica de épocas):** "épocas até o melhor erro" (Tab. 1) depende do agendamento do lr;
  reportar também tempo de relógio e memória (o PID guarda 3 buffers por parâmetro).
