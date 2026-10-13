# Estabilidade das leis de aprendizado em tempo discreto

Núcleo do PFC: **caracterizar a dinâmica das leis de aprendizado sob a
perspectiva de controle e mapear onde o treinamento converge, oscila ou
diverge** (objetivos b e c da proposta). Toda a análise é feita em tempo
discreto, sem aproximação contínua.

## 1. A ideia, em uma página

**A planta.** Perto de um mínimo, a perda de uma rede se parece com uma tigela
quadrática. Em cada direção principal da tigela, f(θ) ≈ ½·a·θ², em que a é a
**curvatura** daquela direção (um autovalor da Hessiana). O gradiente é
g = a·θ. É o "erro" que realimenta o controlador.

**O controlador.** A lei de aprendizado usa o gradiente para decidir o próximo
passo. Substituindo g = a·θ na lei, o treinamento inteiro vira um sistema
linear em tempo discreto:

    x_{t+1} = A · x_t

O vetor de estado x guarda o parâmetro θ e as **memórias** do otimizador
(velocidade V, termo integral I, termo derivativo D...). A é a matriz de malha
fechada: planta e controlador juntos.

**Polos e raio espectral.** Os autovalores de A são os polos do sistema:

| Polos | Comportamento do treino |
|---|---|
| todos com módulo < 1 | converge; quanto menor o maior módulo, mais rápido |
| algum com módulo > 1 | diverge |
| complexos | converge **oscilando**: passa do mínimo (sobressinal) |
| real negativo | converge em **zigue-zague**: troca de lado a cada passo |

O **raio espectral ρ** (o maior módulo entre os polos) resume tudo isso: é o
"termômetro" da convergência. A análise é **exata** para quadráticas, não uma
aproximação. Isso está verificado em `tests/test_linear_model.py`, que
compara Aᵗ·x₀ com a trajetória das classes de otimizador usadas nos
experimentos. A diferença é inferior a 10⁻⁹.

### As matrizes (em `src/optctrl/analysis/linear.py`)

Notação: h = lr·a (passo × curvatura).

- **SGD (controlador P):** θ_{t+1} = (1 − h)·θ_t. Converge se 0 < h < 2.
- **SGD-Momentum (controlador PI),** estado [θ, V]:

      A = [[1 − h, α],
           [  −h , α]]        polinômio característico: z² − (1 + α − h)·z + α = 0

  Fronteiras analíticas (para 0 ≤ α < 1):

  | Faixa de h | Polos | Regime |
  |---|---|---|
  | h < (1 − √α)² | reais positivos | monótono (superamortecido) |
  | (1 − √α)² < h < (1 + √α)² | complexos, \|z\| = √α | oscilatório (subamortecido) |
  | (1 + √α)² < h < 2(1 + α) | real negativo | zigue-zague |
  | h > 2(1 + α) | fora do círculo unitário | **instável** |

- **PID da Eq. (10) de An et al.,** estado [θ_t, θ_{t−1}, V, D] (4ª ordem).
- **PID na forma do código público,** estado [θ_t, θ_{t−1}, lr·I, lr·D] (4ª ordem).

## 2. Mapa analítico + validação no toy

```bash
python experiments/estabilidade_discreta/stability_map.py
```

Gera, em `results/estabilidade_discreta/<data-hora>/`:

| Figura | O que mostra |
|---|---|
| `regimes_momentum.png` | os quatro regimes do SGD-Momentum no plano (h, α), com as duas coordenadas do toy de De Jong marcadas |
| `mapa_pid_artigo.png` | (a) ρ do PID da Eq. (10) no plano (curvatura, Kd), com a fronteira ρ = 1 e os 14 casos do toy sobrepostos; (b) onde o termo D acelera (azul) ou atrasa (vermelho) a convergência em relação ao momentum |
| `mapa_pid_codigo.png` | a região estável do PID na forma do código (Kp = 1, Ki = 3, m = 0,9) no plano (h, Kd) |
| `validacao_toy.md` | tabela: ρ previsto × comportamento observado |

**Resultado (determinístico, não depende de semente):** a teoria linear acerta
convergir/divergir nos **14/14** casos do toy. Ela também explica o que a
inspeção visual não explica:

- **x₁ (a = 0,2)** está na região de polos reais. Por isso **não há
  sobressinal** em nenhum caso, e o Kd negativo, que acrescenta amortecimento,
  **atrasa** a convergência: ρ cresce de 0,974 para 0,980 e a acomodação passa
  de 161 para 195 passos. Amortecer um modo que já é superamortecido só o
  deixa mais lento.
- **x₂ (a = 4)** está na região de polos complexos (ρ = √0,9 = 0,949). Aí o
  Kd negativo amortece a oscilação (sobressinal de 0,44 para 0) e o Kd
  positivo, o sinal escrito no artigo, empurra os polos para fora do círculo
  unitário: ρ = 1,049 com Kd = +0,5 e 1,128 com Kd = +0,93. **Diverge.**
- Consequência para o projeto: **não existe um Kd bom para todas as
  direções.** O ganho que amortece as direções de curvatura alta atrasa as de
  curvatura baixa. A rede tem muitas direções com curvaturas muito
  diferentes, e isso é um indicativo de projeto para o PFC.

**Margem de estabilidade na forma do código** (maior h = lr·a estável):

| Kd | h máximo |
|---|---|
| 0 | 0,772 |
| +30 | 0,348 |
| +100 (valor dos autores) | 0,153 |
| +300 | 0,059 |
| −30 | 0,048 |
| −100 | 0,011 |
| *SGD-Momentum (α = 0,9), para comparação* | *3,8* |

O PID do código (Ki = 3) já nasce com **~5× menos margem** que o momentum, por
causa do ganho integral maior. O Kd = 100 corta essa margem mais 5×. Com o
sinal do artigo (Kd < 0), a margem cai ainda mais rápido.

## 3. Mapa empírico no MNIST

Uma rede real não é uma quadrática: tem ruído de mini-lote e curvatura que
muda durante o treino. A pergunta é: **a fronteira prevista aparece mesmo
assim?**

```bash
python scripts/sweep.py experiments/estabilidade_discreta/configs/mnist_momentum_lr.yaml
python scripts/sweep.py experiments/estabilidade_discreta/configs/mnist_pid_kd.yaml
python scripts/plot_sweep.py results/<nome>/<data-hora>/sweep.json
```

Cada varredura:
1. mede a curvatura máxima da perda na inicialização, **λmax** (maior
   autovalor da Hessiana, por iteração de potência com produtos
   Hessiana-vetor, sem montar a Hessiana);
2. calcula, para cada ponto da grade, o **ρ previsto** pela matriz acima com
   a = λmax;
3. treina e classifica cada execução: **convergiu**, **instável** (a perda de
   treino sobe mais de 10 % entre épocas), **não convergiu** (perda final ≥ ln C,
   ou seja, nada melhor que chutar) ou **divergiu** (NaN);
4. imprime em quantos pontos a previsão "ρ ≥ 1 → diverge" bate com o observado.

O mapeamento é feito na **validação**: o teste fica intocado.

| Varredura | Pontos | Custo (GTX 1650) |
|---|---|---|
| `mnist_momentum_lr.yaml`: lr × momento | 36 × 2 sementes × 5 épocas | ~8 min |
| `mnist_pid_kd.yaml`: Kd de −300 a 3000 | 11 × 3 sementes × 5 épocas | ~6 min |

Depois de rodar a varredura, gere de novo o mapa 3 com o λmax medido:
`stability_map.py --lambda-max <valor>`.

### Resultados

⬜ preencher depois de rodar: λmax, concordância teoria × prática, figuras.

**Cuidados na interpretação:**
- λmax é medido **na inicialização**. Durante o treino a curvatura muda, então
  a previsão é de primeira ordem.
- Com ruído de mini-lote, uma configuração "estável" no sentido linear ainda
  pode oscilar em torno do mínimo.
- Uma discordância entre teoria e prática também é resultado: ela delimita
  até onde a análise linear vale para redes reais.
