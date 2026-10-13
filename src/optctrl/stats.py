"""Testes t de Welch e pareado, sem depender do scipy.

Qual usar:
  - PAREADO: quando os dois otimizadores rodaram com as MESMAS sementes neste
    repositorio. Mesma semente = mesma inicializacao e mesma ordem de
    mini-lotes, entao a diferenca por semente isola o efeito do otimizador e
    cancela a variacao "de sorte" da inicializacao. E o teste certo para o
    nosso protocolo, e o mais sensivel.
  - WELCH: quando as amostras sao independentes (sementes diferentes, ou
    comparacao com numeros de outro trabalho). Nao assume variancias iguais.

A distribuicao t e calculada pela funcao beta incompleta regularizada
(fracao continuada de Lentz), precisa ate ~1e-10.
"""

import math


# ----------------------------------------------------------------------------
# Distribuicao t de Student
# ----------------------------------------------------------------------------
def _betacf(a, b, x, max_iter=300, eps=3e-16):
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betainc_reg(a, b, x):
    """Funcao beta incompleta regularizada I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_pvalue_two_sided(t, df):
    """P(|T| >= |t|) para T ~ t de Student com df graus de liberdade."""
    if not math.isfinite(t):
        return 0.0
    return betainc_reg(df / 2.0, 0.5, df / (df + t * t))


def t_quantile(p, df):
    """Quantil da t (ex.: p = 0.975 para IC de 95 %), por bissecao."""
    target = 2.0 * (1.0 - p)                       # p-valor bilateral correspondente
    lo, hi = 0.0, 1e3
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_pvalue_two_sided(mid, df) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ----------------------------------------------------------------------------
# Estatisticas descritivas e testes
# ----------------------------------------------------------------------------
def mean(x):
    return sum(x) / len(x)


def sd(x):
    """Desvio padrao AMOSTRAL (n - 1). O resumo do train.py usa n; aqui e n - 1."""
    if len(x) < 2:
        return float('nan')
    mu = mean(x)
    return math.sqrt(sum((v - mu) ** 2 for v in x) / (len(x) - 1))


def welch(x, y, conf=0.95):
    """Compara media(x) - media(y) sem assumir variancias iguais."""
    nx, ny = len(x), len(y)
    vx, vy = sd(x) ** 2 / nx, sd(y) ** 2 / ny
    diff = mean(x) - mean(y)
    se = math.sqrt(vx + vy)
    if se == 0:
        return {'diff': diff, 't': math.copysign(math.inf, diff) if diff else 0.0,
                'df': nx + ny - 2, 'p': 0.0 if diff else 1.0, 'ci': (diff, diff)}
    df = (vx + vy) ** 2 / (vx ** 2 / (nx - 1) + vy ** 2 / (ny - 1))
    t = diff / se
    q = t_quantile(0.5 + conf / 2, df)
    return {'diff': diff, 't': t, 'df': df, 'p': t_pvalue_two_sided(t, df),
            'ci': (diff - q * se, diff + q * se)}


def paired(x, y, conf=0.95):
    """Teste t pareado: x[i] e y[i] vem da MESMA semente."""
    if len(x) != len(y):
        raise ValueError('teste pareado exige o mesmo numero de observacoes')
    d = [a - b for a, b in zip(x, y)]
    n = len(d)
    diff, s = mean(d), sd(d)
    if s == 0 or n < 2:
        return {'diff': diff, 't': math.inf if diff else 0.0, 'df': n - 1,
                'p': 0.0 if diff else 1.0, 'ci': (diff, diff)}
    se = s / math.sqrt(n)
    t = diff / se
    q = t_quantile(0.5 + conf / 2, n - 1)
    return {'diff': diff, 't': t, 'df': n - 1, 'p': t_pvalue_two_sided(t, n - 1),
            'ci': (diff - q * se, diff + q * se)}
