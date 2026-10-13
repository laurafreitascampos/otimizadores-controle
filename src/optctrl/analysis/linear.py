"""Modelos lineares EXATOS das leis de aprendizado numa funcao quadratica.

Numa direcao da funcao de perda com curvatura a (f = a/2 * theta^2), o
gradiente e g = a*theta. Substituindo isso na lei de aprendizado, todo o
treinamento vira um sistema linear invariante no tempo, em tempo discreto:

    x_{t+1} = A x_t

em que x_t e o vetor de estado (o parametro e as memorias do otimizador) e
A e a "matriz de malha fechada" (planta quadratica + controlador = lei de
aprendizado). Os autovalores de A sao os POLOS do sistema e:

    raio espectral rho(A) = max |polo|
      rho < 1 -> converge (quanto menor, mais rapido)
      rho > 1 -> diverge
      polos complexos -> oscila ao convergir (sobressinal)

Para uma quadratica em varias dimensoes com Hessiana diagonalizavel, cada
autovalor da Hessiana e uma direcao independente com sua propria matriz A:
o sistema converge se TODAS as direcoes convergirem.

Decaimento de peso L2 (weight_decay = wd) soma wd a curvatura: g = (a+wd)*theta.

As matrizes aqui sao exatas (nao aproximacoes) para quadraticas; isso e
verificado em tests/test_linear_model.py comparando A^t x_0 com a trajetoria
produzida pelas classes de otimizador de verdade.
"""

import inspect

import numpy as np


# ----------------------------------------------------------------------------
# Matrizes de malha fechada
# ----------------------------------------------------------------------------
def gd_matrix(h):
    """SGD sem momento (controlador P). Estado x = [theta].

        theta_{t+1} = theta_t - lr*a*theta_t = (1 - h) theta_t,   h = lr*a
    """
    return np.array([[1.0 - h]])


def momentum_matrix(h, m):
    """SGD-Momentum (controlador PI). Estado x = [theta_t, V_t].

        V_{t+1}     = m V_t - h theta_t                (Eq. 3 de An et al.)
        theta_{t+1} = theta_t + V_{t+1}

    Polinomio caracteristico: z^2 - (1 + m - h) z + m = 0.
    Igual ao torch.optim.SGD(momentum=m), com V = -lr*buf.
    """
    return np.array([[1.0 - h, m],
                     [-h, m]])


def pid_separated_matrix(h, m, Kp, Ki, Kd):
    """SGD_PID na forma SEPARADA (codigo publico de An et al.).
    Estado x_t = [theta_t, theta_{t-1}, lr*I_{t-1}, lr*D_{t-1}].

        I_t   = m I_{t-1} + g_t
        D_t   = m D_{t-1} + (1-m)(g_t - g_{t-1})
        theta_{t+1} = theta_t - lr (Kp g_t + Ki I_t + Kd D_t)

    Multiplicar I e D por lr faz a matriz depender so de h = lr*a.
    """
    I_row = np.array([h, 0.0, m, 0.0])
    D_row = np.array([(1 - m) * h, -(1 - m) * h, 0.0, m])
    theta_row = np.array([1.0 - Kp * h, 0.0, 0.0, 0.0]) - Ki * I_row - Kd * D_row
    return np.vstack([theta_row, [1.0, 0.0, 0.0, 0.0], I_row, D_row])


def pid_paper_matrix(r, a, alpha, Kd):
    """SGD_PID_Paper, Eq. (10) de An et al. (2018).
    Estado x_t = [theta_t, theta_{t-1}, V_t, D_t].

        V_{t+1} = alpha V_t - r g_t
        D_{t+1} = alpha D_t + (1-alpha)(g_t - g_{t-1})
        theta_{t+1} = theta_t + V_{t+1} + Kd D_{t+1}

    Aqui o termo D NAO e multiplicado por r, entao a matriz depende de r*a
    e de Kd*a separadamente (por isso recebe r e a, e nao so h).
    """
    V_row = np.array([-r * a, 0.0, alpha, 0.0])
    D_row = np.array([(1 - alpha) * a, -(1 - alpha) * a, 0.0, alpha])
    theta_row = np.array([1.0, 0.0, 0.0, 0.0]) + V_row + Kd * D_row
    return np.vstack([theta_row, [1.0, 0.0, 0.0, 0.0], V_row, D_row])


# ----------------------------------------------------------------------------
# Ferramentas
# ----------------------------------------------------------------------------
def poles(A):
    """Autovalores (polos) da matriz de malha fechada."""
    return np.linalg.eigvals(A)


def spectral_radius(A):
    """rho(A) = max |polo|."""
    return float(np.max(np.abs(np.linalg.eigvals(A))))


def simulate(A, x0, steps):
    """Trajetoria x_0, x_1, ..., x_steps (cada linha um instante)."""
    xs = [np.asarray(x0, dtype=float)]
    for _ in range(steps):
        xs.append(A @ xs[-1])
    return np.array(xs)


def largest_stable(rho_of, lo=1e-6, hi=10.0, n=600, rtol=1e-7):
    """Maior x em [lo, hi] com rho_of(x) < 1: varredura logaritmica para achar a
    ultima faixa estavel e bisseccao para refinar a fronteira (precisao rtol).
    Retorna None se nenhum ponto for estavel."""
    xs = np.logspace(np.log10(lo), np.log10(hi), n)
    stable = np.array([rho_of(x) < 1 for x in xs])
    if not stable.any():
        return None
    i = np.nonzero(stable)[0][-1]
    if i == n - 1:
        return float(xs[-1])
    a, b = xs[i], xs[i + 1]
    while (b - a) > rtol * a:
        mid = 0.5 * (a + b)
        a, b = (mid, b) if rho_of(mid) < 1 else (a, mid)
    return float(a)


def initial_state(theta0, size):
    """Estado inicial com memorias zeradas e g_{-1} = 0 (theta_{-1} = 0),
    exatamente como os otimizadores deste repositorio sao inicializados."""
    x0 = np.zeros(size)
    x0[0] = theta0
    return x0


# Fronteiras analiticas do SGD-Momentum (tempo discreto)
def momentum_boundaries(m):
    """Para 0 <= m < 1, os regimes do SGD-Momentum em funcao de h = lr*a:

        0 < h < (1 - sqrt m)^2           polos reais positivos: monotono (superamortecido)
        (1-sqrt m)^2 < h < (1+sqrt m)^2  polos complexos: oscila (subamortecido)
        (1+sqrt m)^2 < h < 2(1+m)        polo real negativo: alterna sinal (zigue-zague)
        h > 2(1+m)                       instavel
    """
    s = np.sqrt(m)
    return {'complexo_inicio': (1 - s) ** 2, 'complexo_fim': (1 + s) ** 2,
            'instavel': 2 * (1 + m)}


# ----------------------------------------------------------------------------
# Ponte com o REGISTRY de otimizadores
# ----------------------------------------------------------------------------
def _with_defaults(opt_name, hparams):
    from ..optimizers import REGISTRY
    sig = inspect.signature(REGISTRY[opt_name].__init__)
    full = {k: v.default for k, v in sig.parameters.items()
            if v.default is not inspect.Parameter.empty}
    full.update(hparams)
    return full


def closed_loop_matrix(opt_name, hparams, curvature):
    """Matriz A para um otimizador do REGISTRY numa direcao de curvatura dada.

    Retorna None para leis nao lineares (Adam, Adam-PID): a normalizacao
    por sqrt(v) faz a dinamica depender da amplitude, e a analise linear
    direta nao se aplica.
    """
    name = opt_name.lower()
    if name not in ('sgd', 'sgd_pid', 'sgd_pid_paper'):
        return None
    p = _with_defaults(name, hparams)
    a = curvature + p.get('weight_decay', 0.0)

    if name == 'sgd':
        if p.get('nesterov') or p.get('dampening', 0) != 0:
            return None
        h = p['lr'] * a
        m = p.get('momentum', 0.0)
        return gd_matrix(h) if m == 0 else momentum_matrix(h, m)
    if name == 'sgd_pid':
        return pid_separated_matrix(p['lr'] * a, p['momentum'], p['Kp'], p['Ki'], p['Kd'])
    return pid_paper_matrix(p['lr'], a, p['alpha'], p['Kd'])


def predicted_rho(opt_name, hparams, curvature):
    A = closed_loop_matrix(opt_name, hparams, curvature)
    return None if A is None else spectral_radius(A)
