"""
Otimizadores no estilo PID como subclasses de torch.optim.Optimizer.

Migrado de analise-de-otimizadores-ML/pid_optimizers.py (Lab. de IA, CEFET-MG).
Mudancas em relacao ao original:
  - atribuicoes corrigidas: a forma SEPARADA (SGD_PID) e a do codigo publico
    tensorboy/PIDOptimizer; a forma da Eq. (10) do artigo e SGD_PID_Paper
    (antes chamada SGD_PID_Classic, nome mantido como alias);
  - documentada a relacao exata entre as duas formas, incluindo o SINAL de Kd.

Convencao: o gradiente g_t faz o papel do erro e(t) do controlador.

Referencias:
    An et al., "A PID Controller Approach for Stochastic Optimization of Deep
        Networks", CVPR 2018.
    Dai et al., "PID controller-based adaptive gradient optimizer for deep
        neural networks", IET Control Theory & Applications, 2023.
"""

import torch
from torch.optim.optimizer import Optimizer


def _apply_weight_decay(g, p, wd):
    """Decaimento L2 acoplado (novo tensor; o .grad original fica intacto)."""
    return g.add(p, alpha=wd) if wd != 0.0 else g


class SGD_PID(Optimizer):
    r"""PID sobre SGD na forma SEPARADA (tres ganhos), como no codigo publico
    tensorboy/PIDOptimizer (pid.py):

        I_t = m*I_{t-1} + g_t
        D_t = m*D_{t-1} + (1-m)*(g_t - g_{t-1})
        theta_{t+1} = theta_t - lr * (Kp*g_t + Ki*I_t + Kd*D_t)

    Reproduz o mnist_pid.py dos autores com
        lr=0.01, momentum=0.9, weight_decay=1e-4, Kp=1, Ki=3, Kd=100.
    Unica diferenca: aqui g_{-1}=0 (o original tem um bug de referencia que
    altera o D_buf nos 1-2 primeiros passos). E um transiente, mas documente.

    Casos particulares (testados em tests/test_equivalence.py):
        Kp=1, Ki=0, Kd=0  -> SGD puro
        Kp=0, Ki=1, Kd=0  -> torch.optim.SGD(momentum=m)

    Cuidado com a taxa efetiva: com Ki>0 o buffer integral satura em ~1/(1-m),
    entao o passo efetivo e ~ lr*(Kp + Ki/(1-m)). Comparar com um SGD-Momentum
    de mesmo lr NAO e comparar passos de mesmo tamanho.
    """

    def __init__(self, params, lr=0.1, momentum=0.9, weight_decay=0.0,
                 Kp=1.0, Ki=1.0, Kd=0.3):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        Kp=Kp, Ki=Ki, Kd=Kd)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, m, wd = group['lr'], group['momentum'], group['weight_decay']
            Kp, Ki, Kd = group['Kp'], group['Ki'], group['Kd']

            for p in group['params']:
                if p.grad is None:
                    continue
                g = _apply_weight_decay(p.grad, p, wd)

                state = self.state[p]
                if len(state) == 0:
                    state['I_buf'] = torch.zeros_like(p)
                    state['D_buf'] = torch.zeros_like(p)
                    state['prev_grad'] = torch.zeros_like(p)
                I_buf, D_buf, prev_grad = state['I_buf'], state['D_buf'], state['prev_grad']

                I_buf.mul_(m).add_(g)                                  # I: historico
                D_buf.mul_(m).add_(g - prev_grad, alpha=(1.0 - m))     # D: variacao suavizada
                prev_grad.copy_(g)

                update = g.mul(Kp).add_(I_buf, alpha=Ki).add_(D_buf, alpha=Kd)
                p.add_(update, alpha=-lr)

        return loss


class SGD_PID_Paper(Optimizer):
    r"""PID sobre SGD exatamente como escrito na Eq. (10) de An et al. (2018):

        V_{t+1} = alpha*V_t - lr*g_t
        D_{t+1} = alpha*D_t + (1-alpha)*(g_t - g_{t-1})
        theta_{t+1} = theta_t + V_{t+1} + Kd*D_{t+1}

    Aqui lr fica DENTRO de V e o termo D NAO e multiplicado por lr.

    Relacao exata com a forma separada (teste em tests/test_equivalence.py):
        SGD_PID(Kp=0, Ki=1, Kd=k, momentum=m)
            == SGD_PID_Paper(alpha=m, Kd = -lr*k)

    Ou seja: o MESMO termo derivativo aparece com sinais opostos no texto do
    artigo (+Kd*D) e no codigo publico (-lr*Kd*D). Qual dos dois produz o
    amortecimento da Figura 2 e uma pergunta experimental
    (experiments/an2018_pid/toy_dejong.py).

    Com Kd=0 reduz-se a torch.optim.SGD(momentum=alpha).
    """

    def __init__(self, params, lr=0.1, alpha=0.9, Kd=0.5, weight_decay=0.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, alpha=alpha, Kd=Kd, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, alpha, Kd, wd = group['lr'], group['alpha'], group['Kd'], group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                g = _apply_weight_decay(p.grad, p, wd)

                state = self.state[p]
                if len(state) == 0:
                    state['V'] = torch.zeros_like(p)
                    state['D'] = torch.zeros_like(p)
                    state['prev_grad'] = torch.zeros_like(p)
                V, D, prev_grad = state['V'], state['D'], state['prev_grad']

                V.mul_(alpha).add_(g, alpha=-lr)                        # V = alpha*V - lr*g
                D.mul_(alpha).add_(g - prev_grad, alpha=(1.0 - alpha))
                prev_grad.copy_(g)
                p.add_(V).add_(D, alpha=Kd)                             # theta += V + Kd*D

        return loss


# Nome usado no repositorio do Lab de IA (compatibilidade com JSONs antigos).
SGD_PID_Classic = SGD_PID_Paper


class Adam_PID(Optimizer):
    r"""PID adaptativo sobre Adam, seguindo Dai et al. (IET CTA 2023, Eq. 7-8).

        m_t = b1*m + (1-b1)*g            v_t = b2*v + (1-b2)*g^2
        mhat = m_t/(1-b1^t)              vhat = v_t/(1-b2^t)
        denom = sqrt(vhat) + eps
        Dhat = lr/(1-gamma^t) * (g_t - g_{t-1})

        theta_{t+1} = theta_t - (lr/denom)*(Kp*g_t + Ki*mhat) - Kd*Dhat

    O termo D fica FORA da normalizacao adaptativa (fiel a Dai et al.).
    Reduz-se exatamente ao Adam com Kp=0, Ki=1, Kd=0.

    Passo 1: g_{-1}=0, entao D_1 = g_1 e o fator lr/(1-gamma) = 10*lr da um
    "chute" derivativo inicial. d_warmup=True zera D no passo 1.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, Kp=0.5, Ki=1.0, Kd=0.3, gamma=0.9,
                 d_warmup=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay,
                        Kp=Kp, Ki=Ki, Kd=Kd, gamma=gamma, d_warmup=d_warmup)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            b1, b2 = group['betas']
            eps, wd = group['eps'], group['weight_decay']
            Kp, Ki, Kd = group['Kp'], group['Ki'], group['Kd']
            gamma, d_warmup = group['gamma'], group['d_warmup']

            for p in group['params']:
                if p.grad is None:
                    continue
                g = _apply_weight_decay(p.grad, p, wd)

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)
                    state['prev_grad'] = torch.zeros_like(p)

                state['step'] += 1
                t = state['step']
                m, v, prev_grad = state['m'], state['v'], state['prev_grad']

                m.mul_(b1).add_(g, alpha=(1.0 - b1))
                v.mul_(b2).addcmul_(g, g, value=(1.0 - b2))
                mhat = m / (1.0 - b1 ** t)
                denom = (v / (1.0 - b2 ** t)).sqrt_().add_(eps)

                D = torch.zeros_like(g) if (t == 1 and d_warmup) else g - prev_grad
                Dhat_scale = lr / (1.0 - gamma ** t)
                prev_grad.copy_(g)

                pi = g.mul(Kp).add_(mhat, alpha=Ki)          # P + I com escala adaptativa
                p.addcdiv_(pi, denom, value=-lr)
                p.add_(D, alpha=-Kd * Dhat_scale)            # D separado

        return loss
