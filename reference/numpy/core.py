"""
core.py
Núcleo do experimento: um MLP (784-1000-10) treinado em MNIST com quatro
otimizadores escritos DIRETAMENTE a partir das equações dos artigos:

  SGD        -> controlador P            (An et al., 2018)
  Adam       -> controlador I adaptativo (Dai et al., 2023)
  SGD-PID    -> P + I + D                (An et al., 2018, Eq. 10)
  Adam-PID   -> Adam + P + D             (Dai et al., 2023, Eq. 7)

Implementacao em NumPy puro, de proposito, para que cada passo de atualizacao
seja literalmente a equacao do artigo (facil de explicar no texto).
"""
import time
import numpy as np


# ----------------------------------------------------------------------------
# Modelo: MLP de uma camada escondida (784 -> H -> 10), ReLU + softmax
# ----------------------------------------------------------------------------
class MLP:
    def __init__(self, n_in=784, n_hidden=1000, n_out=10, seed=0):
        rng = np.random.default_rng(seed)
        # Inicializacao He (adequada para ReLU)
        self.W1 = rng.standard_normal((n_in, n_hidden)).astype(np.float32) * np.sqrt(2.0 / n_in)
        self.b1 = np.zeros(n_hidden, dtype=np.float32)
        self.W2 = rng.standard_normal((n_hidden, n_out)).astype(np.float32) * np.sqrt(2.0 / n_hidden)
        self.b2 = np.zeros(n_out, dtype=np.float32)

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def forward(self, X):
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0.0, self.z1)          # ReLU
        self.z2 = self.a1 @ self.W2 + self.b2
        # softmax estavel
        z = self.z2 - self.z2.max(axis=1, keepdims=True)
        e = np.exp(z)
        self.probs = e / e.sum(axis=1, keepdims=True)
        return self.probs

    def backward(self, y_onehot):
        n = self.X.shape[0]
        dz2 = (self.probs - y_onehot) / n           # derivada da cross-entropy+softmax
        dW2 = self.a1.T @ dz2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (self.z1 > 0)                    # derivada da ReLU
        dW1 = self.X.T @ dz1
        db1 = dz1.sum(axis=0)
        return [dW1, db1, dW2, db2]


def onehot(y, k=10):
    o = np.zeros((y.shape[0], k), dtype=np.float32)
    o[np.arange(y.shape[0]), y] = 1.0
    return o


def cross_entropy(probs, y):
    return float(-np.log(probs[np.arange(y.shape[0]), y] + 1e-12).mean())


def accuracy(probs, y):
    return float((probs.argmax(axis=1) == y).mean())


# ----------------------------------------------------------------------------
# Otimizadores -- cada classe implementa uma equacao dos artigos
# ----------------------------------------------------------------------------
class SGD:
    """Controlador P:  theta <- theta - lr * g   (An et al., 2018, Sec. 3.2)"""
    def __init__(self, params, lr=0.1):
        self.lr = lr

    def step(self, params, grads):
        for p, g in zip(params, grads):
            p -= self.lr * g


class Adam:
    """
    Controlador I adaptativo (Dai et al., 2023, Eqs. 3-5):
        m <- b1*m + (1-b1)*g
        v <- b2*v + (1-b2)*g^2
        m_hat = m/(1-b1^t),  v_hat = v/(1-b2^t)
        theta <- theta - lr * m_hat / (sqrt(v_hat) + eps)
    """
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            m_hat = self.m[i] / (1 - self.b1 ** self.t)
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class SGD_PID:
    """
    PID sobre SGD-Momentum (An et al., 2018, Eq. 10):
        V <- a*V - lr*g
        D <- a*D + (1-a)*(g - g_prev)        # media movel da variacao do gradiente
        theta <- theta + V + Kd*D
    Momentum (P+I) mais o termo derivativo D que antecipa a variacao do gradiente
    e reduz o overshoot.
    """
    def __init__(self, params, lr=0.1, alpha=0.9, Kd=1.0):
        self.lr, self.alpha, self.Kd = lr, alpha, Kd
        self.V = [np.zeros_like(p) for p in params]
        self.D = [np.zeros_like(p) for p in params]
        self.g_prev = [np.zeros_like(p) for p in params]

    def step(self, params, grads):
        for i, (p, g) in enumerate(zip(params, grads)):
            self.V[i] = self.alpha * self.V[i] - self.lr * g
            self.D[i] = self.alpha * self.D[i] + (1 - self.alpha) * (g - self.g_prev[i])
            p += self.V[i] + self.Kd * self.D[i]
            self.g_prev[i] = g.copy()


class Adam_PID:
    """
    Adam tratado como PID completo (Dai et al., 2023, Eq. 7):
        m, v, m_hat, v_hat  -> iguais ao Adam
        D = g - g_prev
        D_hat = lr/(1-gamma^t) * D
        theta <- theta
                 - (lr/(sqrt(v_hat)+eps)) * Kp * g        # termo P (adaptativo)
                 - (lr/(sqrt(v_hat)+eps)) * Ki * m_hat    # termo I (= Adam)
                 - Kd * D_hat                              # termo D (preditivo)
    Com Kp=0, Ki=1, Kd=0 recai EXATAMENTE no Adam.
    """
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8,
                 gamma=0.9, Kp=0.0, Ki=1.0, Kd=0.0):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.gamma, self.Kp, self.Ki, self.Kd = gamma, Kp, Ki, Kd
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.g_prev = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            m_hat = self.m[i] / (1 - self.b1 ** self.t)
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            denom = np.sqrt(v_hat) + self.eps
            D = g - self.g_prev[i]
            D_hat = (self.lr / (1 - self.gamma ** self.t)) * D
            p -= (self.lr / denom) * self.Kp * g          # P
            p -= (self.lr / denom) * self.Ki * m_hat      # I  (Adam)
            p -= self.Kd * D_hat                          # D
            self.g_prev[i] = g.copy()


# ----------------------------------------------------------------------------
# Laco de treino
# ----------------------------------------------------------------------------
def train(make_opt, data, epochs=20, batch=100, seed=0, eval_every=1,
          eval_train_n=10000, verbose=False):
    x_tr, y_tr, x_te, y_te = data
    net = MLP(seed=seed)                 # MESMA inicializacao para todos -> comparacao justa
    opt = make_opt(net.params())
    rng = np.random.default_rng(123)     # MESMA ordem de mini-batches para todos
    n = x_tr.shape[0]
    # amostra fixa para medir loss/acc de treino rapidamente
    etn = min(eval_train_n, n)
    eidx = np.random.default_rng(7).permutation(n)[:etn]
    xtr_eval, ytr_eval = x_tr[eidx], y_tr[eidx]
    hist = {"epoch": [], "train_loss": [], "train_acc": [],
            "test_loss": [], "test_acc": [],
            "cum_time": [], "cum_flops": []}

    # FLOPs por amostra: forward + backward de um MLP n_in-H-n_out.
    # Multiplicacao matriz-vetor de a->b custa ~2*a*b FLOPs (mult + soma).
    # Forward = 2*(in*H + H*out); backward ~ 2x o forward. Total ~ 6*(in*H + H*out).
    n_in, H, n_out = net.W1.shape[0], net.W1.shape[1], net.W2.shape[1]
    flops_per_sample = 6.0 * (n_in * H + H * n_out)

    cum_time = 0.0           # tempo de COMPUTO de treino acumulado (sem avaliacao)
    cum_flops = 0.0
    for ep in range(1, epochs + 1):
        idx = rng.permutation(n)
        t_ep = time.perf_counter()
        for s in range(0, n, batch):
            b = idx[s:s + batch]
            xb, yb = x_tr[b], y_tr[b]
            probs = net.forward(xb)
            grads = net.backward(onehot(yb))
            opt.step(net.params(), grads)
        cum_time += time.perf_counter() - t_ep
        cum_flops += flops_per_sample * n

        if ep % eval_every == 0 or ep == epochs:
            hist["cum_time"].append(cum_time)
            hist["cum_flops"].append(cum_flops)
            ptr = net.forward(xtr_eval)
            pte = net.forward(x_te)
            hist["epoch"].append(ep)
            hist["train_loss"].append(cross_entropy(ptr, ytr_eval))
            hist["train_acc"].append(accuracy(ptr, ytr_eval))
            hist["test_loss"].append(cross_entropy(pte, y_te))
            hist["test_acc"].append(accuracy(pte, y_te))
            if verbose:
                print(f"  ep {ep:3d}  trL {hist['train_loss'][-1]:.4f}  "
                      f"teAcc {hist['test_acc'][-1]*100:.2f}%")
    return hist
