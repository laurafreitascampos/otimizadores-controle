"""
conv.py
CNN "crua" em NumPy para o CIFAR-10, no mesmo espirito do core.py: cada camada
implementa forward e backward NA MAO, para enxergar todo o processo de aprendizado.

Arquitetura (pequena de proposito, p/ caber em NumPy/CPU):
  entrada (3, 32, 32)
  -> Conv 3->8   (3x3, pad 1) -> ReLU -> MaxPool 2 -> (8, 16, 16)
  -> Conv 8->16  (3x3, pad 1) -> ReLU -> MaxPool 2 -> (16, 8, 8)
  -> Flatten (1024) -> Linear 1024->10 -> softmax

Os QUATRO otimizadores do core.py (SGD, Adam, SGD_PID, Adam_PID) funcionam SEM
mudar nada: a CNN expoe params()/backward() com a mesma interface da MLP
(listas de arrays alinhadas, modificadas in-place pelos otimizadores).

Truque de desempenho: a convolucao e' feita por im2col (transforma a convolucao
em UMA multiplicacao de matrizes), o que e' ordens de magnitude mais rapido que
lacos explicitos -- mas continua sendo NumPy puro, sem nenhuma biblioteca de ML.
"""
import time
import numpy as np

import core  # reaproveita onehot, cross_entropy, accuracy e os otimizadores


# ----------------------------------------------------------------------------
# im2col / col2im: o coracao da convolucao vetorizada
# ----------------------------------------------------------------------------
def im2col(x, k, stride, pad):
    """
    Reorganiza cada janela kxk da imagem como uma LINHA de uma matriz.
    x: (N, C, H, W)  ->  cols: (N*oh*ow, C*k*k)
    Assim a convolucao vira: cols @ W_reshaped.T  (uma matmul).
    """
    N, C, H, W = x.shape
    oh = (H + 2 * pad - k) // stride + 1
    ow = (W + 2 * pad - k) // stride + 1
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    cols = np.zeros((N, C, k, k, oh, ow), dtype=x.dtype)
    for i in range(k):
        i_max = i + stride * oh
        for j in range(k):
            j_max = j + stride * ow
            cols[:, :, i, j, :, :] = xp[:, :, i:i_max:stride, j:j_max:stride]
    cols = cols.transpose(0, 4, 5, 1, 2, 3).reshape(N * oh * ow, C * k * k)
    return cols, oh, ow


def col2im(cols, x_shape, k, stride, pad, oh, ow):
    """Inverso do im2col: acumula os gradientes das janelas de volta na imagem."""
    N, C, H, W = x_shape
    cols = cols.reshape(N, oh, ow, C, k, k).transpose(0, 3, 4, 5, 1, 2)
    Hp, Wp = H + 2 * pad, W + 2 * pad
    xp = np.zeros((N, C, Hp, Wp), dtype=cols.dtype)
    for i in range(k):
        i_max = i + stride * oh
        for j in range(k):
            j_max = j + stride * ow
            xp[:, :, i:i_max:stride, j:j_max:stride] += cols[:, :, i, j, :, :]
    if pad == 0:
        return xp
    return xp[:, :, pad:pad + H, pad:pad + W]


# ----------------------------------------------------------------------------
# Camadas -- cada uma com forward e backward explicitos
# ----------------------------------------------------------------------------
class Conv2D:
    """
    Convolucao 2D. Forward:  out = cols @ W.T + b   (via im2col).
    W tem forma (out_ch, in_ch, k, k); cada filtro varre a imagem inteira.
    """
    def __init__(self, in_ch, out_ch, k=3, stride=1, pad=1, seed=0, dtype=np.float32):
        rng = np.random.default_rng(seed)
        fan_in = in_ch * k * k
        # Inicializacao He (ReLU)
        self.W = (rng.standard_normal((out_ch, in_ch, k, k))
                  * np.sqrt(2.0 / fan_in)).astype(dtype)
        self.b = np.zeros(out_ch, dtype=dtype)
        self.k, self.stride, self.pad = k, stride, pad

    def forward(self, x):
        self.x_shape = x.shape
        cols, oh, ow = im2col(x, self.k, self.stride, self.pad)
        self.cols = cols
        N, out_ch = x.shape[0], self.W.shape[0]
        Wr = self.W.reshape(out_ch, -1)                 # (out_ch, in_ch*k*k)
        out = cols @ Wr.T + self.b                       # (N*oh*ow, out_ch)
        return out.reshape(N, oh, ow, out_ch).transpose(0, 3, 1, 2)

    def backward(self, dout):
        N, out_ch, oh, ow = dout.shape
        dout_r = dout.transpose(0, 2, 3, 1).reshape(-1, out_ch)   # (N*oh*ow, out_ch)
        Wr = self.W.reshape(out_ch, -1)
        self.dW = (dout_r.T @ self.cols).reshape(self.W.shape)    # grad do filtro
        self.db = dout_r.sum(axis=0)                              # grad do bias
        dcols = dout_r @ Wr                                       # grad p/ a entrada
        return col2im(dcols, self.x_shape, self.k, self.stride, self.pad, oh, ow)

    def params(self):
        return [self.W, self.b]

    def grads(self):
        return [self.dW, self.db]


class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask

    def params(self):
        return []

    def grads(self):
        return []


class MaxPool2D:
    """Pooling maximo NAO sobreposto (k = stride). Reduz a resolucao pela metade."""
    def __init__(self, k=2):
        self.k = k

    def forward(self, x):
        N, C, H, W = x.shape
        k = self.k
        oh, ow = H // k, W // k
        self.x_shape = x.shape
        xr = x.reshape(N, C, oh, k, ow, k)
        out = xr.max(axis=(3, 5))
        # guarda quais posicoes foram o maximo (p/ rotear o gradiente no backward)
        self.mask = (xr == out[:, :, :, None, :, None])
        return out

    def backward(self, dout):
        k = self.k
        dout_b = dout[:, :, :, None, :, None]
        # empates (raros): divide o gradiente igualmente entre os maximos iguais
        counts = self.mask.sum(axis=(3, 5), keepdims=True)
        dx = self.mask * dout_b / counts
        return dx.reshape(self.x_shape)

    def params(self):
        return []

    def grads(self):
        return []


class Flatten:
    def forward(self, x):
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.shape)

    def params(self):
        return []

    def grads(self):
        return []


class Linear:
    """Camada densa: y = x @ W + b (mesma matematica da MLP do core.py)."""
    def __init__(self, n_in, n_out, seed=0, dtype=np.float32):
        rng = np.random.default_rng(seed)
        self.W = (rng.standard_normal((n_in, n_out))
                  * np.sqrt(2.0 / n_in)).astype(dtype)
        self.b = np.zeros(n_out, dtype=dtype)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        self.dW = self.x.T @ dout
        self.db = dout.sum(axis=0)
        return dout @ self.W.T

    def params(self):
        return [self.W, self.b]

    def grads(self):
        return [self.dW, self.db]


# ----------------------------------------------------------------------------
# Modelo: encadeia as camadas. Mesma interface da MLP (params/forward/backward)
# ----------------------------------------------------------------------------
class CNN:
    def __init__(self, n_classes=10, in_hw=32, ch1=8, ch2=16, seed=0, dtype=np.float32):
        self.conv1 = Conv2D(3, ch1, 3, pad=1, seed=seed + 1, dtype=dtype)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(2)
        self.conv2 = Conv2D(ch1, ch2, 3, pad=1, seed=seed + 2, dtype=dtype)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D(2)
        self.flat = Flatten()
        spatial = in_hw // 4                      # duas reducoes /2
        self.fc = Linear(ch2 * spatial * spatial, n_classes, seed=seed + 3, dtype=dtype)
        self.layers = [self.conv1, self.relu1, self.pool1,
                       self.conv2, self.relu2, self.pool2,
                       self.flat, self.fc]
        self._dims = (in_hw, ch1, ch2, n_classes)

    def forward(self, X):
        h = X
        for L in self.layers:
            h = L.forward(h)
        # softmax estavel (h sao os logits)
        z = h - h.max(axis=1, keepdims=True)
        e = np.exp(z)
        self.probs = e / e.sum(axis=1, keepdims=True)
        return self.probs

    def backward(self, y_onehot):
        n = y_onehot.shape[0]
        d = (self.probs - y_onehot) / n          # derivada cross-entropy+softmax
        for L in reversed(self.layers):
            d = L.backward(d)
        return self._grads()

    def params(self):
        ps = []
        for L in self.layers:
            ps += L.params()
        return ps

    def _grads(self):
        gs = []
        for L in self.layers:
            gs += L.grads()
        return gs

    def flops_per_sample(self):
        """FLOPs por amostra (forward+backward ~ 6*MACs), p/ a metrica de custo."""
        in_hw, ch1, ch2, ncls = self._dims
        h1 = in_hw
        h2 = in_hw // 2
        spatial = in_hw // 4
        f = 0.0
        f += 6.0 * (ch1 * h1 * h1) * (3 * 3 * 3)        # conv1
        f += 6.0 * (ch2 * h2 * h2) * (ch1 * 3 * 3)      # conv2
        f += 6.0 * (ch2 * spatial * spatial) * ncls     # fc
        return f


# ----------------------------------------------------------------------------
# Laco de treino (parente do core.train, adaptado p/ entrada 4D e CNN)
# ----------------------------------------------------------------------------
def train(make_opt, data, epochs=10, batch=128, seed=0, eval_every=1,
          eval_train_n=5000, max_train=None, verbose=False):
    x_tr, y_tr, x_te, y_te = data
    if max_train is not None:           # subamostra p/ tornar o NumPy tratavel
        x_tr, y_tr = x_tr[:max_train], y_tr[:max_train]

    net = CNN(seed=seed)                 # MESMA inicializacao p/ todos -> justo
    opt = make_opt(net.params())
    rng = np.random.default_rng(123)     # MESMA ordem de mini-batches p/ todos
    n = x_tr.shape[0]
    etn = min(eval_train_n, n)
    eidx = np.random.default_rng(7).permutation(n)[:etn]
    xtr_eval, ytr_eval = x_tr[eidx], y_tr[eidx]

    hist = {"epoch": [], "train_loss": [], "train_acc": [],
            "test_loss": [], "test_acc": [], "cum_time": [], "cum_flops": []}
    flops_per_sample = net.flops_per_sample()
    cum_time = 0.0
    cum_flops = 0.0

    for ep in range(1, epochs + 1):
        idx = rng.permutation(n)
        t_ep = time.perf_counter()
        for s in range(0, n, batch):
            b = idx[s:s + batch]
            net.forward(x_tr[b])
            grads = net.backward(core.onehot(y_tr[b]))
            opt.step(net.params(), grads)
        cum_time += time.perf_counter() - t_ep
        cum_flops += flops_per_sample * n

        if ep % eval_every == 0 or ep == epochs:
            hist["cum_time"].append(cum_time)
            hist["cum_flops"].append(cum_flops)
            ptr = net.forward(xtr_eval)
            pte = net.forward(x_te)
            hist["epoch"].append(ep)
            hist["train_loss"].append(core.cross_entropy(ptr, ytr_eval))
            hist["train_acc"].append(core.accuracy(ptr, ytr_eval))
            hist["test_loss"].append(core.cross_entropy(pte, y_te))
            hist["test_acc"].append(core.accuracy(pte, y_te))
            if verbose:
                print(f"  ep {ep:3d}  trL {hist['train_loss'][-1]:.4f}  "
                      f"teAcc {hist['test_acc'][-1] * 100:.2f}%", flush=True)
    return hist