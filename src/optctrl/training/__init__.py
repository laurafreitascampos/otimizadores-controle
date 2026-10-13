"""Loop de treino generico (dados ja na memoria do dispositivo).

Migrado de compare_mnist_mlp.py (run_once/evaluate) e tune_mnist.py
(train_eval). Garantias de comparacao justa herdadas do Lab de IA:
  - set_seed(seed) antes de criar o modelo -> mesma inicializacao para todos
    os otimizadores com a mesma semente;
  - a ordem dos mini-lotes vem do mesmo gerador -> mesma sequencia de lotes;
  - perda nao finita interrompe a execucao e marca diverged=True.
"""

import math
import random
import time

import numpy as np
import torch
import torch.nn as nn

from ..metrics import epochs_to_target


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, X, Y, criterion, batch=1000):
    model.eval()
    loss_sum, correct = 0.0, 0
    for i in range(0, X.size(0), batch):
        out = model(X[i:i + batch])
        yb = Y[i:i + batch]
        loss_sum += criterion(out, yb).item() * yb.size(0)
        correct += (out.argmax(1) == yb).sum().item()
    return loss_sum / X.size(0), correct / X.size(0)


def train_eval(model_fn, opt_fn, train, eval_set, *, epochs, batch_size, seed,
               target=None, clip=0.0, device='cpu', verbose=True, label=''):
    """Treina um modelo com um otimizador e avalia em eval_set a cada epoca.

    model_fn(): -> nn.Module ; opt_fn(params): -> Optimizer
    Retorna um dict com as curvas por epoca e metricas de convergencia.
    """
    Xtr, Ytr = train
    Xev, Yev = eval_set
    set_seed(seed)
    model = model_fn().to(device)
    opt = opt_fn(model.parameters())
    criterion = nn.CrossEntropyLoss()
    N = Xtr.size(0)

    hist = {'seed': seed, 'train_loss': [], 'train_acc': [], 'eval_loss': [],
            'eval_acc': [], 'epoch_time_s': [], 'diverged': False}

    for epoch in range(epochs):
        model.train()
        t0 = time.perf_counter()
        perm = torch.randperm(N, device=Xtr.device)
        run_loss, run_correct = 0.0, 0

        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = criterion(out, yb)
            loss_val = loss.item()
            if not math.isfinite(loss_val):
                hist['diverged'] = True
                break
            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            run_loss += loss_val * yb.size(0)
            run_correct += (out.argmax(1) == yb).sum().item()

        if hist['diverged']:
            if verbose:
                print(f"  !! [{label} s{seed}] perda nao finita na epoca {epoch + 1}")
            break

        if Xtr.is_cuda:
            torch.cuda.synchronize()
        hist['epoch_time_s'].append(time.perf_counter() - t0)   # so treino, sem avaliacao
        ev_loss, ev_acc = evaluate(model, Xev, Yev, criterion)
        hist['train_loss'].append(run_loss / N)
        hist['train_acc'].append(run_correct / N)
        hist['eval_loss'].append(ev_loss)
        hist['eval_acc'].append(ev_acc)

        if verbose:
            print(f"  [{label:>14} s{seed}] ep {epoch + 1:3d}/{epochs} "
                  f"train {run_loss / N:.4f}/{run_correct / N:.4f} | "
                  f"eval {ev_loss:.4f}/{ev_acc:.4f} | {hist['epoch_time_s'][-1]:.2f}s")

    hist['epochs_to_target'] = (epochs_to_target(hist['eval_acc'], target)
                                if target is not None else None)
    return hist


def aggregate(seed_hists):
    """Media e desvio entre sementes (descarta execucoes que divergiram)."""
    ok = [h for h in seed_hists if not h['diverged'] and h['eval_acc']]
    agg = {'n_seeds': len(seed_hists), 'n_completed': len(ok),
           'final_acc': [h['eval_acc'][-1] for h in ok],
           'best_acc': [max(h['eval_acc']) for h in ok],
           'epochs_to_target': [h['epochs_to_target'] for h in ok],
           'curves': {}}
    if not ok:
        return agg
    n_ep = min(len(h['eval_acc']) for h in ok)
    for key in ('train_loss', 'eval_loss', 'train_acc', 'eval_acc', 'epoch_time_s'):
        arr = np.array([h[key][:n_ep] for h in ok])
        agg['curves'][key] = {'mean': arr.mean(0).tolist(), 'std': arr.std(0).tolist()}
    return agg
