"""Registro de datasets.

Datasets pequenos (MNIST e variantes) sao carregados INTEIROS na memoria do
dispositivo como tensores achatados e normalizados: sem DataLoader, sem
processos auxiliares (evita travamentos no Windows) e cada epoca leva fracao
de segundo. Migrado de compare_mnist_mlp.py / tune_mnist.py.

Protocolo de divisao (mesmo do Lab de IA):
  - val_size > 0: separa validacao do treino; o TESTE fica intocado ate a
    avaliacao final;
  - val_size = 0: treina no conjunto de treino inteiro (para replicar artigos
    que reportam direto no teste).

Datasets que nao cabem na GPU (CIFAR com aumento de dados, Tiny-ImageNet)
vao precisar de um caminho com DataLoader: TODO quando chegar no CIFAR.
"""

import torch

# Normalizacao + numero de classes. Todos 28x28 em tons de cinza -> 784.
TORCHVISION_SPECS = {
    'mnist':   {'cls': 'MNIST',        'mean': 0.1307, 'std': 0.3081, 'classes': 10},
    'fashion': {'cls': 'FashionMNIST', 'mean': 0.2860, 'std': 0.3530, 'classes': 10},
    'emnist':  {'cls': 'EMNIST',       'mean': 0.1751, 'std': 0.3332, 'classes': 47,
                'split': 'balanced'},
}


def _load_torchvision_flat(name, data_dir, device):
    import torchvision
    spec = TORCHVISION_SPECS[name]
    cls = getattr(torchvision.datasets, spec['cls'])
    kw = {'split': spec['split']} if 'split' in spec else {}
    train = cls(data_dir, train=True, download=True, **kw)
    test = cls(data_dir, train=False, download=True, **kw)

    def prep(ds):
        x = ds.data.float().div(255.0).sub(spec['mean']).div(spec['std'])
        return x.view(x.size(0), -1).to(device), ds.targets.to(device)

    return prep(train), prep(test), spec['classes']


def _load_synthetic(device, n_train=2000, n_test=500, dim=20, classes=3, seed=0):
    """Gaussianas separaveis. Serve para testes rapidos sem download."""
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(classes, dim, generator=g) * 3.0

    def make(n):
        y = torch.randint(0, classes, (n,), generator=g)
        x = centers[y] + torch.randn(n, dim, generator=g)
        return x.to(device), y.to(device)

    return make(n_train), make(n_test), classes


def load_splits(name, data_dir='./data', device='cpu', val_size=10000, split_seed=0):
    """Retorna dict com 'train', 'val' (ou None), 'test', 'num_classes', 'input_dim'."""
    name = name.lower()
    if name in TORCHVISION_SPECS:
        (Xtr, Ytr), (Xte, Yte), num_classes = _load_torchvision_flat(name, data_dir, device)
    elif name == 'synthetic':
        (Xtr, Ytr), (Xte, Yte), num_classes = _load_synthetic(device)
    else:
        raise ValueError(f"Dataset desconhecido '{name}'. "
                         f"Opcoes: {sorted(TORCHVISION_SPECS) + ['synthetic']}")

    val = None
    if val_size and val_size > 0:
        g = torch.Generator().manual_seed(split_seed)
        perm = torch.randperm(Xtr.size(0), generator=g).to(Xtr.device)
        val_idx, tr_idx = perm[:val_size], perm[val_size:]
        val = (Xtr[val_idx], Ytr[val_idx])
        Xtr, Ytr = Xtr[tr_idx], Ytr[tr_idx]

    return {'train': (Xtr, Ytr), 'val': val, 'test': (Xte, Yte),
            'num_classes': num_classes, 'input_dim': Xtr.size(1)}
