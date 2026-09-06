# Model definitions
import numpy as np
import torch
import torch.nn as nn

# Bilstmencoder
class BiLSTMEncoder(nn.Module):
    # Init
    def __init__(self, nfeat: int, hidden: int = 128, layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.nfeat = nfeat
        self.hidden = hidden
        self.lstm = nn.LSTM(nfeat, hidden, num_layers=layers, batch_first=True,
                            bidirectional=True, dropout=dropout)
        self.head = nn.Linear(hidden * 2, 1)

    @property
    # Embedding Dim
    def embedding_dim(self):
        return self.hidden * 2

    # Encode
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return out[:, -1, :]

    # Forward
    def forward(self, x: torch.Tensor):
        emb = self.encode(x)
        return emb, self.head(emb).squeeze(-1)

# Faultfusionmlp
class FaultFusionMLP(nn.Module):
    # Init
    def __init__(self, stats_dim: int, emb_dim: int, n_faults: int = 8,
                 hidden: list = None, dropout: float = 0.25):
        super().__init__()
        hidden = hidden or [256, 128]
        in_dim = stats_dim + emb_dim + 1
        layers = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, n_faults))
        self.net = nn.Sequential(*layers)

    # Forward
    def forward(self, stats, if_score, emb):
        x = torch.cat([stats, if_score.unsqueeze(-1), emb], dim=-1)
        return self.net(x)

# Tokenblock
class _TokenBlock(nn.Module):
    # Init
    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    # Forward
    def forward(self, x):
        return self.drop(self.act(self.fc(x)))

# Pinball
def _pinball(pred: torch.Tensor, target: torch.Tensor, q: float) -> torch.Tensor:
    err = target - pred
    return torch.mean(torch.maximum(q * err, (q - 1.0) * err))

# Quantilemlp
class QuantileMLP(nn.Module):
    # Init
    def __init__(self, feat_dim: int, quantiles: tuple = (0.10, 0.50, 0.90),
                 hidden: list = None, dropout: float = 0.2):
        super().__init__()
        hidden = hidden or [256, 128]
        blocks = []
        prev = feat_dim
        for h in hidden:
            blocks.append(_TokenBlock(prev, h, dropout))
            prev = h
        self.torso = nn.Sequential(*blocks)
        heads = [nn.Linear(prev, 1) for _ in quantiles]
        self.heads = nn.ModuleList(heads)
        self.quantiles = list(quantiles)

    # Forward
    def forward(self, x: torch.Tensor):
        h = self.torso(x)
        return [head(h).squeeze(-1) for head in self.heads]

    # Loss
    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        preds = self.forward(x)
        y = y.float()
        return sum(_pinball(p, y, q) for p, q in zip(preds, self.quantiles))

# Healthmlp
class HealthMLP(nn.Module):
    # Init
    def __init__(self, feat_dim: int, hidden: list = None, dropout: float = 0.2):
        super().__init__()
        hidden = hidden or [256, 128]
        blocks = []
        prev = feat_dim
        for h in hidden:
            blocks.append(_TokenBlock(prev, h, dropout))
            prev = h
        blocks.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*blocks)

    # Forward
    def forward(self, x: torch.Tensor):
        return self.net(x).squeeze(-1)

# Gradient Features
def gradient_features(model: nn.Module, x: torch.Tensor, out_idx: int = 0):
    model.eval()
    x = x.clone().requires_grad_(True)
    out = model(x)
    if isinstance(out, (list, tuple)):
        out = out[out_idx]
    if out.dim() == 2:
        out = out[:, out_idx] if out_idx >= out.shape[1] else out
    grad = torch.autograd.grad(out.sum(), x, create_graph=False)[0]
    return (grad.abs() * x.abs()).detach().cpu().numpy()

# Tensor Loader
def tensor_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True):
    xt = torch.from_numpy(np.ascontiguousarray(x)).float()
    yt = torch.from_numpy(np.ascontiguousarray(y)).float().reshape(-1)
    ds = torch.utils.data.TensorDataset(xt, yt)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                       num_workers=0, drop_last=bool(shuffle))

# Seq Loader
def seq_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True):
    xt = torch.from_numpy(np.ascontiguousarray(x)).float()
    yt = torch.from_numpy(np.ascontiguousarray(y)).float().reshape(-1)
    ds = torch.utils.data.TensorDataset(xt, yt)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                       num_workers=0, drop_last=bool(shuffle))
