"""
Small 1D-CNN for per-minute apnea classification from RR-interval sequences.

Deliberately tiny (two conv layers, global average pool, two FC layers) --
per CLAUDE.md rule 4 ("start simple: classical -> 1D-CNN -> only if plateaued,
CNN-LSTM") and the CPU-only constraint (rule in Phase 3 instructions): this
must train quickly on a laptop CPU across many LOSO folds, not chase the
literature ceiling on the first try.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class RRCNN(nn.Module):
    def __init__(self, seq_len: int = 60, n_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, seq_len) -> logits: (batch,)
        z = self.features(x).squeeze(-1)
        return self.classifier(z).squeeze(-1)


def train_cnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seq_len: int = 60,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    seed: int = 0,
    pos_weight_factor: float = 1.0,
) -> RRCNN:
    """Train a fresh RRCNN from scratch on one fold's training data.

    X_train: (n, seq_len) for a single-channel model, or (n, channels, seq_len)
             for a multi-channel model. Raw units, NOT yet normalized --
             normalization stats must come from this same training fold only,
             done by the caller, to avoid any test-fold leakage.
    y_train: (n,) binary labels, 1=apnea.
    pos_weight_factor: multiplies the natural class-balance weight
             (n_neg/n_pos) by this factor. >1 deliberately biases the loss
             toward catching more apnea minutes (higher sensitivity) at the
             cost of more false positives (lower specificity) -- a fixed,
             documented design choice, not tuned against test-fold results.
    """
    torch.manual_seed(seed)
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([pos_weight_factor * n_neg / max(n_pos, 1)], dtype=torch.float32)

    n_channels = 1 if X_train.ndim == 2 else X_train.shape[1]
    model = RRCNN(seq_len=seq_len, n_channels=n_channels)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    X = torch.tensor(X_train, dtype=torch.float32)
    if X.ndim == 2:
        X = X.unsqueeze(1)  # (n, seq_len) -> (n, 1, seq_len)
    y = torch.tensor(y_train, dtype=torch.float32)
    n = len(X)

    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            xb, yb = X[idx], y[idx]
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

    return model


@torch.no_grad()
def predict_cnn(model: RRCNN, X: np.ndarray) -> np.ndarray:
    """Return apnea probabilities for X, already normalized (n, seq_len) or
    (n, channels, seq_len) matching how the model was trained."""
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32)
    if Xt.ndim == 2:
        Xt = Xt.unsqueeze(1)
    logits = model(Xt)
    return torch.sigmoid(logits).numpy()


class FeatureMLP(nn.Module):
    """Small feedforward net for scalar (already-aggregated) feature vectors,
    e.g. a per-minute HRV feature bank -- unlike RRCNN, this has no
    convolution/sequence assumption, so it's a fairer classifier for inputs
    that are a handful of unordered scalar features rather than a real
    within-minute time series."""

    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    seed: int = 0,
    pos_weight_factor: float = 1.0,
) -> FeatureMLP:
    """Train a fresh FeatureMLP from scratch on one fold's training data.

    X_train: (n, n_features), raw units -- normalization done by the caller
             from this fold's training stats only, same discipline as
             train_cnn.
    y_train: (n,) binary labels, 1=apnea.
    """
    torch.manual_seed(seed)
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([pos_weight_factor * n_neg / max(n_pos, 1)], dtype=torch.float32)

    model = FeatureMLP(n_features=X_train.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    X = torch.tensor(X_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.float32)
    n = len(X)

    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            xb, yb = X[idx], y[idx]
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

    return model


@torch.no_grad()
def predict_mlp(model: FeatureMLP, X: np.ndarray) -> np.ndarray:
    """Return apnea probabilities for X, already normalized (n, n_features)."""
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32)
    logits = model(Xt)
    return torch.sigmoid(logits).numpy()
