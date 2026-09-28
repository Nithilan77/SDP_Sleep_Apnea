"""
Small 2D-CNN for per-minute apnea classification from image-shaped ECG
representations (method 7: CWT scalogram of the RR series; method 8:
spectrogram of the raw ECG segment). Same "start simple" spirit as RRCNN
(cnn_model.py) -- two conv layers, global average pool, two FC layers --
just with Conv2d instead of Conv1d, since these inputs are genuine 2D
time-frequency images rather than 1D sequences. Kept deliberately separate
from cnn_model.py (RRCNN is the 1D-feature-methods' fixed classifier; this
is the 2D-image-methods' classifier, per the study's stated split of
"1D-feature methods share one classifier, image methods share a separate
2D-CNN, reported as its own sub-comparison").
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class ScalogramCNN(nn.Module):
    def __init__(self, n_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(n_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, H, W) -> logits: (batch,)
        z = self.features(x).flatten(1)
        return self.classifier(z).squeeze(-1)


def train_cnn2d(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    seed: int = 0,
    pos_weight_factor: float = 1.0,
) -> ScalogramCNN:
    """Train a fresh ScalogramCNN from scratch on one fold's training data.

    X_train: (n, H, W) for single-channel, or (n, channels, H, W). Raw
             units, NOT yet normalized -- normalization stats must come from
             this same training fold only, done by the caller.
    y_train: (n,) binary labels, 1=apnea.
    """
    torch.manual_seed(seed)
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([pos_weight_factor * n_neg / max(n_pos, 1)], dtype=torch.float32)

    n_channels = 1 if X_train.ndim == 3 else X_train.shape[1]
    model = ScalogramCNN(n_channels=n_channels)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    X = torch.tensor(X_train, dtype=torch.float32)
    if X.ndim == 3:
        X = X.unsqueeze(1)  # (n, H, W) -> (n, 1, H, W)
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
def predict_cnn2d(model: ScalogramCNN, X: np.ndarray) -> np.ndarray:
    """Return apnea probabilities for X, already normalized (n, H, W) or
    (n, channels, H, W) matching how the model was trained."""
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32)
    if Xt.ndim == 3:
        Xt = Xt.unsqueeze(1)
    logits = model(Xt)
    return torch.sigmoid(logits).numpy()
