"""A CNN inspection module for wafer maps.

The gradient-boosted model over handcrafted features was fine for a
manufacturing venue, but the drift-detection literature this paper now argues
with assumes deep classifiers throughout, and a referee will not accept a
comparison where the monitored model is a tree ensemble. This module replaces
it.

The monitoring layer downstream is untouched: it consumes class probabilities
and nothing else. That the inspection model can be swapped from a tree ensemble
to a CNN without changing a line of the monitor is a property worth stating in
the paper, not an implementation detail.

Encoding: each wafer map becomes a 2-channel image on a fixed 32x32 grid, one
channel for "die is on the wafer", one for "die failed". Two channels rather
than a single 0/1/2 map because the raw encoding implies an ordering between
off-wafer, good and failed dies that does not exist. Resampling is
nearest-neighbour --- interpolating between die states would invent dies.
"""

from __future__ import annotations

from dataclasses import dataclass

import pathlib

import numpy as np
import torch
import torch.nn as nn
from skimage.transform import resize
from sklearn.metrics import classification_report, confusion_matrix, f1_score

GRID = 32


def encode(wafer_map: np.ndarray) -> np.ndarray:
    """One wafer map -> (2, GRID, GRID) float32."""
    m = np.asarray(wafer_map)
    valid = (m > 0).astype(np.float32)
    fail = (m == 2).astype(np.float32)
    out = np.empty((2, GRID, GRID), dtype=np.float32)
    for i, plane in enumerate((valid, fail)):
        out[i] = resize(
            plane, (GRID, GRID), order=0, preserve_range=True, anti_aliasing=False
        ).astype(np.float32)
    return out


def encode_batch(wafer_maps, progress_every: int = 0) -> np.ndarray:
    maps = list(wafer_maps)
    out = np.empty((len(maps), 2, GRID, GRID), dtype=np.float32)
    for i, m in enumerate(maps):
        out[i] = encode(m)
        if progress_every and i and i % progress_every == 0:
            print(f"  encode {i:,}/{len(maps):,}", flush=True)
    return out


class WaferCNN(nn.Module):
    """Small convolutional classifier.

    Deliberately modest --- three conv blocks, ~200k parameters. The paper's
    claim is about the monitoring layer, and an oversized backbone would invite
    the objection that results depend on a particular architecture. Batch
    normalisation is included because the class imbalance is severe enough
    (~85% of labelled wafers are 'none') that training is unstable without it.
    """

    def __init__(self, n_classes: int = 9):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3),
            nn.Linear(64 * 16, 128), nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


@dataclass
class CNNInspectionModel:
    model: WaferCNN
    classes: list[str]

    def predict_proba(self, images: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """Class probabilities for an array of encoded wafer maps."""
        self.model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = torch.from_numpy(np.asarray(images[i : i + batch_size]))
                out.append(torch.softmax(self.model(batch), dim=1).numpy())
        return np.vstack(out) if out else np.zeros((0, len(self.classes)))


def train_cnn(
    images: np.ndarray,
    labels: list[str],
    classes: list[str],
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 0,
    verbose: bool = True,
    checkpoint: str | None = None,
    resume_from: int = 0,
) -> CNNInspectionModel:
    """Train with class-balanced cross-entropy.

    Balanced weights rather than resampling: at ~85% 'none', oversampling the
    rare classes by the required factor would show the same few Near-full
    wafers hundreds of times and memorise them. Weighting leaves the data
    alone.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    index = {c: i for i, c in enumerate(classes)}
    y = np.array([index[str(l)] for l in labels], dtype=np.int64)

    counts = np.bincount(y, minlength=len(classes)).astype(float)
    weights = np.where(counts > 0, len(y) / (np.maximum(counts, 1) * len(classes)), 0.0)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))

    model = WaferCNN(len(classes))
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    # Foreground runs here are time-capped, so training checkpoints and resumes
    # rather than restarting. The optimiser and scheduler state are saved too;
    # restoring weights alone would silently reset Adam's moments and the
    # cosine schedule, which is not the same training run.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    x = torch.from_numpy(np.asarray(images, dtype=np.float32))
    y_t = torch.from_numpy(y)
    n = len(x)

    if resume_from and checkpoint and pathlib.Path(checkpoint).exists():
        state = torch.load(checkpoint, weights_only=False)
        model.load_state_dict(state["model"])
        optimiser.load_state_dict(state["optimiser"])
        scheduler.load_state_dict(state["scheduler"])
        if verbose:
            print(f"  resumed at epoch {state['epoch'] + 1}", flush=True)

    for epoch in range(resume_from, epochs):
        model.train()
        order = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = order[i : i + batch_size]
            optimiser.zero_grad()
            loss = criterion(model(x[idx]), y_t[idx])
            loss.backward()
            optimiser.step()
            total += loss.item() * len(idx)
        scheduler.step()
        if verbose:
            print(f"  epoch {epoch + 1:>3}/{epochs}  loss {total / n:.4f}", flush=True)
        if checkpoint:
            torch.save({"model": model.state_dict(),
                        "optimiser": optimiser.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "epoch": epoch}, checkpoint)

    return CNNInspectionModel(model=model, classes=list(classes))


def evaluate(model: CNNInspectionModel, images: np.ndarray, labels: list[str]) -> dict:
    proba = model.predict_proba(images)
    predicted = [model.classes[i] for i in proba.argmax(axis=1)]
    return {
        "macro_f1": float(f1_score(labels, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(labels, predicted, average="weighted", zero_division=0)
        ),
        "report": classification_report(labels, predicted, zero_division=0),
        "confusion": confusion_matrix(labels, predicted, labels=model.classes).tolist(),
    }
