"""
LSTM Autoencoder for Anomaly Detection on Time-Series Data
===========================================================
Dataset: Synthetic multi-variate sensor data (CMAPSS-inspired).
         Swap `generate_synthetic_dataset()` with a real CMAPSS loader
         to use NASA's Turbofan Jet Engine dataset.

Architecture
------------
  Encoder  : LSTM  →  hidden state  →  bottleneck
  Decoder  : LSTM  →  reconstructed sequence

Anomaly Detection Strategy
--------------------------
  1.  Train only on "normal" data (labels == 0).
  2.  Compute per-sample MSE on the training set.
  3.  Threshold = mean(MSE) + k * std(MSE)  (default k = 3).
  4.  Any test sample with MSE > threshold is flagged as an anomaly.

Usage
-----
  python lstm_anomaly_detection.py          # full run with plots
  python lstm_anomaly_detection.py --help   # CLI options
"""

# ──────────────────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────────────────
import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Matplotlib is optional – gracefully skip plotting when unavailable
try:
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend (safe for scripts)
    import matplotlib.pyplot as plt
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility helper
# ──────────────────────────────────────────────────────────────────────────────
def seed_everything(seed: int = 42) -> None:
    """Fix all random seeds for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ──────────────────────────────────────────────────────────────────────────────
# 1. Synthetic Dataset Generator (CMAPSS-inspired)
# ──────────────────────────────────────────────────────────────────────────────
def generate_synthetic_dataset(
    n_normal: int = 800,
    n_anomaly: int = 200,
    n_features: int = 6,
    noise_std: float = 0.05,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic multivariate sensor dataset.

    Normal samples  : Gaussian noise around a slowly drifting sinusoidal
                      baseline — mimicking healthy engine cycles.
    Anomalous samples: Abrupt amplitude spikes / mean shifts on a random
                      subset of sensors — mimicking degradation events.

    Returns
    -------
    data   : float32 array of shape (N, n_features)
    labels : int array of shape (N,)  — 0 = normal, 1 = anomaly
    """
    rng = np.random.default_rng(seed)

    t_normal = np.linspace(0, 8 * np.pi, n_normal)
    normal_base = np.column_stack(
        [np.sin(t_normal + i * np.pi / n_features) for i in range(n_features)]
    )
    normal_data = normal_base + rng.normal(0, noise_std, normal_base.shape)

    # Anomalous: random mean shift + amplitude spike
    t_anom = np.linspace(0, 8 * np.pi, n_anomaly)
    anom_base = np.column_stack(
        [np.sin(t_anom + i * np.pi / n_features) for i in range(n_features)]
    )
    spikes = rng.choice([0.0, 1.0], size=anom_base.shape, p=[0.6, 0.4])
    anom_data = (
        anom_base
        + rng.normal(0, noise_std, anom_base.shape)
        + spikes * rng.uniform(1.5, 3.0, anom_base.shape)
    )

    data = np.vstack([normal_data, anom_data]).astype(np.float32)
    labels = np.hstack([np.zeros(n_normal, dtype=int), np.ones(n_anomaly, dtype=int)])

    # Shuffle together
    idx = rng.permutation(len(data))
    return data[idx], labels[idx]


# ──────────────────────────────────────────────────────────────────────────────
# 2. Sliding-Window Dataset
# ──────────────────────────────────────────────────────────────────────────────
class SlidingWindowDataset(Dataset):
    """
    Converts a 2-D time-series array (T × F) into overlapping windows.

    Parameters
    ----------
    data       : np.ndarray of shape (T, n_features)
    labels     : np.ndarray of shape (T,)  – one label per time-step
    window_size: length of each sliding window (sequence length fed to LSTM)
    stride     : step between consecutive windows (default = 1)

    Each item returns
    -----------------
    window : float32 Tensor of shape (window_size, n_features)
    label  : int Tensor – majority label inside the window (0 or 1)
    """

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        window_size: int = 30,
        stride: int = 1,
    ) -> None:
        super().__init__()
        assert len(data) == len(labels), "data and labels must have the same length"
        assert window_size > 0 and stride > 0, "window_size and stride must be positive"

        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.window_size = window_size
        self.stride = stride

        # Pre-compute valid start indices
        self.indices = list(range(0, len(data) - window_size + 1, stride))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = self.indices[idx]
        end = start + self.window_size
        window = self.data[start:end]                   # (window_size, n_features)
        # Majority vote over the window for the label
        label = self.labels[start:end].float().mean().round().long()
        return window, label

    def filter_normal(self) -> "SlidingWindowDataset":
        """
        Return a new dataset containing only windows whose label is 0 (normal).
        Used to train the autoencoder exclusively on normal behaviour.
        """
        clone = SlidingWindowDataset.__new__(SlidingWindowDataset)
        clone.data = self.data
        clone.labels = self.labels
        clone.window_size = self.window_size
        clone.stride = self.stride
        clone.indices = [i for i in self.indices if
                         self.labels[i:i + self.window_size].float().mean() < 0.5]
        return clone


# ──────────────────────────────────────────────────────────────────────────────
# 3. LSTM Autoencoder
# ──────────────────────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    """
    LSTM Encoder: compresses a sequence into a fixed-size bottleneck vector.

    Input  : (batch, seq_len, n_features)
    Output : (batch, hidden_size)  – last hidden state of the final layer
    """

    def __init__(self, n_features: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        _, (h_n, _) = self.lstm(x)   # h_n: (num_layers, B, hidden_size)
        return h_n[-1]               # take the topmost layer: (B, hidden_size)


class Decoder(nn.Module):
    """
    LSTM Decoder: reconstructs the original sequence from the bottleneck.

    Strategy: repeat the bottleneck vector `seq_len` times, then run through
    an LSTM, and project back to n_features with a linear head.

    Input  : (batch, hidden_size), seq_len (int)
    Output : (batch, seq_len, n_features)
    """

    def __init__(
        self,
        hidden_size: int,
        n_features: int,
        seq_len: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_layer = nn.Linear(hidden_size, n_features)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, hidden_size)  →  repeat to (B, T, hidden_size)
        z_rep = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.lstm(z_rep)                 # (B, T, hidden_size)
        return self.output_layer(out)             # (B, T, n_features)


class LSTMAutoencoder(nn.Module):
    """
    Full LSTM Autoencoder = Encoder + Decoder.

    Parameters
    ----------
    n_features  : number of sensor channels (input dimensionality)
    hidden_size : LSTM hidden-state size (bottleneck width)
    seq_len     : window length (required by the Decoder)
    num_layers  : number of stacked LSTM layers in each sub-module
    dropout     : dropout probability between LSTM layers (ignored for 1 layer)
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        seq_len: int = 30,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(n_features, hidden_size, num_layers, dropout)
        self.decoder = Decoder(hidden_size, n_features, seq_len, num_layers, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)      # (B, hidden_size)
        return self.decoder(z)   # (B, seq_len, n_features)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Training Loop
# ──────────────────────────────────────────────────────────────────────────────
def train(
    model: LSTMAutoencoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    checkpoint_path: str = "best_model.pt",
) -> dict:
    """
    Train the LSTM Autoencoder with early stopping via best-validation-loss
    model checkpointing.

    Returns
    -------
    history : dict with keys 'train_loss' and 'val_loss' (list per epoch)
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")

    print(f"\n{'─'*60}")
    print(f"  Training on: {device}")
    print(f"  Epochs: {epochs}  |  LR: {lr}  |  Checkpoint: {checkpoint_path}")
    print(f"{'─'*60}\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_losses = []
        for windows, _ in train_loader:
            windows = windows.to(device)
            optimizer.zero_grad()
            reconstructed = model(windows)
            loss = criterion(reconstructed, windows)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        val_losses = []
        with torch.no_grad():
            for windows, _ in val_loader:
                windows = windows.to(device)
                reconstructed = model(windows)
                loss = criterion(reconstructed, windows)
                val_losses.append(loss.item())

        avg_train = float(np.mean(train_losses))
        avg_val = float(np.mean(val_losses))
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)

        scheduler.step(avg_val)

        # ── Checkpoint ─────────────────────────────────────────────────────
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), checkpoint_path)
            saved_marker = " ✓ saved"
        else:
            saved_marker = ""

        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch:>4}/{epochs}  |  "
            f"Train Loss: {avg_train:.6f}  |  "
            f"Val Loss: {avg_val:.6f}  |  "
            f"{elapsed:.1f}s{saved_marker}"
        )

    print(f"\n  Best validation loss: {best_val_loss:.6f}")
    return history


# ──────────────────────────────────────────────────────────────────────────────
# 5. Anomaly Threshold Computation
# ──────────────────────────────────────────────────────────────────────────────
def compute_reconstruction_errors(
    model: LSTMAutoencoder,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """
    Compute per-sample reconstruction MSE for every window in `loader`.

    Returns
    -------
    errors : float32 ndarray of shape (N,)
    """
    criterion = nn.MSELoss(reduction="none")
    model.eval()
    all_errors = []

    with torch.no_grad():
        for windows, _ in loader:
            windows = windows.to(device)
            reconstructed = model(windows)
            # MSE per sample: mean over (seq_len, n_features) dims
            error = criterion(reconstructed, windows).mean(dim=(1, 2))
            all_errors.append(error.cpu().numpy())

    return np.concatenate(all_errors, axis=0).astype(np.float32)


def compute_threshold(
    model: LSTMAutoencoder,
    train_loader: DataLoader,
    device: torch.device,
    k: float = 3.0,
) -> Tuple[float, np.ndarray]:
    """
    Derive an anomaly detection threshold from the training-set MSE distribution.

    Formula:  threshold = mean(errors) + k * std(errors)

    A larger k → fewer false positives but may miss subtle anomalies.
    A smaller k → more sensitive but higher false-positive rate.

    Parameters
    ----------
    model        : trained LSTMAutoencoder
    train_loader : DataLoader over *normal* training windows
    device       : torch device
    k            : number of standard deviations above the mean

    Returns
    -------
    threshold : scalar float
    errors    : per-sample MSE array (useful for visualisation)
    """
    errors = compute_reconstruction_errors(model, train_loader, device)
    mean_err = errors.mean()
    std_err = errors.std()
    threshold = float(mean_err + k * std_err)

    print(f"\n  Threshold Analysis (k={k})")
    print(f"    Training MSE — mean: {mean_err:.6f}  std: {std_err:.6f}")
    print(f"    Anomaly threshold  : {threshold:.6f}")
    return threshold, errors


# ──────────────────────────────────────────────────────────────────────────────
# 6. Evaluation
# ──────────────────────────────────────────────────────────────────────────────
def evaluate(
    model: LSTMAutoencoder,
    test_loader: DataLoader,
    threshold: float,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Run inference on the test set and report detection metrics.

    Returns
    -------
    errors        : per-sample MSE (N,)
    true_labels   : ground-truth labels (N,)
    metrics       : dict with precision, recall, F1, accuracy
    """
    errors = compute_reconstruction_errors(model, test_loader, device)
    predictions = (errors > threshold).astype(int)

    # Collect true labels in same order
    true_labels = np.concatenate(
        [labels.numpy() for _, labels in test_loader], axis=0
    )

    tp = int(((predictions == 1) & (true_labels == 1)).sum())
    fp = int(((predictions == 1) & (true_labels == 0)).sum())
    fn = int(((predictions == 0) & (true_labels == 1)).sum())
    tn = int(((predictions == 0) & (true_labels == 0)).sum())

    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    accuracy  = (tp + tn) / len(true_labels)

    metrics = {
        "precision": precision, "recall": recall,
        "f1": f1, "accuracy": accuracy,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }

    print(f"\n{'─'*60}")
    print("  Test Set Evaluation")
    print(f"{'─'*60}")
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"{'─'*60}\n")

    return errors, true_labels, metrics


# ──────────────────────────────────────────────────────────────────────────────
# 7. Visualisation
# ──────────────────────────────────────────────────────────────────────────────
def plot_results(
    history: dict,
    train_errors: np.ndarray,
    test_errors: np.ndarray,
    test_labels: np.ndarray,
    threshold: float,
    save_dir: str = ".",
) -> None:
    """Save three diagnostic plots to `save_dir`."""
    if not PLOT_AVAILABLE:
        print("  [INFO] matplotlib not installed – skipping plots.")
        return

    os.makedirs(save_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    # ── Plot 1: Training & validation loss ───────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["train_loss"], label="Train Loss", linewidth=2)
    ax.plot(history["val_loss"], label="Val Loss", linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("LSTM Autoencoder — Training Curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(save_dir) / "training_curves.png", dpi=120)
    plt.close(fig)

    # ── Plot 2: Training MSE distribution & threshold ────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(train_errors, bins=60, color="steelblue", alpha=0.7, label="Train MSE")
    ax.axvline(threshold, color="crimson", linewidth=2,
               linestyle="--", label=f"Threshold = {threshold:.4f}")
    ax.set_xlabel("Reconstruction MSE")
    ax.set_ylabel("Count")
    ax.set_title("Training Error Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(save_dir) / "error_distribution.png", dpi=120)
    plt.close(fig)

    # ── Plot 3: Test errors vs threshold (anomaly detection) ─────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    normal_idx = np.where(test_labels == 0)[0]
    anomaly_idx = np.where(test_labels == 1)[0]
    ax.scatter(normal_idx, test_errors[normal_idx],
               s=10, alpha=0.5, color="steelblue", label="Normal")
    ax.scatter(anomaly_idx, test_errors[anomaly_idx],
               s=15, alpha=0.7, color="orange", label="Anomaly (ground truth)")
    ax.axhline(threshold, color="crimson", linewidth=1.5,
               linestyle="--", label=f"Threshold = {threshold:.4f}")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Reconstruction MSE")
    ax.set_title("Anomaly Detection on Test Set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(save_dir) / "anomaly_detection.png", dpi=120)
    plt.close(fig)

    print(f"  Plots saved to: {Path(save_dir).resolve()}")


# ──────────────────────────────────────────────────────────────────────────────
# 8. Main Pipeline
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LSTM Autoencoder — Time-Series Anomaly Detection"
    )
    parser.add_argument("--epochs",       type=int,   default=50,     help="Training epochs (default: 50)")
    parser.add_argument("--batch_size",   type=int,   default=64,     help="Batch size (default: 64)")
    parser.add_argument("--lr",           type=float, default=1e-3,   help="Learning rate (default: 1e-3)")
    parser.add_argument("--window_size",  type=int,   default=30,     help="Sliding window length (default: 30)")
    parser.add_argument("--stride",       type=int,   default=1,      help="Window stride (default: 1)")
    parser.add_argument("--hidden_size",  type=int,   default=64,     help="LSTM hidden size (default: 64)")
    parser.add_argument("--num_layers",   type=int,   default=2,      help="LSTM stacked layers (default: 2)")
    parser.add_argument("--dropout",      type=float, default=0.2,    help="LSTM dropout (default: 0.2)")
    parser.add_argument("--k",            type=float, default=3.0,    help="Threshold k-sigma (default: 3.0)")
    parser.add_argument("--n_normal",     type=int,   default=800,    help="# normal samples (default: 800)")
    parser.add_argument("--n_anomaly",    type=int,   default=200,    help="# anomaly samples (default: 200)")
    parser.add_argument("--n_features",   type=int,   default=6,      help="# sensor features (default: 6)")
    parser.add_argument("--seed",         type=int,   default=42,     help="Random seed (default: 42)")
    parser.add_argument("--checkpoint",   type=str,   default="best_model.pt", help="Checkpoint path")
    parser.add_argument("--plot_dir",     type=str,   default="plots",         help="Directory for plots")
    parser.add_argument("--no_plots",     action="store_true",                  help="Disable plot generation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 1. Generate / load data ──────────────────────────────────────────
    print("\n[1/6] Generating synthetic multivariate sensor data …")
    data, labels = generate_synthetic_dataset(
        n_normal=args.n_normal,
        n_anomaly=args.n_anomaly,
        n_features=args.n_features,
        seed=args.seed,
    )
    print(f"      Total samples : {len(data)}")
    print(f"      Normal        : {(labels == 0).sum()}")
    print(f"      Anomalous     : {(labels == 1).sum()}")

    # ── 2. Normalise (z-score) ───────────────────────────────────────────
    # Fit scaler only on normal samples to avoid data leakage
    normal_mask = labels == 0
    mean_ = data[normal_mask].mean(axis=0, keepdims=True)
    std_  = data[normal_mask].std(axis=0,  keepdims=True) + 1e-8
    data  = (data - mean_) / std_

    # ── 3. Train / val / test split (60% train, 20% val, 20% test) ───────
    n = len(data)
    n_train = int(0.6 * n)
    n_val   = int(0.2 * n)

    train_data,   train_labels   = data[:n_train],          labels[:n_train]
    val_data,     val_labels     = data[n_train:n_train+n_val], labels[n_train:n_train+n_val]
    test_data,    test_labels    = data[n_train+n_val:],    labels[n_train+n_val:]

    print(f"\n[2/6] Creating sliding-window datasets (window={args.window_size}, stride={args.stride}) …")

    full_train_ds = SlidingWindowDataset(train_data, train_labels, args.window_size, args.stride)
    # Train only on normal windows to model the "normal manifold"
    train_ds  = full_train_ds.filter_normal()
    val_ds    = SlidingWindowDataset(val_data,   val_labels,   args.window_size, args.stride).filter_normal()
    test_ds   = SlidingWindowDataset(test_data,  test_labels,  args.window_size, args.stride)

    print(f"      Train windows (normal only) : {len(train_ds)}")
    print(f"      Val   windows (normal only) : {len(val_ds)}")
    print(f"      Test  windows (all)         : {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False)

    # ── 4. Build model ───────────────────────────────────────────────────
    print(f"\n[3/6] Building LSTM Autoencoder …")
    model = LSTMAutoencoder(
        n_features=args.n_features,
        hidden_size=args.hidden_size,
        seq_len=args.window_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"      Trainable parameters: {total_params:,}")
    print(f"      Encoder hidden size : {args.hidden_size}")
    print(f"      LSTM layers         : {args.num_layers}")

    # ── 5. Train ─────────────────────────────────────────────────────────
    print(f"\n[4/6] Training …")
    history = train(
        model, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr,
        device=device, checkpoint_path=args.checkpoint,
    )

    # ── 6. Load best model & compute threshold ───────────────────────────
    print(f"\n[5/6] Computing anomaly threshold from best checkpoint …")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    threshold, train_errors = compute_threshold(
        model, train_loader, device, k=args.k
    )

    # ── Save model_config.json for the FastAPI server ────────────────────
    config_path = Path(args.checkpoint).with_name("model_config.json")
    model_config = {
        "n_features":  args.n_features,
        "hidden_size": args.hidden_size,
        "seq_len":     args.window_size,
        "num_layers":  args.num_layers,
        "dropout":     args.dropout,
        "threshold":   float(threshold),
        # z-score scaler params so the API can normalise raw inputs
        "scaler_mean": mean_.squeeze().tolist(),
        "scaler_std":  std_.squeeze().tolist(),
    }
    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=2)
    print(f"  Model config saved \u2192 {config_path}")

    # ── 7. Evaluate on test set ──────────────────────────────────────────
    print(f"\n[6/6] Evaluating on test set …")
    test_errors, test_true_labels, metrics = evaluate(
        model, test_loader, threshold, device
    )

    # ── 8. Plots ─────────────────────────────────────────────────────────
    if not args.no_plots:
        plot_results(
            history, train_errors, test_errors,
            test_true_labels, threshold, save_dir=args.plot_dir,
        )

    print("Done.\n")


if __name__ == "__main__":
    main()
