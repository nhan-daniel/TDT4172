import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset


def seed_everything(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(False)


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_sizes: Sequence[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = in_features
        for size in hidden_sizes:
            layers.append(nn.Linear(last, size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last = size
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)


class WarmupCosineScheduler:
    """Manual scheduler with linear warmup followed by cosine decay."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        max_lr: float,
        min_lr: float = 0.0,
    ) -> None:
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if warmup_steps < 0 or warmup_steps >= total_steps:
            raise ValueError("warmup_steps must be in [0, total_steps)")
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self._step = 0
        self._set_lr(min_lr)

    def step(self) -> float:
        self._step += 1
        if self._step <= self.warmup_steps:
            progress = self._step / max(1, self.warmup_steps)
            lr = self.min_lr + progress * (self.max_lr - self.min_lr)
        else:
            progress = (self._step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        self._set_lr(lr)
        return lr

    def _set_lr(self, lr: float) -> None:
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    @property
    def step_count(self) -> int:
        return self._step


@dataclass
class DatasetBundle:
    train_loader: DataLoader
    test_loader: DataLoader
    mean: Tensor
    std: Tensor


def load_dataset(path: Path, batch_size: int) -> DatasetBundle:
    df = pd.read_csv(path)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    x_train = torch.from_numpy(train_df[["x0", "x1"]].values).float()
    y_train = torch.from_numpy(train_df["y"].values).float()
    x_test = torch.from_numpy(test_df[["x0", "x1"]].values).float()
    y_test = torch.from_numpy(test_df["y"].values).float()

    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    train_ds = TensorDataset(x_train, y_train)
    test_ds = TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return DatasetBundle(train_loader, test_loader, mean.squeeze(0), std.squeeze(0))


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features)
            loss = criterion(logits, targets)
            total_loss += loss.item() * features.size(0)
            preds = torch.sigmoid(logits) >= 0.5
            correct += (preds == targets.bool()).sum().item()
            total += targets.size(0)
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train(
    data_path: Path,
    epochs: int,
    batch_size: int,
    hidden_sizes: Sequence[int],
    dropout: float,
    warmup_epochs: float,
    max_lr: float,
    min_lr: float,
    weight_decay: float,
) -> None:
    seed_everything(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_dataset(data_path, batch_size)

    model = MLP(in_features=2, hidden_sizes=hidden_sizes, dropout=dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=min_lr, weight_decay=weight_decay)

    steps_per_epoch = len(bundle.train_loader)
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(steps_per_epoch * warmup_epochs)
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        max_lr=max_lr,
        min_lr=min_lr,
    )

    best_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for features, targets in bundle.train_loader:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            current_lr = scheduler.step()
            running_loss += loss.item() * features.size(0)
        train_loss = running_loss / len(bundle.train_loader.dataset)
        val_loss, val_acc = evaluate(model, bundle.test_loader, device)
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict()
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | lr={current_lr:.6f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, test_acc = evaluate(model, bundle.test_loader, device)
    print(f"Best test accuracy: {test_acc:.4f}, test loss: {test_loss:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an MLP on SkyNet data")
    parser.add_argument("--data", type=Path, default=Path("nn_data.csv"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--warmup-epochs", type=float, default=5.0, help="Warmup duration in epochs")
    parser.add_argument("--max-lr", type=float, default=5e-3)
    parser.add_argument("--min-lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_sizes=args.hidden_sizes,
        dropout=args.dropout,
        warmup_epochs=args.warmup_epochs,
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
    )
