"""
Part 2: Dense MLP baseline training script.
Trains a 3-layer MLP (~50,000+ parameters) on standardized digit classification,
logging loss and accuracy curves to results/part2_learning_curve.png and metrics to results/part2_metrics.json.
"""

import json
import os
import time
import matplotlib.pyplot as plt
import numpy as np

from engine.tensor import Tensor
from nn.layers import MLP
from nn.loss import SoftmaxCrossEntropy
from nn.optim import Adam
from train.data import DataLoader, load_digit_dataset


def evaluate(model: MLP, X: np.ndarray, y: np.ndarray, criterion: SoftmaxCrossEntropy) -> tuple[float, float]:
    """Evaluates model on entire array X, returning (mean_loss, accuracy)."""
    x_tensor = Tensor(X, requires_grad=False)
    logits = model(x_tensor)
    loss = criterion(logits, y).item()
    preds = np.argmax(logits.data, axis=1)
    acc = float(np.mean(preds == y))
    return loss, acc


def train_dense(
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
    results_dir: str = "results",
) -> dict:
    np.random.seed(seed)
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 60)
    print("PART 2: Dense MLP Baseline Training")
    print("=" * 60)

    # 1. Load standardized dataset
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_digit_dataset(seed=seed)
    train_loader = DataLoader(X_train, y_train, batch_size=batch_size, shuffle=True, seed=seed)

    # 2. Build 3-layer MLP: 64 -> 256 -> 128 -> 10 (~50,826 parameters)
    model = MLP(in_features=64, hidden_dims=[256, 128], out_features=10, activation="gelu")
    num_params = sum(p.size for p in model.parameters())
    print(f"Initialized MLP model with {num_params:,} parameters.")

    criterion = SoftmaxCrossEntropy()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_losses = []
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            logits = model(X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        train_loss, train_acc = evaluate(model, X_train, y_train, criterion)
        val_loss, val_acc = evaluate(model, X_val, y_val, criterion)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if epoch % 5 == 0 or epoch == epochs or epoch == 1:
            print(
                f"Epoch {epoch:02d}/{epochs:02d} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%"
            )

    train_duration = time.perf_counter() - start_time
    test_loss, test_acc = evaluate(model, X_test, y_test, criterion)
    print("-" * 60)
    print(f"Training completed in {train_duration:.2f}s.")
    print(f"Final Test Loss: {test_loss:.4f} | Final Test Accuracy: {test_acc * 100:.2f}%")

    metrics = {
        "model_architecture": "64 -> 256 -> 128 -> 10 (GELU)",
        "num_parameters": num_params,
        "epochs": epochs,
        "train_duration_sec": round(train_duration, 4),
        "test_loss": round(test_loss, 5),
        "test_accuracy": round(test_acc, 5),
        "final_val_loss": round(history["val_loss"][-1], 5),
        "final_val_accuracy": round(history["val_acc"][-1], 5),
        "history": history,
    }

    # Save metrics JSON
    metrics_path = os.path.join(results_dir, "part2_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # Generate Learning Curve Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    epochs_range = history["epoch"]

    # Loss curve
    ax1.plot(epochs_range, history["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
    ax1.plot(epochs_range, history["val_loss"], label="Val Loss", color="#ff7f0e", lw=2, linestyle="--")
    ax1.set_title("Dense MLP Loss Trajectory", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Softmax Cross-Entropy Loss", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=11)

    # Accuracy curve
    ax2.plot(epochs_range, [a * 100 for a in history["train_acc"]], label="Train Accuracy", color="#2ca02c", lw=2)
    ax2.plot(epochs_range, [a * 100 for a in history["val_acc"]], label="Val Accuracy", color="#d62728", lw=2, linestyle="--")
    ax2.axhline(test_acc * 100, color="purple", linestyle=":", label=f"Test Acc ({test_acc * 100:.1f}%)")
    ax2.set_title("Dense MLP Classification Accuracy", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=11)

    plt.tight_layout()
    plot_path = os.path.join(results_dir, "part2_learning_curve.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Learning curve plot saved to {plot_path}")

    return metrics


if __name__ == "__main__":
    train_dense()
