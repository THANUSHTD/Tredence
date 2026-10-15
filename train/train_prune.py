"""
Part 3: Dynamic Self-Pruning Training Script.
Trains a 3-layer MLP on digits to hit exactly 90.0% weight sparsity using
Zhu & Gupta cubic schedule with First-Order Taylor Saliency and RigL regrowth.
Saves metrics to results/part3_metrics.json and plots to results/part3_self_pruning.png.
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
from nn.sparse_infer import CSRLinear, benchmark_dense_vs_csr
from prune.criteria import TaylorSaliencyCriterion
from prune.pruner import Pruner
from train.data import DataLoader, load_digit_dataset


def evaluate(model: MLP, X: np.ndarray, y: np.ndarray, criterion: SoftmaxCrossEntropy) -> tuple[float, float]:
    """Evaluates model, returning (loss, accuracy)."""
    x_tensor = Tensor(X, requires_grad=False)
    logits = model(x_tensor)
    loss = criterion(logits, y).item()
    preds = np.argmax(logits.data, axis=1)
    acc = float(np.mean(preds == y))
    return loss, acc


def train_prune(
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    target_sparsity: float = 0.90,
    seed: int = 42,
    results_dir: str = "results",
) -> dict:
    np.random.seed(seed)
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 65)
    print(f"PART 3: Self-Pruning Training to {target_sparsity * 100:.1f}% Sparsity")
    print("=" * 65)

    # 1. Load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_digit_dataset(seed=seed)
    train_loader = DataLoader(X_train, y_train, batch_size=batch_size, shuffle=True, seed=seed)
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch

    # 2. Build model & components
    model = MLP(in_features=64, hidden_dims=[256, 128], out_features=10, activation="gelu")
    num_params = sum(p.size for p in model.parameters())
    print(f"Model parameters: {num_params:,} | Total training steps: {total_steps}")

    criterion = SoftmaxCrossEntropy()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    saliency_criterion = TaylorSaliencyCriterion(beta=0.9, lambda_floor=1e-4)

    # Sparsity schedule: warmup 5 epochs, prune until epoch 40, cooldown 10 epochs
    t_0 = 5 * steps_per_epoch
    t_end = 40 * steps_per_epoch
    delta_t = max(5, steps_per_epoch // 2)

    pruner = Pruner(
        model=model,
        optimizer=optimizer,
        criterion=saliency_criterion,
        target_sparsity=target_sparsity,
        initial_sparsity=0.0,
        t_0=t_0,
        t_end=t_end,
        delta_t=delta_t,
        regrowth_ratio=0.15,
        output_sparsity_cap=0.50,
    )

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "sparsity": [],
    }

    global_step = 0
    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        for X_b, y_b in train_loader:
            global_step += 1
            optimizer.zero_grad()
            logits = model(X_b)
            loss = criterion(logits, y_b)
            loss.backward()

            # Dynamic pruning & RigL regrowth
            pruner.step(global_step)

            # Mask-isolated Adam step
            optimizer.step()

        curr_s = pruner.get_sparsity()
        train_loss, train_acc = evaluate(model, X_train, y_train, criterion)
        val_loss, val_acc = evaluate(model, X_val, y_val, criterion)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["sparsity"].append(curr_s)

        if epoch % 5 == 0 or epoch == epochs or epoch == 1:
            print(
                f"Epoch {epoch:02d}/{epochs:02d} | "
                f"Sparsity: {curr_s * 100:.1f}% | "
                f"Train Acc: {train_acc * 100:.2f}% | "
                f"Val Acc: {val_acc * 100:.2f}% | "
                f"Val Loss: {val_loss:.4f}"
            )

    train_duration = time.perf_counter() - start_time
    test_loss, test_acc = evaluate(model, X_test, y_test, criterion)
    final_sparsity = pruner.get_sparsity()

    print("-" * 65)
    print(f"Training finished in {train_duration:.2f}s.")
    print(f"Final Model Sparsity: {final_sparsity * 100:.2f}%")
    print(f"Final Test Accuracy: {test_acc * 100:.2f}% | Test Loss: {test_loss:.4f}")

    # FLOP & Active weights calculation
    total_weights = sum(layer.weight.data.size for layer in model.layers)
    active_weights = sum(int(np.sum(layer.weight.mask)) for layer in model.layers)
    dense_macs = sum(2 * layer.weight.data.size for layer in model.layers)
    sparse_macs = sum(2 * int(np.sum(layer.weight.mask)) for layer in model.layers)
    flop_reduction = 1.0 - (sparse_macs / dense_macs)

    # Benchmark CSR sparse execution vs dense execution
    benchmark_layer = model.layers[1]  # 256 -> 128
    csr_bench = benchmark_dense_vs_csr(benchmark_layer, batch_sizes=[1, 16, 64], num_runs=50)

    metrics = {
        "model_architecture": "64 -> 256 -> 128 -> 10 (GELU)",
        "target_sparsity": target_sparsity,
        "final_sparsity": round(final_sparsity, 5),
        "total_weights": total_weights,
        "active_weights": active_weights,
        "dense_macs_per_sample": dense_macs,
        "sparse_macs_per_sample": sparse_macs,
        "flop_reduction": round(flop_reduction, 5),
        "test_loss": round(test_loss, 5),
        "test_accuracy": round(test_acc, 5),
        "final_val_accuracy": round(history["val_acc"][-1], 5),
        "train_duration_sec": round(train_duration, 4),
        "csr_benchmarks": csr_bench,
        "history": history,
    }

    # Save metrics JSON
    metrics_path = os.path.join(results_dir, "part3_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # Plot 3-panel figure: Loss, Accuracy, Sparsity Ramp
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
    epochs_range = history["epoch"]

    # Panel 1: Loss
    ax1.plot(epochs_range, history["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
    ax1.plot(epochs_range, history["val_loss"], label="Val Loss", color="#ff7f0e", lw=2, linestyle="--")
    ax1.set_title("Self-Pruned MLP Loss", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    # Panel 2: Accuracy
    ax2.plot(epochs_range, [a * 100 for a in history["train_acc"]], label="Train Acc", color="#2ca02c", lw=2)
    ax2.plot(epochs_range, [a * 100 for a in history["val_acc"]], label="Val Acc", color="#d62728", lw=2, linestyle="--")
    ax2.axhline(test_acc * 100, color="purple", linestyle=":", label=f"Test Acc ({test_acc * 100:.1f}%)")
    ax2.set_title("Classification Accuracy", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    # Panel 3: Sparsity Ramp
    ax3.plot(epochs_range, [s * 100 for s in history["sparsity"]], label="Achieved Sparsity", color="#9467bd", lw=2.5)
    ax3.axhline(target_sparsity * 100, color="black", linestyle="--", label=f"Target ({target_sparsity * 100:.0f}%)")
    ax3.set_title("Zhu & Gupta Cubic Sparsity Ramp", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Epoch", fontsize=11)
    ax3.set_ylabel("Weight Sparsity (%)", fontsize=11)
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend()

    plt.tight_layout()
    plot_path = os.path.join(results_dir, "part3_self_pruning.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Self-pruning curves plot saved to {plot_path}")

    return metrics


if __name__ == "__main__":
    train_prune()
