"""
Part 4: Multi-Seed Pareto Sweep and Empirical Baseline Comparison.
Sweeps target sparsities [0.0, 0.50, 0.75, 0.90, 0.95] across 3 deterministic seeds (42, 43, 44),
comparing First-Order Taylor Saliency + RigL Regrowth against Pure Magnitude Pruning.
Measures test accuracy, achieved sparsity, FLOP reduction, and CSR wall-clock latency.
Saves raw JSON to results/pareto_results.json and plots to results/pareto_curve.png and results/cost_measurement.png.
"""

import json
import os
import time
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np

from engine.tensor import Tensor
from nn.layers import MLP
from nn.loss import SoftmaxCrossEntropy
from nn.optim import Adam
from nn.sparse_infer import CSRLinear
from prune.criteria import MagnitudeCriterion, TaylorSaliencyCriterion
from prune.pruner import Pruner
from train.data import DataLoader, load_digit_dataset


def evaluate_accuracy(model: MLP, X: np.ndarray, y: np.ndarray) -> float:
    x_tensor = Tensor(X, requires_grad=False)
    logits = model(x_tensor)
    preds = np.argmax(logits.data, axis=1)
    return float(np.mean(preds == y))


def measure_sparse_latency(model: MLP, X: np.ndarray, num_runs: int = 40) -> Tuple[float, float]:
    """
    Measures wall-clock inference latency for Dense forward vs CSRLinear forward on test set X.
    Returns (dense_latency_ms, csr_latency_ms).
    """
    batch_size = X.shape[0]
    csr_layers = [CSRLinear.from_dense(layer) for layer in model.layers]

    # Dense timing
    x_dense = Tensor(X, requires_grad=False)
    for _ in range(5):  # warmup
        _ = model(x_dense)
    t0 = time.perf_counter()
    for _ in range(num_runs):
        _ = model(x_dense)
    dense_ms = (time.perf_counter() - t0) / num_runs * 1000.0

    # CSR sparse timing
    def csr_forward(x_arr: np.ndarray) -> np.ndarray:
        h = x_arr
        for i, csr_layer in enumerate(csr_layers):
            h = csr_layer(h)
            if i < len(model.activations):
                # Apply activation (GELU approx)
                c = 0.7978845608
                u = c * (h + 0.044715 * (h ** 3))
                h = 0.5 * h * (1.0 + np.tanh(u))
        return h

    for _ in range(5):  # warmup
        _ = csr_forward(X)
    t0 = time.perf_counter()
    for _ in range(num_runs):
        _ = csr_forward(X)
    csr_ms = (time.perf_counter() - t0) / num_runs * 1000.0

    return dense_ms, csr_ms


def run_single_experiment(
    method: str,
    target_sparsity: float,
    seed: int,
    epochs: int = 35,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> dict:
    np.random.seed(seed)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_digit_dataset(seed=seed)
    train_loader = DataLoader(X_train, y_train, batch_size=batch_size, shuffle=True, seed=seed)
    steps_per_epoch = len(train_loader)

    model = MLP(in_features=64, hidden_dims=[256, 128], out_features=10, activation="gelu")
    criterion = SoftmaxCrossEntropy()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if target_sparsity > 0.0:
        if method == "taylor_regrowth":
            prune_crit = TaylorSaliencyCriterion(beta=0.9, lambda_floor=1e-4)
            regrowth = 0.15
        elif method == "magnitude":
            prune_crit = MagnitudeCriterion()
            regrowth = 0.0
        else:
            raise ValueError(f"Unknown method {method}")

        t_0 = 4 * steps_per_epoch
        t_end = 28 * steps_per_epoch
        delta_t = max(4, steps_per_epoch // 2)

        pruner = Pruner(
            model=model,
            optimizer=optimizer,
            criterion=prune_crit,
            target_sparsity=target_sparsity,
            initial_sparsity=0.0,
            t_0=t_0,
            t_end=t_end,
            delta_t=delta_t,
            regrowth_ratio=regrowth,
            output_sparsity_cap=0.50,
        )
    else:
        pruner = None

    global_step = 0
    for epoch in range(1, epochs + 1):
        for X_b, y_b in train_loader:
            global_step += 1
            optimizer.zero_grad()
            logits = model(X_b)
            loss = criterion(logits, y_b)
            loss.backward()

            if pruner is not None:
                pruner.step(global_step)

            optimizer.step()

    test_acc = evaluate_accuracy(model, X_test, y_test)
    achieved_sparsity = pruner.get_sparsity() if pruner is not None else 0.0

    dense_macs = sum(2 * layer.weight.data.size for layer in model.layers)
    sparse_macs = sum(
        2 * (int(np.sum(layer.weight.mask)) if layer.weight.mask is not None else layer.weight.data.size)
        for layer in model.layers
    )
    flop_reduction = 1.0 - (sparse_macs / dense_macs)

    dense_ms, csr_ms = measure_sparse_latency(model, X_test)

    return {
        "method": method,
        "target_sparsity": target_sparsity,
        "seed": seed,
        "test_accuracy": test_acc,
        "achieved_sparsity": achieved_sparsity,
        "dense_macs": dense_macs,
        "sparse_macs": sparse_macs,
        "flop_reduction": flop_reduction,
        "dense_latency_ms": dense_ms,
        "csr_latency_ms": csr_ms,
        "speedup": dense_ms / csr_ms if csr_ms > 0 else 1.0,
    }


def run_pareto_sweep(
    sparsities: List[float] = [0.0, 0.50, 0.75, 0.90, 0.95],
    seeds: List[int] = [42, 43, 44],
    results_dir: str = "results",
) -> dict:
    os.makedirs(results_dir, exist_ok=True)
    methods = ["taylor_regrowth", "magnitude"]
    all_runs: List[dict] = []
    summary: Dict[str, Dict[str, dict]] = {m: {} for m in methods}

    print("=" * 70)
    print("PART 4: Pareto Sweep Across Sparsity Targets & Random Seeds")
    print("=" * 70)

    for method in methods:
        print(f"\n--- Method: {method.upper()} ---")
        for s in sparsities:
            acc_list = []
            s_list = []
            flop_red_list = []
            csr_time_list = []
            dense_time_list = []
            speedup_list = []

            for seed in seeds:
                res = run_single_experiment(method=method, target_sparsity=s, seed=seed)
                all_runs.append(res)
                acc_list.append(res["test_accuracy"])
                s_list.append(res["achieved_sparsity"])
                flop_red_list.append(res["flop_reduction"])
                csr_time_list.append(res["csr_latency_ms"])
                dense_time_list.append(res["dense_latency_ms"])
                speedup_list.append(res["speedup"])

            mean_acc = float(np.mean(acc_list))
            std_acc = float(np.std(acc_list))
            mean_s = float(np.mean(s_list))
            std_s = float(np.std(s_list))
            mean_flop_red = float(np.mean(flop_red_list))
            mean_csr_time = float(np.mean(csr_time_list))
            mean_speedup = float(np.mean(speedup_list))

            summary[method][str(s)] = {
                "target_sparsity": s,
                "mean_accuracy": round(mean_acc, 5),
                "std_accuracy": round(std_acc, 5),
                "mean_achieved_sparsity": round(mean_s, 5),
                "std_achieved_sparsity": round(std_s, 5),
                "mean_flop_reduction": round(mean_flop_red, 5),
                "mean_csr_latency_ms": round(mean_csr_time, 4),
                "mean_speedup": round(mean_speedup, 3),
            }

            print(
                f"Target Sparsity: {s * 100:4.1f}% | "
                f"Accuracy: {mean_acc * 100:.2f}% (+/- {std_acc * 100:.2f}%) | "
                f"FLOP Red: {mean_flop_red * 100:.1f}% | "
                f"CSR Speedup: {mean_speedup:.2f}x"
            )

    full_results = {
        "sweep_configuration": {
            "sparsities": sparsities,
            "seeds": seeds,
            "methods": methods,
        },
        "summary": summary,
        "raw_runs": all_runs,
    }

    # Save JSON results
    json_path = os.path.join(results_dir, "pareto_results.json")
    with open(json_path, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\nResults successfully saved to {json_path}")

    # 1. Publication Plot: Pareto Curve (Accuracy vs Sparsity with error bars)
    plt.figure(figsize=(9, 6), dpi=300)
    for method, label, color, marker in [
        ("taylor_regrowth", "Taylor Saliency + RigL Regrowth", "#1f77b4", "o"),
        ("magnitude", "Pure Magnitude Pruning", "#d62728", "s"),
    ]:
        s_vals = [summary[method][str(s)]["mean_achieved_sparsity"] * 100 for s in sparsities]
        acc_means = [summary[method][str(s)]["mean_accuracy"] * 100 for s in sparsities]
        acc_stds = [summary[method][str(s)]["std_accuracy"] * 100 for s in sparsities]

        plt.errorbar(
            s_vals,
            acc_means,
            yerr=acc_stds,
            label=label,
            color=color,
            marker=marker,
            capsize=4,
            lw=2.2,
            markersize=7,
        )

    plt.title("Pareto Frontier: Accuracy vs. Achieved Sparsity (3 Seeds)", fontsize=13, fontweight="bold")
    plt.xlabel("Achieved Weight Sparsity (%)", fontsize=11)
    plt.ylabel("Test Classification Accuracy (%)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11, loc="lower left")
    plt.tight_layout()
    pareto_plot_path = os.path.join(results_dir, "pareto_curve.png")
    plt.savefig(pareto_plot_path)
    plt.close()
    print(f"Pareto curve plot saved to {pareto_plot_path}")

    # 2. Publication Plot: Cost & Latency Measurement
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), dpi=300)

    # Subplot 1: FLOP reduction vs Target Sparsity
    flop_targets = [s * 100 for s in sparsities]
    taylor_flops = [summary["taylor_regrowth"][str(s)]["mean_flop_reduction"] * 100 for s in sparsities]
    ax1.plot(flop_targets, taylor_flops, marker="o", color="#2ca02c", lw=2.5, label="Achieved FLOP Reduction")
    ax1.plot(flop_targets, flop_targets, linestyle="--", color="gray", label="Ideal Theoretical Scaling")
    ax1.set_title("MAC FLOP Reduction vs. Sparsity", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Target Sparsity (%)", fontsize=11)
    ax1.set_ylabel("FLOP / MAC Reduction (%)", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=10)

    # Subplot 2: CSR Wall-Clock Latency & Speedup
    csr_times = [summary["taylor_regrowth"][str(s)]["mean_csr_latency_ms"] for s in sparsities]
    speedups = [summary["taylor_regrowth"][str(s)]["mean_speedup"] for s in sparsities]

    color = "#9467bd"
    ax2.set_xlabel("Target Sparsity (%)", fontsize=11)
    ax2.set_ylabel("CSR Inference Latency (ms)", color=color, fontsize=11)
    line1 = ax2.plot(flop_targets, csr_times, marker="s", color=color, lw=2.2, label="CSR Latency (ms)")
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.grid(True, linestyle=":", alpha=0.6)

    ax2_twin = ax2.twinx()
    color2 = "#ff7f0e"
    ax2_twin.set_ylabel("Relative Speedup vs Dense (x)", color=color2, fontsize=11)
    line2 = ax2_twin.plot(flop_targets, speedups, marker="^", color=color2, lw=2.2, linestyle="--", label="Speedup (x)")
    ax2_twin.tick_params(axis="y", labelcolor=color2)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc="center left", fontsize=10)
    ax2.set_title("Sparse Inference Efficiency & Latency", fontsize=12, fontweight="bold")

    plt.tight_layout()
    cost_plot_path = os.path.join(results_dir, "cost_measurement.png")
    plt.savefig(cost_plot_path)
    plt.close()
    print(f"Cost measurement plot saved to {cost_plot_path}")

    return full_results


if __name__ == "__main__":
    run_pareto_sweep()
