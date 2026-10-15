"""
Comprehensive unit tests for masked-weight semantics, gradient isolation,
Adam optimizer moment protection, RigL momentum reset, and CSR sparse inference.
"""

import numpy as np
import pytest
from engine.tensor import Tensor
from nn.layers import Linear, Parameter
from nn.optim import Adam
from nn.sparse_infer import CSRLinear


def test_masked_weights_contribute_zero_forward():
    """
    Test 1: Masked weights contribute strictly zero to forward pass
    even if underlying data before masking was non-zero.
    """
    np.random.seed(42)
    layer = Linear(in_features=4, out_features=3)

    # Set underlying data to large non-zero values
    layer.weight.data.fill(100.0)
    if layer.bias is not None:
        layer.bias.data.fill(0.0)

    # Mask half the weights
    mask = np.array([
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    layer.weight.mask = mask

    x = Tensor(np.ones((2, 4)))
    out = layer(x)

    # Change the masked entries in weight.data to completely different non-zero values
    layer.weight.data[mask == 0.0] = -9999.0
    out2 = layer(x)

    # Output must be bitwise identical since masked weights contribute strictly 0.0
    assert np.array_equal(out.data, out2.data)

    # Expected analytical output: x @ (weight.data * mask)
    expected = x.data @ (100.0 * mask)
    assert np.allclose(out.data, expected, atol=1e-12)


def test_masked_gradient_zero_and_unmasked_grad_finite_difference():
    """
    Test 2: Gradient of masked weight is strictly zero; unmasked_grad accurately matches
    finite-difference sensitivity of waking the weight up.
    """
    np.random.seed(42)
    layer = Linear(in_features=4, out_features=3)
    mask = np.ones((4, 3))
    # Mask out connection (1, 2)
    mask[1, 2] = 0.0
    layer.weight.mask = mask
    layer.weight.data *= mask

    x_val = np.random.randn(5, 4)
    x = Tensor(x_val, requires_grad=False)

    out = layer(x)
    loss = (out * out).sum()
    loss.backward()

    # 1. Effective gradient of masked weight must be strictly zero
    assert layer.weight.grad[1, 2] == 0.0

    # 2. Sensitivity of waking up weight (1, 2):
    # If connection (1, 2) is perturbed by eps in forward pass, how does loss change?
    eps = 1e-6
    # Perturb weight (1, 2) positively as an active connection
    w_perturbed_plus = layer.weight.data * mask
    w_perturbed_plus[1, 2] += eps
    loss_plus = np.sum((x_val @ w_perturbed_plus + (layer.bias.data if layer.bias else 0.0)) ** 2)

    w_perturbed_minus = layer.weight.data * mask
    w_perturbed_minus[1, 2] -= eps
    loss_minus = np.sum((x_val @ w_perturbed_minus + (layer.bias.data if layer.bias else 0.0)) ** 2)

    fd_sensitivity = (loss_plus - loss_minus) / (2.0 * eps)

    # unmasked_grad must match finite difference sensitivity
    assert np.isclose(layer.weight.unmasked_grad[1, 2], fd_sensitivity, rtol=1e-4, atol=1e-6)


def test_adam_moments_and_weights_isolated_over_steps():
    """
    Test 3: Over 20 Adam steps, masked weights remain strictly 0.0 and their
    m_t and v_t moments are strictly 0.0.
    """
    np.random.seed(42)
    layer = Linear(in_features=5, out_features=4)

    # Mask out weights
    mask = np.array([
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
    ])
    layer.weight.mask = mask
    layer.weight.data *= mask

    optim = Adam(layer.parameters(), lr=1e-2, betas=(0.9, 0.999), weight_decay=1e-3)

    for step in range(20):
        optim.zero_grad()
        x = Tensor(np.random.randn(8, 5))
        targets = np.random.randint(0, 4, size=(8,))
        out = layer(x)
        loss = out.softmax_cross_entropy(targets)
        loss.backward()

        optim.step()

        # Pruned weights must be strictly 0.0 floats
        assert np.all(layer.weight.data[mask == 0.0] == 0.0)
        # Moments m and v for pruned weights must be strictly 0.0
        state = optim.state[id(layer.weight)]
        assert np.all(state["m"][mask == 0.0] == 0.0)
        assert np.all(state["v"][mask == 0.0] == 0.0)
        assert np.all(layer.weight.grad[mask == 0.0] == 0.0)


def test_revived_connection_no_momentum_jump():
    """
    Test 4: When a pruned connection is revived (mask: 0 -> 1), it does not
    experience a momentum jump from pre-pruning history.
    """
    np.random.seed(42)
    layer = Linear(in_features=3, out_features=2)
    optim = Adam(layer.parameters(), lr=1e-2, betas=(0.9, 0.999))

    # Phase 1: Train for 5 steps with all weights active
    for _ in range(5):
        optim.zero_grad()
        x = Tensor(np.random.randn(4, 3))
        loss = layer(x).sum()
        loss.backward()
        optim.step()

    # Verify connection (1, 1) has non-zero momentum history
    state = optim.state[id(layer.weight)]
    assert state["m"][1, 1] != 0.0
    assert state["v"][1, 1] != 0.0

    # Phase 2: Prune connection (1, 1)
    layer.weight.mask[1, 1] = 0.0
    layer.weight.data[1, 1] = 0.0

    # Train for 5 more steps with (1, 1) masked
    for _ in range(5):
        optim.zero_grad()
        x = Tensor(np.random.randn(4, 3))
        loss = layer(x).sum()
        loss.backward()
        optim.step()

    # Verify moments are zeroed during masked period
    assert state["m"][1, 1] == 0.0
    assert state["v"][1, 1] == 0.0

    # Phase 3: Revive connection (1, 1)
    newly_active = np.zeros_like(layer.weight.mask)
    newly_active[1, 1] = 1.0
    layer.weight.mask[1, 1] = 1.0
    layer.weight.data[1, 1] = 0.0  # initialized to 0.0

    optim.reset_state_for_mask(layer.weight, newly_active)

    # Next step: should behave as step 1 for (1, 1) without stale momentum
    optim.zero_grad()
    x = Tensor(np.random.randn(4, 3))
    loss = layer(x).sum()
    loss.backward()

    observed_g = layer.weight.grad[1, 1]
    optim.step()

    # Verify m_1 = (1 - beta1) * g
    expected_m = (1.0 - optim.beta1) * observed_g
    assert np.isclose(state["m"][1, 1], expected_m, rtol=1e-5)


def test_csr_linear_mathematical_equivalence_1e12():
    """
    Test CSRLinear: Pure-NumPy CSR sparse matmul matches dense linear to 1e-12.
    """
    np.random.seed(42)
    dense_layer = Linear(in_features=32, out_features=16, dtype=np.float64)

    # Apply 85% sparsity mask
    mask = (np.random.rand(32, 16) > 0.85).astype(np.float64)
    dense_layer.weight.mask = mask
    dense_layer.weight.data *= mask

    csr_layer = CSRLinear.from_dense(dense_layer)

    # Multi-batch input
    x_np = np.random.randn(10, 32).astype(np.float64)
    x_tensor = Tensor(x_np, requires_grad=False)

    dense_out = dense_layer(x_tensor).data
    csr_out = csr_layer(x_np)

    max_diff = np.max(np.abs(dense_out - csr_out))
    assert max_diff < 1e-12, f"Max diff {max_diff} exceeded 1e-12"
    assert np.allclose(dense_out, csr_out, atol=1e-12, rtol=1e-12)


def test_pruner_cubic_schedule_and_regrowth():
    """
    Test Zhu & Gupta cubic schedule progression and RigL regrowth activation.
    """
    np.random.seed(42)
    from nn.layers import MLP
    from prune.pruner import Pruner
    from prune.criteria import TaylorSaliencyCriterion

    mlp = MLP(in_features=16, hidden_dims=[32, 16], out_features=4)
    optim = Adam(mlp.parameters(), lr=1e-2)
    criterion = TaylorSaliencyCriterion()

    pruner = Pruner(
        model=mlp,
        optimizer=optim,
        criterion=criterion,
        target_sparsity=0.80,
        initial_sparsity=0.0,
        t_0=10,
        t_end=50,
        delta_t=10,
        regrowth_ratio=0.20,
        output_sparsity_cap=0.50,
    )

    # Initial sparsity should be 0.0
    assert pruner.get_scheduled_sparsity(0) == 0.0
    assert pruner.get_scheduled_sparsity(10) == 0.0
    # Final sparsity should be target
    assert pruner.get_scheduled_sparsity(50) == 0.80
    assert pruner.get_scheduled_sparsity(100) == 0.80

    # Intermediate schedule is monotonically non-decreasing
    s20 = pruner.get_scheduled_sparsity(20)
    s30 = pruner.get_scheduled_sparsity(30)
    s40 = pruner.get_scheduled_sparsity(40)
    assert 0.0 < s20 < s30 < s40 < 0.80

    # Step through training with pruner
    for step in range(55):
        optim.zero_grad()
        x = Tensor(np.random.randn(8, 16))
        targets = np.random.randint(0, 4, size=(8,))
        loss = mlp(x).softmax_cross_entropy(targets)
        loss.backward()

        pruner.step(step)
        optim.step()

    # Verify that layers reached target sparsity (with output cap)
    assert mlp.layers[0].get_sparsity() >= 0.75
    assert mlp.layers[1].get_sparsity() >= 0.75
    assert mlp.layers[2].get_sparsity() <= 0.51

