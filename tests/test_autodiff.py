"""
Finite-difference numerical gradient checks for reverse-mode autodiff engine.
Validates all operations, broadcasting reductions, reductions, activations,
and fused softmax cross-entropy against central finite differences:
[f(x + eps) - f(x - eps)] / (2 * eps).
"""

import numpy as np
import pytest
from engine.tensor import Tensor


def compute_numerical_gradient(f, x_arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Computes numerical gradient of a scalar-valued function f w.r.t input array x_arr.
    f is a callable: f(Tensor) -> Tensor (scalar).
    """
    grad = np.zeros_like(x_arr, dtype=np.float64)
    it = np.nditer(x_arr, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        orig = x_arr[idx]

        x_arr[idx] = orig + eps
        t_plus = Tensor(x_arr, requires_grad=False, dtype=np.float64)
        out_plus = f(t_plus)
        val_plus = out_plus.item()

        x_arr[idx] = orig - eps
        t_minus = Tensor(x_arr, requires_grad=False, dtype=np.float64)
        out_minus = f(t_minus)
        val_minus = out_minus.item()

        grad[idx] = (val_plus - val_minus) / (2.0 * eps)
        x_arr[idx] = orig
        it.iternext()

    return grad


def test_gradcheck_add():
    np.random.seed(42)
    a_init = np.random.randn(3, 4)
    b_init = np.random.randn(3, 4)

    # Autodiff
    a = Tensor(a_init, requires_grad=True)
    b = Tensor(b_init, requires_grad=True)
    out = (a + b).sum()
    out.backward()

    # Numerical
    num_a = compute_numerical_gradient(lambda t: (t + Tensor(b_init, requires_grad=False)).sum(), a_init.copy())
    num_b = compute_numerical_gradient(lambda t: (Tensor(a_init, requires_grad=False) + t).sum(), b_init.copy())

    assert np.allclose(a.grad, num_a, rtol=1e-4, atol=1e-6)
    assert np.allclose(b.grad, num_b, rtol=1e-4, atol=1e-6)


def test_gradcheck_sub():
    np.random.seed(42)
    a_init = np.random.randn(3, 4)
    b_init = np.random.randn(3, 4)

    a = Tensor(a_init, requires_grad=True)
    b = Tensor(b_init, requires_grad=True)
    out = (a - b).sum()
    out.backward()

    num_a = compute_numerical_gradient(lambda t: (t - Tensor(b_init, requires_grad=False)).sum(), a_init.copy())
    num_b = compute_numerical_gradient(lambda t: (Tensor(a_init, requires_grad=False) - t).sum(), b_init.copy())

    assert np.allclose(a.grad, num_a, rtol=1e-4, atol=1e-6)
    assert np.allclose(b.grad, num_b, rtol=1e-4, atol=1e-6)


def test_gradcheck_mul():
    np.random.seed(42)
    a_init = np.random.randn(3, 4)
    b_init = np.random.randn(3, 4)

    a = Tensor(a_init, requires_grad=True)
    b = Tensor(b_init, requires_grad=True)
    out = (a * b).sum()
    out.backward()

    num_a = compute_numerical_gradient(lambda t: (t * Tensor(b_init, requires_grad=False)).sum(), a_init.copy())
    num_b = compute_numerical_gradient(lambda t: (Tensor(a_init, requires_grad=False) * t).sum(), b_init.copy())

    assert np.allclose(a.grad, num_a, rtol=1e-4, atol=1e-6)
    assert np.allclose(b.grad, num_b, rtol=1e-4, atol=1e-6)


def test_gradcheck_div():
    np.random.seed(42)
    a_init = np.random.randn(3, 4)
    b_init = np.random.uniform(0.5, 2.0, size=(3, 4))  # Avoid divide by zero

    a = Tensor(a_init, requires_grad=True)
    b = Tensor(b_init, requires_grad=True)
    out = (a / b).sum()
    out.backward()

    num_a = compute_numerical_gradient(lambda t: (t / Tensor(b_init, requires_grad=False)).sum(), a_init.copy())
    num_b = compute_numerical_gradient(lambda t: (Tensor(a_init, requires_grad=False) / t).sum(), b_init.copy())

    assert np.allclose(a.grad, num_a, rtol=1e-4, atol=1e-6)
    assert np.allclose(b.grad, num_b, rtol=1e-4, atol=1e-6)


def test_gradcheck_matmul():
    np.random.seed(42)
    x_init = np.random.randn(4, 5)
    w_init = np.random.randn(5, 3)

    x = Tensor(x_init, requires_grad=True)
    w = Tensor(w_init, requires_grad=True)
    out = (x @ w).sum()
    out.backward()

    num_x = compute_numerical_gradient(lambda t: (t @ Tensor(w_init, requires_grad=False)).sum(), x_init.copy())
    num_w = compute_numerical_gradient(lambda t: (Tensor(x_init, requires_grad=False) @ t).sum(), w_init.copy())

    assert np.allclose(x.grad, num_x, rtol=1e-4, atol=1e-6)
    assert np.allclose(w.grad, num_w, rtol=1e-4, atol=1e-6)


def test_gradcheck_sum_and_mean():
    np.random.seed(42)
    x_init = np.random.randn(3, 4, 5)

    # Sum along axis
    x = Tensor(x_init, requires_grad=True)
    out = x.sum(axis=1, keepdims=False).sum()
    out.backward()
    num_x = compute_numerical_gradient(lambda t: t.sum(axis=1, keepdims=False).sum(), x_init.copy())
    assert np.allclose(x.grad, num_x, rtol=1e-4, atol=1e-6)

    # Mean along axis with keepdims
    x_m = Tensor(x_init, requires_grad=True)
    out_m = x_m.mean(axis=(0, 2), keepdims=True).sum()
    out_m.backward()
    num_xm = compute_numerical_gradient(lambda t: t.mean(axis=(0, 2), keepdims=True).sum(), x_init.copy())
    assert np.allclose(x_m.grad, num_xm, rtol=1e-4, atol=1e-6)


def test_gradcheck_relu():
    np.random.seed(42)
    x_init = np.random.randn(4, 4)
    # Ensure values are not too close to zero to avoid kink
    x_init[np.abs(x_init) < 0.05] = 0.1

    x = Tensor(x_init, requires_grad=True)
    out = x.relu().sum()
    out.backward()

    num_x = compute_numerical_gradient(lambda t: t.relu().sum(), x_init.copy())
    assert np.allclose(x.grad, num_x, rtol=1e-4, atol=1e-6)


def test_gradcheck_tanh():
    np.random.seed(42)
    x_init = np.random.randn(3, 4)

    x = Tensor(x_init, requires_grad=True)
    out = x.tanh().sum()
    out.backward()

    num_x = compute_numerical_gradient(lambda t: t.tanh().sum(), x_init.copy())
    assert np.allclose(x.grad, num_x, rtol=1e-4, atol=1e-6)


def test_gradcheck_sigmoid():
    np.random.seed(42)
    x_init = np.random.randn(3, 4)

    x = Tensor(x_init, requires_grad=True)
    out = x.sigmoid().sum()
    out.backward()

    num_x = compute_numerical_gradient(lambda t: t.sigmoid().sum(), x_init.copy())
    assert np.allclose(x.grad, num_x, rtol=1e-4, atol=1e-6)


def test_gradcheck_gelu():
    np.random.seed(42)
    x_init = np.random.randn(4, 5)

    x = Tensor(x_init, requires_grad=True)
    out = x.gelu().sum()
    out.backward()

    num_x = compute_numerical_gradient(lambda t: t.gelu().sum(), x_init.copy())
    assert np.allclose(x.grad, num_x, rtol=1e-4, atol=1e-6)


def test_gradcheck_softmax_cross_entropy():
    np.random.seed(42)
    logits_init = np.random.randn(5, 4)
    targets_indices = np.array([0, 2, 1, 3, 1], dtype=np.int64)

    # 1D targets
    logits = Tensor(logits_init, requires_grad=True)
    loss = logits.softmax_cross_entropy(targets_indices)
    loss.backward()

    num_logits = compute_numerical_gradient(lambda t: t.softmax_cross_entropy(targets_indices), logits_init.copy())
    assert np.allclose(logits.grad, num_logits, rtol=1e-4, atol=1e-6)

    # 2D one-hot targets
    targets_one_hot = np.zeros_like(logits_init)
    targets_one_hot[np.arange(5), targets_indices] = 1.0

    logits_oh = Tensor(logits_init, requires_grad=True)
    loss_oh = logits_oh.softmax_cross_entropy(targets_one_hot)
    loss_oh.backward()

    num_logits_oh = compute_numerical_gradient(lambda t: t.softmax_cross_entropy(targets_one_hot), logits_init.copy())
    assert np.allclose(logits_oh.grad, num_logits_oh, rtol=1e-4, atol=1e-6)


def test_gradcheck_broadcasting():
    np.random.seed(42)
    # Case 1: (N, D) + (D,)
    x_init = np.random.randn(6, 4)
    b_init = np.random.randn(4)

    x = Tensor(x_init, requires_grad=True)
    b = Tensor(b_init, requires_grad=True)
    out = (x + b).sum()
    out.backward()

    assert x.grad.shape == x_init.shape
    assert b.grad.shape == b_init.shape

    num_x = compute_numerical_gradient(lambda t: (t + Tensor(b_init, requires_grad=False)).sum(), x_init.copy())
    num_b = compute_numerical_gradient(lambda t: (Tensor(x_init, requires_grad=False) + t).sum(), b_init.copy())

    assert np.allclose(x.grad, num_x, rtol=1e-4, atol=1e-6)
    assert np.allclose(b.grad, num_b, rtol=1e-4, atol=1e-6)

    # Case 2: (N, 1) * (1, D) -> (N, D)
    u_init = np.random.randn(5, 1)
    v_init = np.random.randn(1, 4)

    u = Tensor(u_init, requires_grad=True)
    v = Tensor(v_init, requires_grad=True)
    out2 = (u * v).sum()
    out2.backward()

    assert u.grad.shape == u_init.shape
    assert v.grad.shape == v_init.shape

    num_u = compute_numerical_gradient(lambda t: (t * Tensor(v_init, requires_grad=False)).sum(), u_init.copy())
    num_v = compute_numerical_gradient(lambda t: (Tensor(u_init, requires_grad=False) * t).sum(), v_init.copy())

    assert np.allclose(u.grad, num_u, rtol=1e-4, atol=1e-6)
    assert np.allclose(v.grad, num_v, rtol=1e-4, atol=1e-6)
