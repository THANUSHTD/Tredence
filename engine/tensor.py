"""
Pure Python & NumPy reverse-mode automatic differentiation Tensor engine.

Features:
- N-dimensional reverse-mode autograd with iterative/recursive topological sort.
- Robust multidimensional broadcasting reduction (_unbroadcast).
- Elementwise operations (+, -, *, /, neg).
- 2D Matrix multiplication (@).
- Reduction operations (sum, mean).
- Non-linearities (relu, tanh, sigmoid, gelu) with exact analytical gradients.
- Numerically stable softmax cross-entropy loss with log-sum-exp stabilization.
- Explicit binary mask support for dynamic sparse training (W_eff = W * mask)
  capturing raw unmasked_grad for regrowth without optimizer corruption.
"""

from __future__ import annotations
import math
from typing import Optional, Set, Tuple, Union
import numpy as np


def _unbroadcast(grad: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """
    Reduces `grad` to match `target_shape` by summing across broadcasted dimensions:
    1. Leading prepended dimensions (when grad.ndim > len(target_shape)).
    2. Dimensions where target_shape has size 1 but grad has size > 1.
    """
    if grad.shape == target_shape:
        return grad

    # Scalar target
    if len(target_shape) == 0:
        return np.array(grad.sum(), dtype=grad.dtype)

    # 1. Sum over leading prepended dimensions
    extra_dims = grad.ndim - len(target_shape)
    if extra_dims > 0:
        grad = grad.sum(axis=tuple(range(extra_dims)))

    # 2. Sum over dimensions where target_shape is 1
    axes_to_sum = []
    for i, dim in enumerate(target_shape):
        if dim == 1 and grad.shape[i] > 1:
            axes_to_sum.append(i)

    if axes_to_sum:
        grad = grad.sum(axis=tuple(axes_to_sum), keepdims=True)

    if grad.shape != target_shape:
        grad = grad.reshape(target_shape)

    return grad


class Tensor:
    """
    Reverse-mode autodiff Tensor backed by NumPy arrays.
    """

    def __init__(
        self,
        data: Union[int, float, list, np.ndarray, "Tensor"],
        _prev: Tuple["Tensor", ...] = (),
        _op: str = "",
        requires_grad: bool = True,
        mask: Optional[np.ndarray] = None,
        dtype: Optional[np.dtype] = None,
    ):
        if isinstance(data, Tensor):
            self.data = np.array(data.data, dtype=dtype if dtype is not None else data.data.dtype, copy=True)
        elif isinstance(data, np.ndarray):
            self.data = data.astype(dtype) if dtype is not None and data.dtype != dtype else data
        else:
            self.data = np.asarray(data, dtype=dtype if dtype is not None else np.float64)

        self._prev: Set[Tensor] = set(_prev)
        self._op: str = _op
        self.requires_grad: bool = requires_grad
        self._backward = lambda: None

        self.grad: Optional[np.ndarray] = None
        self.unmasked_grad: Optional[np.ndarray] = None

        if mask is not None:
            self.mask: Optional[np.ndarray] = np.asarray(mask, dtype=self.data.dtype)
            if self.mask.shape != self.data.shape:
                raise ValueError(f"Mask shape {self.mask.shape} does not match data shape {self.data.shape}")
        else:
            self.mask: Optional[np.ndarray] = None

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def T(self) -> "Tensor":
        if self.ndim != 2:
            raise ValueError("Transpose .T property is currently implemented for 2D tensors")
        out = Tensor(self.data.T, _prev=(self,), _op="T", requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                g = out.grad.T
                if self.grad is None:
                    self.grad = np.copy(g)
                else:
                    self.grad += g

        out._backward = _backward
        return out

    def effective(self) -> "Tensor":
        """
        Returns the effective forward tensor.
        When a binary mask is present:
            Forward: W_eff = W.data * W.mask
            Backward:
                out.grad is dL / d(W_eff).
                W.unmasked_grad accumulates out.grad (sensitivity for regrowth).
                W.grad accumulates out.grad * W.mask (isolated from optimizer corruption).
        """
        if self.mask is None:
            return self

        eff_data = self.data * self.mask
        out = Tensor(eff_data, _prev=(self,), _op="mask", requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                raw = out.grad
                if self.unmasked_grad is None:
                    self.unmasked_grad = np.copy(raw)
                else:
                    self.unmasked_grad += raw

                masked = raw * self.mask
                if self.grad is None:
                    self.grad = np.copy(masked)
                else:
                    self.grad += masked

        out._backward = _backward
        return out

    def zero_grad(self) -> None:
        """Resets gradients to zero."""
        if self.grad is not None:
            self.grad.fill(0.0)
        if self.unmasked_grad is not None:
            self.unmasked_grad.fill(0.0)

    def numpy(self) -> np.ndarray:
        return self.data

    def detach(self) -> "Tensor":
        return Tensor(self.data.copy(), requires_grad=False)

    def item(self) -> float:
        return float(self.data.item())

    # -------------------------------------------------------------------------
    # Core Arithmetic Operations
    # -------------------------------------------------------------------------

    def __add__(self, other: Union["Tensor", float, int, np.ndarray]) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(
            self.data + other.data,
            _prev=(self, other),
            _op="+",
            requires_grad=self.requires_grad or other.requires_grad,
        )

        def _backward():
            if self.requires_grad:
                g_self = _unbroadcast(out.grad, self.shape)
                if self.grad is None:
                    self.grad = np.copy(g_self)
                else:
                    self.grad += g_self
            if other.requires_grad:
                g_other = _unbroadcast(out.grad, other.shape)
                if other.grad is None:
                    other.grad = np.copy(g_other)
                else:
                    other.grad += g_other

        out._backward = _backward
        return out

    def __radd__(self, other: Union[float, int, np.ndarray]) -> "Tensor":
        return self.__add__(other)

    def __neg__(self) -> "Tensor":
        out = Tensor(-self.data, _prev=(self,), _op="neg", requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = -out.grad
                else:
                    self.grad -= out.grad

        out._backward = _backward
        return out

    def __sub__(self, other: Union["Tensor", float, int, np.ndarray]) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(
            self.data - other.data,
            _prev=(self, other),
            _op="-",
            requires_grad=self.requires_grad or other.requires_grad,
        )

        def _backward():
            if self.requires_grad:
                g_self = _unbroadcast(out.grad, self.shape)
                if self.grad is None:
                    self.grad = np.copy(g_self)
                else:
                    self.grad += g_self
            if other.requires_grad:
                g_other = _unbroadcast(-out.grad, other.shape)
                if other.grad is None:
                    other.grad = np.copy(g_other)
                else:
                    other.grad += g_other

        out._backward = _backward
        return out

    def __rsub__(self, other: Union[float, int, np.ndarray]) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        return other.__sub__(self)

    def __mul__(self, other: Union["Tensor", float, int, np.ndarray]) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(
            self.data * other.data,
            _prev=(self, other),
            _op="*",
            requires_grad=self.requires_grad or other.requires_grad,
        )

        def _backward():
            if self.requires_grad:
                g_self = _unbroadcast(out.grad * other.data, self.shape)
                if self.grad is None:
                    self.grad = np.copy(g_self)
                else:
                    self.grad += g_self
            if other.requires_grad:
                g_other = _unbroadcast(out.grad * self.data, other.shape)
                if other.grad is None:
                    other.grad = np.copy(g_other)
                else:
                    other.grad += g_other

        out._backward = _backward
        return out

    def __rmul__(self, other: Union[float, int, np.ndarray]) -> "Tensor":
        return self.__mul__(other)

    def __truediv__(self, other: Union["Tensor", float, int, np.ndarray]) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(
            self.data / other.data,
            _prev=(self, other),
            _op="/",
            requires_grad=self.requires_grad or other.requires_grad,
        )

        def _backward():
            if self.requires_grad:
                g_self = _unbroadcast(out.grad / other.data, self.shape)
                if self.grad is None:
                    self.grad = np.copy(g_self)
                else:
                    self.grad += g_self
            if other.requires_grad:
                g_other = _unbroadcast(-out.grad * self.data / (other.data ** 2), other.shape)
                if other.grad is None:
                    other.grad = np.copy(g_other)
                else:
                    other.grad += g_other

        out._backward = _backward
        return out

    def __rtruediv__(self, other: Union[float, int, np.ndarray]) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        return other.__truediv__(self)

    def __matmul__(self, other: "Tensor") -> "Tensor":
        if not isinstance(other, Tensor):
            raise TypeError(f"Matmul unsupported operand type: Tensor @ {type(other)}")
        if self.ndim != 2 or other.ndim != 2:
            raise ValueError(f"2D matmul expected, got shapes {self.shape} and {other.shape}")

        out = Tensor(
            self.data @ other.data,
            _prev=(self, other),
            _op="@",
            requires_grad=self.requires_grad or other.requires_grad,
        )

        def _backward():
            if self.requires_grad:
                g_self = out.grad @ other.data.T
                if self.grad is None:
                    self.grad = np.copy(g_self)
                else:
                    self.grad += g_self
            if other.requires_grad:
                g_other = self.data.T @ out.grad
                if other.grad is None:
                    other.grad = np.copy(g_other)
                else:
                    other.grad += g_other

        out._backward = _backward
        return out

    # -------------------------------------------------------------------------
    # Reductions
    # -------------------------------------------------------------------------

    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> "Tensor":
        out_data = np.sum(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(out_data, _prev=(self,), _op="sum", requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                grad = out.grad
                if axis is None:
                    g = np.full(self.shape, grad, dtype=self.dtype)
                else:
                    if not keepdims:
                        axes = (axis,) if isinstance(axis, int) else tuple(axis)
                        axes = tuple(a if a >= 0 else a + self.ndim for a in axes)
                        g = grad
                        for a in sorted(axes):
                            g = np.expand_dims(g, axis=a)
                    else:
                        g = grad
                    g = np.broadcast_to(g, self.shape)

                if self.grad is None:
                    self.grad = np.copy(g)
                else:
                    self.grad += g

        out._backward = _backward
        return out

    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> "Tensor":
        out_data = np.mean(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(out_data, _prev=(self,), _op="mean", requires_grad=self.requires_grad)

        if axis is None:
            num_elem = self.data.size
        else:
            axes = (axis,) if isinstance(axis, int) else tuple(axis)
            axes = tuple(a if a >= 0 else a + self.ndim for a in axes)
            num_elem = 1
            for a in axes:
                num_elem *= self.shape[a]

        def _backward():
            if self.requires_grad:
                grad = out.grad / num_elem
                if axis is None:
                    g = np.full(self.shape, grad, dtype=self.dtype)
                else:
                    if not keepdims:
                        axes = (axis,) if isinstance(axis, int) else tuple(axis)
                        axes = tuple(a if a >= 0 else a + self.ndim for a in axes)
                        g = grad
                        for a in sorted(axes):
                            g = np.expand_dims(g, axis=a)
                    else:
                        g = grad
                    g = np.broadcast_to(g, self.shape)

                if self.grad is None:
                    self.grad = np.copy(g)
                else:
                    self.grad += g

        out._backward = _backward
        return out

    # -------------------------------------------------------------------------
    # Non-linearities
    # -------------------------------------------------------------------------

    def relu(self) -> "Tensor":
        out_data = np.maximum(0.0, self.data)
        out = Tensor(out_data, _prev=(self,), _op="relu", requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                g = out.grad * (self.data > 0.0)
                if self.grad is None:
                    self.grad = np.copy(g)
                else:
                    self.grad += g

        out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        out_data = np.tanh(self.data)
        out = Tensor(out_data, _prev=(self,), _op="tanh", requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                g = out.grad * (1.0 - out_data ** 2)
                if self.grad is None:
                    self.grad = np.copy(g)
                else:
                    self.grad += g

        out._backward = _backward
        return out

    def sigmoid(self) -> "Tensor":
        # Numerically stable sigmoid formulation
        pos_mask = self.data >= 0
        neg_mask = ~pos_mask
        out_data = np.empty_like(self.data, dtype=np.float64)
        out_data[pos_mask] = 1.0 / (1.0 + np.exp(-self.data[pos_mask]))
        exp_neg = np.exp(self.data[neg_mask])
        out_data[neg_mask] = exp_neg / (1.0 + exp_neg)

        out = Tensor(out_data, _prev=(self,), _op="sigmoid", requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                g = out.grad * out_data * (1.0 - out_data)
                if self.grad is None:
                    self.grad = np.copy(g)
                else:
                    self.grad += g

        out._backward = _backward
        return out

    def gelu(self) -> "Tensor":
        """
        Exact tanh approximation of GELU:
        GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        With matching exact analytical derivative.
        """
        c = math.sqrt(2.0 / math.pi)
        x = self.data
        u = c * (x + 0.044715 * (x ** 3))
        tanh_u = np.tanh(u)
        out_data = 0.5 * x * (1.0 + tanh_u)

        out = Tensor(out_data, _prev=(self,), _op="gelu", requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                du_dx = c * (1.0 + 3.0 * 0.044715 * (x ** 2))
                dtanh_dx = (1.0 - tanh_u ** 2) * du_dx
                dgelu_dx = 0.5 * (1.0 + tanh_u) + 0.5 * x * dtanh_dx
                g = out.grad * dgelu_dx
                if self.grad is None:
                    self.grad = np.copy(g)
                else:
                    self.grad += g

        out._backward = _backward
        return out

    # -------------------------------------------------------------------------
    # Loss: Softmax Cross Entropy
    # -------------------------------------------------------------------------

    def softmax_cross_entropy(self, targets: Union[np.ndarray, "Tensor"]) -> "Tensor":
        """
        Fused differentiable softmax cross-entropy loss with log-sum-exp stabilization.
        `self` is logits of shape (N, C).
        `targets` can be:
            - 1D integer class indices of shape (N,)
            - 2D one-hot probabilities of shape (N, C)
        Returns scalar loss Tensor.
        """
        logits = self.data
        if logits.ndim != 2:
            raise ValueError(f"Logits must be 2D of shape (N, C), got {logits.shape}")

        N, C = logits.shape
        targets_arr = targets.data if isinstance(targets, Tensor) else np.asarray(targets)

        # Log-sum-exp trick
        max_logits = np.max(logits, axis=-1, keepdims=True)
        shifted_logits = logits - max_logits
        exp_logits = np.exp(shifted_logits)
        sum_exp = np.sum(exp_logits, axis=-1, keepdims=True)
        probs = exp_logits / sum_exp

        # Cross entropy computation
        eps = 1e-15
        if targets_arr.ndim == 1:
            # Integer indices
            target_indices = targets_arr.astype(np.int64)
            log_probs = shifted_logits - np.log(sum_exp)
            nll = -log_probs[np.arange(N), target_indices]
            loss_val = np.mean(nll)

            out = Tensor(loss_val, _prev=(self,), _op="softmax_ce", requires_grad=self.requires_grad)

            def _backward():
                if self.requires_grad:
                    grad_logits = np.copy(probs)
                    grad_logits[np.arange(N), target_indices] -= 1.0
                    grad_logits = (grad_logits / N) * out.grad
                    if self.grad is None:
                        self.grad = np.copy(grad_logits)
                    else:
                        self.grad += grad_logits

            out._backward = _backward
            return out

        elif targets_arr.ndim == 2:
            # One-hot or soft labels
            log_probs = shifted_logits - np.log(sum_exp)
            loss_val = -np.sum(targets_arr * log_probs) / N

            out = Tensor(loss_val, _prev=(self,), _op="softmax_ce", requires_grad=self.requires_grad)

            def _backward():
                if self.requires_grad:
                    grad_logits = (probs - targets_arr) / N * out.grad
                    if self.grad is None:
                        self.grad = np.copy(grad_logits)
                    else:
                        self.grad += grad_logits

            out._backward = _backward
            return out
        else:
            raise ValueError(f"Targets must be 1D or 2D, got shape {targets_arr.shape}")

    # -------------------------------------------------------------------------
    # Backward Pass & Topological Sort
    # -------------------------------------------------------------------------

    def backward(self, grad: Optional[np.ndarray] = None) -> None:
        """
        Executes reverse-mode autodiff topological traversal.
        """
        if not self.requires_grad:
            return

        if grad is None:
            if self.data.ndim == 0 or self.data.size == 1:
                grad = np.array(1.0, dtype=self.data.dtype)
            else:
                raise RuntimeError("Grad can only be implicitly created for scalar outputs")

        if self.grad is None:
            self.grad = np.copy(grad)
        else:
            self.grad += grad

        # Post-order DFS topological sort
        topo: list[Tensor] = []
        visited: Set[Tensor] = set()

        def build_topo(v: Tensor):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        for node in reversed(topo):
            node._backward()

    def __repr__(self) -> str:
        mask_info = f", mask_sparsity={1.0 - (self.mask.sum() / self.mask.size):.2%}" if self.mask is not None else ""
        return f"Tensor(shape={self.shape}, dtype={self.dtype}, op='{self._op}'{mask_info})"
