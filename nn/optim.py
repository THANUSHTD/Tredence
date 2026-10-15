"""
Optimizers with mask-aware state isolation.

Features:
- Adam with explicit mask isolation preventing momentum drift on pruned weights.
- Momentum SGD with mask isolation.
- Momentum zero-reset for newly revived weights during RigL-style dynamic training.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from nn.layers import Parameter


class Optimizer:
    """Base class for all first-principles optimizers."""

    def __init__(self, params: List[Parameter], lr: float = 1e-3, weight_decay: float = 0.0):
        self.params: List[Parameter] = [p for p in params if p.requires_grad]
        self.lr: float = lr
        self.weight_decay: float = weight_decay
        self.state: Dict[int, dict] = {}

    def zero_grad(self) -> None:
        """Zeroes gradients and unmasked gradients for all tracked parameters."""
        for p in self.params:
            p.zero_grad()

    def step(self) -> None:
        raise NotImplementedError

    def reset_state_for_mask(self, param: Parameter, newly_active_mask: np.ndarray) -> None:
        """Resets optimizer historical moments to 0 for newly active connections."""
        raise NotImplementedError


class Adam(Optimizer):
    """
    Mask-Aware Adam optimizer.
    Guarantees that pruned weights (mask == 0) receive zero updates and zero momentum
    accumulation, preventing silent corruption of pruned weights and momentum spikes
    upon regrowth.
    """

    def __init__(
        self,
        params: List[Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        super().__init__(params, lr=lr, weight_decay=weight_decay)
        self.beta1: float = betas[0]
        self.beta2: float = betas[1]
        self.eps: float = eps

    def step(self) -> None:
        for p in self.params:
            if p.grad is None:
                continue

            param_id = id(p)
            if param_id not in self.state:
                self.state[param_id] = {
                    "step": 0,
                    "m": np.zeros_like(p.data),
                    "v": np.zeros_like(p.data),
                }

            state = self.state[param_id]
            state["step"] += 1
            step = state["step"]
            m, v = state["m"], state["v"]

            grad = p.grad
            # 1. Enforce gradient isolation for masked weights
            if p.mask is not None:
                grad = grad * p.mask

            # 2. Decoupled weight decay (AdamW) applied strictly to active weights
            if self.weight_decay != 0.0:
                if p.mask is not None:
                    p.data -= self.lr * self.weight_decay * (p.data * p.mask)
                else:
                    p.data -= self.lr * self.weight_decay * p.data

            # 3. Moment updates
            m[:] = self.beta1 * m + (1.0 - self.beta1) * grad
            v[:] = self.beta2 * v + (1.0 - self.beta2) * (grad ** 2)

            # 4. Explicitly zero moments for pruned weights
            if p.mask is not None:
                m *= p.mask
                v *= p.mask

            # 5. Bias correction
            bias_corr1 = 1.0 - (self.beta1 ** step)
            bias_corr2 = 1.0 - (self.beta2 ** step)

            m_hat = m / bias_corr1
            v_hat = v / bias_corr2

            # 6. Parameter update with strict zero enforcement
            update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            if p.mask is not None:
                update *= p.mask
                p.data -= update
                p.data *= p.mask  # Enforce hardware zeros
            else:
                p.data -= update

    def reset_state_for_mask(self, param: Parameter, newly_active_mask: np.ndarray) -> None:
        param_id = id(param)
        if param_id in self.state:
            active_bool = newly_active_mask.astype(bool)
            self.state[param_id]["m"][active_bool] = 0.0
            self.state[param_id]["v"][active_bool] = 0.0


class SGD(Optimizer):
    """
    Mask-Aware Stochastic Gradient Descent with Momentum.
    """

    def __init__(
        self,
        params: List[Parameter],
        lr: float = 1e-2,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
    ):
        super().__init__(params, lr=lr, weight_decay=weight_decay)
        self.momentum: float = momentum

    def step(self) -> None:
        for p in self.params:
            if p.grad is None:
                continue

            param_id = id(p)
            if param_id not in self.state:
                self.state[param_id] = {
                    "v": np.zeros_like(p.data),
                }

            v = self.state[param_id]["v"]
            grad = p.grad
            if p.mask is not None:
                grad = grad * p.mask

            if self.weight_decay != 0.0:
                if p.mask is not None:
                    p.data -= self.lr * self.weight_decay * (p.data * p.mask)
                else:
                    p.data -= self.lr * self.weight_decay * p.data

            v[:] = self.momentum * v + grad
            if p.mask is not None:
                v *= p.mask

            update = self.lr * v
            if p.mask is not None:
                update *= p.mask
                p.data -= update
                p.data *= p.mask
            else:
                p.data -= update

    def reset_state_for_mask(self, param: Parameter, newly_active_mask: np.ndarray) -> None:
        param_id = id(param)
        if param_id in self.state:
            self.state[param_id]["v"][newly_active_mask.astype(bool)] = 0.0
