"""
Pruning importance criteria: Magnitude vs. First-Order Taylor Saliency.
"""

from typing import Dict, List, Optional
import numpy as np
from nn.layers import Linear, Module, Parameter


class PruningCriterion:
    """Base class for weight importance scoring criteria."""

    def update(self, model: Module) -> None:
        """Hook called at every training step to update running statistics."""
        pass

    def compute_scores(self, layer: Linear) -> np.ndarray:
        """Computes importance scores S_ij for weights of a given layer."""
        raise NotImplementedError


class MagnitudeCriterion(PruningCriterion):
    """
    Standard Magnitude-based pruning criterion: S_ij = |W_ij|.
    Weights with the smallest absolute values are deemed least important.
    """

    def compute_scores(self, layer: Linear) -> np.ndarray:
        w = layer.weight.data
        if layer.weight.mask is not None:
            w = w * layer.weight.mask
        return np.abs(w)


class TaylorSaliencyCriterion(PruningCriterion):
    """
    First-Order Taylor Expansion Saliency Criterion with EMA smoothing:
    Delta L_i approx |(dL / dW_i) * W_i|.

    Maintains an exponential moving average (EMA, beta=0.9) of |W * unmasked_grad|
    across mini-batches to suppress gradient noise, combined with a small magnitude floor
    lambda_floor * |W| to prevent deadlocks at initialization and low-gradient plateaus.
    """

    def __init__(self, beta: float = 0.9, lambda_floor: float = 1e-4):
        self.beta: float = beta
        self.lambda_floor: float = lambda_floor
        self.ema_saliency: Dict[int, np.ndarray] = {}

    def update(self, model: Module) -> None:
        """
        Updates EMA saliency using unmasked_grad (captures full parameter sensitivity).
        """
        for mod in model.modules():
            if isinstance(mod, Linear):
                p = mod.weight
                grad = p.unmasked_grad if p.unmasked_grad is not None else p.grad
                if grad is None:
                    continue

                param_id = id(p)
                w_eff = p.data * (p.mask if p.mask is not None else 1.0)
                inst_saliency = np.abs(w_eff * grad)

                if param_id not in self.ema_saliency:
                    self.ema_saliency[param_id] = np.copy(inst_saliency)
                else:
                    self.ema_saliency[param_id] = (
                        self.beta * self.ema_saliency[param_id] + (1.0 - self.beta) * inst_saliency
                    )

    def compute_scores(self, layer: Linear) -> np.ndarray:
        p = layer.weight
        param_id = id(p)
        w_eff = np.abs(p.data * (p.mask if p.mask is not None else 1.0))

        if param_id in self.ema_saliency:
            saliency = self.ema_saliency[param_id]
        else:
            grad = p.unmasked_grad if p.unmasked_grad is not None else p.grad
            saliency = np.abs(w_eff * grad) if grad is not None else np.zeros_like(p.data)

        # Combine EMA Taylor saliency with magnitude floor
        return saliency + self.lambda_floor * w_eff
