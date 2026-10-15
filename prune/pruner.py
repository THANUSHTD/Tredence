"""
Dynamic sparse pruning engine with Zhu & Gupta cubic sparsity schedule
and RigL-style gradient-based regrowth.
"""

import math
from typing import List, Optional
import numpy as np
from nn.layers import Linear, Module
from nn.optim import Optimizer
from prune.criteria import MagnitudeCriterion, PruningCriterion, TaylorSaliencyCriterion


class Pruner:
    """
    Sparsity scheduler and mask update engine.
    Supports:
    - Zhu & Gupta (2017) cubic sparsity scheduling.
    - Taylor saliency and magnitude importance criteria.
    - RigL-style dynamic prune-and-regrow with gradient sensitivity.
    - Output layer protection caps to preserve classification heads.
    """

    def __init__(
        self,
        model: Module,
        optimizer: Optional[Optimizer] = None,
        criterion: Optional[PruningCriterion] = None,
        target_sparsity: float = 0.90,
        initial_sparsity: float = 0.0,
        t_0: int = 100,
        t_end: int = 2000,
        delta_t: int = 50,
        regrowth_ratio: float = 0.20,
        output_sparsity_cap: float = 0.50,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion if criterion is not None else TaylorSaliencyCriterion()
        self.target_sparsity = target_sparsity
        self.initial_sparsity = initial_sparsity
        self.t_0 = t_0
        self.t_end = t_end
        self.delta_t = delta_t
        self.regrowth_ratio = regrowth_ratio
        self.output_sparsity_cap = output_sparsity_cap

        # Discover linear layers
        self.linear_layers: List[Linear] = []
        for mod in model.modules():
            if isinstance(mod, Linear):
                self.linear_layers.append(mod)

    def get_scheduled_sparsity(self, step: int) -> float:
        """
        Calculates sparsity at step t via Zhu & Gupta cubic schedule:
        s_t = s_f + (s_i - s_f) * (1 - (t - t_0)/(t_end - t_0))^3
        """
        if step < self.t_0:
            return self.initial_sparsity
        if step >= self.t_end:
            return self.target_sparsity

        progress = (step - self.t_0) / float(self.t_end - self.t_0)
        s_t = self.target_sparsity + (self.initial_sparsity - self.target_sparsity) * ((1.0 - progress) ** 3)
        return float(s_t)

    def is_pruning_step(self, step: int) -> bool:
        """Returns True if masks should be recalculated at this step."""
        if step < self.t_0 or step > self.t_end:
            return False
        return (step - self.t_0) % self.delta_t == 0

    def step(self, current_step: int) -> Optional[float]:
        """
        Performs scheduled mask updates, importance pruning, and gradient-based regrowth.
        Returns the new target sparsity if a pruning step occurred, else None.
        """
        # Always update criterion stats (e.g. EMA)
        self.criterion.update(self.model)

        if not self.is_pruning_step(current_step):
            return None

        s_t = self.get_scheduled_sparsity(current_step)
        progress = (current_step - self.t_0) / float(max(1, self.t_end - self.t_0))
        # Linearly decay regrowth ratio toward 0 at t_end
        gamma_t = self.regrowth_ratio * max(0.0, 1.0 - progress)

        for i, layer in enumerate(self.linear_layers):
            is_output_layer = (i == len(self.linear_layers) - 1)
            layer_target_s = min(s_t, self.output_sparsity_cap) if is_output_layer else s_t

            self._update_layer_mask(layer, layer_target_s, gamma_t)

        return s_t

    def _update_layer_mask(self, layer: Linear, target_sparsity: float, regrowth_ratio: float) -> None:
        """
        Updates binary mask for a single linear layer using criterion and regrowth.
        """
        w = layer.weight.data
        old_mask = layer.weight.mask if layer.weight.mask is not None else np.ones_like(w)
        total_weights = old_mask.size
        target_active = int(round((1.0 - target_sparsity) * total_weights))
        target_active = max(1, min(target_active, total_weights))

        # Compute importance scores
        scores = self.criterion.compute_scores(layer)

        # Separate active and pruned indices
        active_indices = np.where(old_mask.ravel() == 1.0)[0]
        pruned_indices = np.where(old_mask.ravel() == 0.0)[0]

        n_active = len(active_indices)
        n_pruned = len(pruned_indices)

        if regrowth_ratio > 0.0 and n_pruned > 0 and n_active > 0:
            # RigL-style dynamic prune & grow
            n_regrow = int(round(regrowth_ratio * n_active))
            n_regrow = min(n_regrow, target_active, n_pruned)
            n_keep = target_active - n_regrow

            # 1. Keep top n_keep active connections by saliency score
            if n_keep > 0:
                active_scores = scores.ravel()[active_indices]
                keep_sub_idx = np.argsort(active_scores)[-n_keep:]
                chosen_active = active_indices[keep_sub_idx]
            else:
                chosen_active = np.array([], dtype=np.int64)

            # 2. Revive top n_regrow pruned connections by |unmasked_grad|
            grad = layer.weight.unmasked_grad if layer.weight.unmasked_grad is not None else layer.weight.grad
            if grad is not None:
                grad_mag = np.abs(grad.ravel()[pruned_indices])
                revive_sub_idx = np.argsort(grad_mag)[-n_regrow:]
                chosen_revived = pruned_indices[revive_sub_idx]
            else:
                chosen_revived = np.random.choice(pruned_indices, size=n_regrow, replace=False)

            new_active_flat = np.concatenate([chosen_active, chosen_revived])
        else:
            # Standard top-K magnitude / saliency pruning
            flat_scores = scores.ravel()
            top_idx = np.argsort(flat_scores)[-target_active:]
            new_active_flat = top_idx

        # Construct new binary mask
        new_mask = np.zeros(total_weights, dtype=old_mask.dtype)
        new_mask[new_active_flat] = 1.0
        new_mask = new_mask.reshape(old_mask.shape)

        # Identify newly active and newly pruned connections
        newly_active = (new_mask == 1.0) & (old_mask == 0.0)
        newly_pruned = (new_mask == 0.0)

        # Reset weights
        layer.weight.data[newly_active] = 0.0
        layer.weight.data[newly_pruned] = 0.0
        layer.weight.mask = new_mask

        # Clean optimizer momentum state for revived weights
        if self.optimizer is not None and np.any(newly_active):
            self.optimizer.reset_state_for_mask(layer.weight, newly_active)

    def get_sparsity(self) -> float:
        """Returns overall weight sparsity across all tracked linear layers."""
        total = 0
        active = 0
        for layer in self.linear_layers:
            if layer.weight.mask is not None:
                total += layer.weight.mask.size
                active += int(np.sum(layer.weight.mask))
            else:
                total += layer.weight.data.size
                active += layer.weight.data.size
        return 1.0 - (active / total) if total > 0 else 0.0
