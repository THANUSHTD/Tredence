"""
Pruning package initialization.
"""

from prune.criteria import MagnitudeCriterion, PruningCriterion, TaylorSaliencyCriterion
from prune.pruner import Pruner

__all__ = ["Pruner", "PruningCriterion", "MagnitudeCriterion", "TaylorSaliencyCriterion"]
