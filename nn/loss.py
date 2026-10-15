"""
Loss functions with numerical stabilization.
"""

from typing import Union
import numpy as np
from engine.tensor import Tensor
from nn.layers import Module


class SoftmaxCrossEntropy(Module):
    """
    Numerically stable Softmax Cross-Entropy Loss module.
    """

    def forward(self, logits: Tensor, targets: Union[np.ndarray, Tensor]) -> Tensor:
        return logits.softmax_cross_entropy(targets)
