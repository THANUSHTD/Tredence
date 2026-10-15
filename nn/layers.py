"""
Neural network layers, parameters, modules, and activation functions.
"""

from __future__ import annotations
import math
from typing import Iterator, List, Optional, Tuple, Union
import numpy as np
from engine.tensor import Tensor


class Parameter(Tensor):
    """
    Trainable parameter Tensor, requiring gradient accumulation by default.
    """

    def __init__(
        self,
        data: Union[int, float, list, np.ndarray, Tensor],
        mask: Optional[np.ndarray] = None,
        dtype: Optional[np.dtype] = None,
    ):
        super().__init__(data, requires_grad=True, mask=mask, dtype=dtype)


class Module:
    """
    Base class for all neural network modules.
    Tracks parameters and child modules.
    """

    def __init__(self):
        self._parameters: dict[str, Parameter] = {}
        self._modules: dict[str, Module] = {}

    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
        if param is not None:
            self._parameters[name] = param

    def register_module(self, name: str, module: Optional[Module]) -> None:
        if module is not None:
            self._modules[name] = module

    def __setattr__(self, name: str, value: object) -> None:
        if isinstance(value, Parameter):
            if not hasattr(self, "_parameters"):
                super().__setattr__("_parameters", {})
            self._parameters[name] = value
        elif isinstance(value, Module):
            if not hasattr(self, "_modules"):
                super().__setattr__("_modules", {})
            self._modules[name] = value
        super().__setattr__(name, value)

    def parameters(self) -> List[Parameter]:
        """Returns a flat list of all parameters in this module and submodules."""
        params: List[Parameter] = list(self._parameters.values())
        for module in self._modules.values():
            params.extend(module.parameters())
        return params

    def named_parameters(self, prefix: str = "") -> List[Tuple[str, Parameter]]:
        """Returns an iterator over module parameter names and the parameter itself."""
        items: List[Tuple[str, Parameter]] = []
        for name, param in self._parameters.items():
            full_name = f"{prefix}.{name}" if prefix else name
            items.append((full_name, param))
        for mod_name, module in self._modules.items():
            sub_prefix = f"{prefix}.{mod_name}" if prefix else mod_name
            items.extend(module.named_parameters(prefix=sub_prefix))
        return items

    def modules(self) -> List[Module]:
        """Returns a flat list of all submodules."""
        mods: List[Module] = [self]
        for module in self._modules.values():
            mods.extend(module.modules())
        return mods

    def zero_grad(self) -> None:
        """Zeroes gradients of all parameters."""
        for p in self.parameters():
            p.zero_grad()

    def forward(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)


class Linear(Module):
    """
    Fully-connected linear layer: y = x @ W_eff + b.
    Weights are initialized with Kaiming He Normal distribution: N(0, sqrt(2 / in_features)).
    Weights contain a binary mask initialized to all ones.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, dtype: np.dtype = np.float64):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Kaiming He Normal initialization
        std = math.sqrt(2.0 / in_features)
        w_data = np.random.randn(in_features, out_features).astype(dtype) * std
        w_mask = np.ones((in_features, out_features), dtype=dtype)
        self.weight = Parameter(w_data, mask=w_mask, dtype=dtype)

        if bias:
            b_data = np.zeros(out_features, dtype=dtype)
            self.bias: Optional[Parameter] = Parameter(b_data, dtype=dtype)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        # Use effective masked weight in forward graph
        w_eff = self.weight.effective()
        out = x @ w_eff
        if self.bias is not None:
            out = out + self.bias
        return out

    def get_sparsity(self) -> float:
        """Returns fraction of pruned (zeroed) weights."""
        if self.weight.mask is None:
            return 0.0
        total = self.weight.mask.size
        active = float(np.sum(self.weight.mask))
        return 1.0 - (active / total)


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class GELU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.gelu()


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()


class MLP(Module):
    """
    Multi-Layer Perceptron configurable with hidden dimensions and activation function.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dims: List[int],
        out_features: int,
        activation: str = "gelu",
        dtype: np.dtype = np.float64,
    ):
        super().__init__()
        self.layers: List[Linear] = []
        dims = [in_features] + hidden_dims + [out_features]

        act_cls = {
            "relu": ReLU,
            "gelu": GELU,
            "tanh": Tanh,
            "sigmoid": Sigmoid,
        }.get(activation.lower(), GELU)

        self.activations: List[Module] = []
        for i in range(len(dims) - 1):
            layer = Linear(dims[i], dims[i + 1], bias=True, dtype=dtype)
            self.layers.append(layer)
            self.register_module(f"linear_{i}", layer)
            if i < len(dims) - 2:
                act = act_cls()
                self.activations.append(act)
                self.register_module(f"act_{i}", act)

    def forward(self, x: Tensor) -> Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.activations):
                h = self.activations[i](h)
        return h

    def get_sparsity(self) -> float:
        """Computes aggregate weight sparsity across all Linear layers."""
        total_weights = 0
        active_weights = 0
        for layer in self.layers:
            if layer.weight.mask is not None:
                total_weights += layer.weight.mask.size
                active_weights += int(np.sum(layer.weight.mask))
            else:
                total_weights += layer.weight.data.size
                active_weights += layer.weight.data.size
        if total_weights == 0:
            return 0.0
        return 1.0 - (active_weights / total_weights)
