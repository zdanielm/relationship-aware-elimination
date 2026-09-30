"""
Protocol definition(s) for the relationship-aware elimination algorithm.
"""

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class WeightBackend(Protocol):
    """
    Protocol for a weight backend that provides access to hidden layers, kernels, and layer names.
    Used in ``KerasBackend`` and ``PyTorchBackend``.
    """

    def hidden_layers(self) -> list: ...
    def get_kernel(self, layer) -> np.ndarray: ...
    def set_kernel(self, layer, W: np.ndarray, b: np.ndarray) -> None: ...
    def layer_name(self, layer) -> str: ...
