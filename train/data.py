"""
Dataset loader and mini-batch generator.

NOTE ON PERMITTED LIBRARIES:
As per specification constraints, scikit-learn is used solely in this file
to load standard benchmark dataset arrays (e.g., load_digits).
All preprocessing, tensor wrapping, mini-batching, and shuffling are
implemented in pure Python and NumPy.
"""

from typing import Iterator, Tuple
import numpy as np
from engine.tensor import Tensor

# scikit-learn is ONLY permitted for loading dataset arrays
from sklearn.datasets import load_digits


def load_digit_dataset(
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """
    Loads 8x8 handwritten digits (1797 samples, 64 features, 10 classes).
    Applies z-score standardization (zero mean, unit variance based on train split).
    Splits into deterministic train, val, and test subsets.
    """
    rng = np.random.RandomState(seed)
    digits = load_digits()
    X = digits.data.astype(np.float64)  # Shape (1797, 64)
    y = digits.target.astype(np.int64)  # Shape (1797,)

    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Standardize features using training statistics
    mean = np.mean(X_train, axis=0, keepdims=True)
    std = np.std(X_train, axis=0, keepdims=True)
    # Prevent division by zero on constant border pixels
    std[std < 1e-6] = 1.0

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


class DataLoader:
    """
    Deterministic mini-batch iterator yielding (Tensor(X_batch), y_batch).
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 64,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.n_samples = X.shape[0]

    def __len__(self) -> int:
        return int(np.ceil(self.n_samples / self.batch_size))

    def __iter__(self) -> Iterator[Tuple[Tensor, np.ndarray]]:
        rng = np.random.RandomState(self.seed + self.epoch)
        self.epoch += 1

        indices = np.arange(self.n_samples)
        if self.shuffle:
            rng.shuffle(indices)

        for start in range(0, self.n_samples, self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            X_batch = Tensor(self.X[batch_idx], requires_grad=False)
            y_batch = self.y[batch_idx]
            yield X_batch, y_batch
