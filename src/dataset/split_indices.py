"""HDF5 split-index loading shared by dataset and dataloader modules."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def _load_split_indices(split_path: Path, split_name: str, num_graphs: int) -> np.ndarray:
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    with h5py.File(split_path, "r") as f:
        if split_name not in f:
            available = ", ".join(sorted(f.keys()))
            raise ValueError(f"Unknown split '{split_name}'. Available splits: {available}")
        indices = np.asarray(f[split_name][()], dtype=np.int64)

    if np.any(indices < 0) or np.any(indices >= num_graphs):
        raise ValueError("Split indices contain values outside [0, num_graphs).")
    return indices


__all__ = ["_load_split_indices"]
