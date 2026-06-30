"""Batch iterator for PCQM4Mv2 collapsed graph batches."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from typing import Dict, Iterable, Sequence


def _load_split_indices_module():
    split_file = Path(__file__).resolve().parent / "split_indices.py"
    spec = importlib.util.spec_from_file_location("_pcqm_split_indices", split_file)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load split indices module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SPLIT_INDICES_MODULE = _load_split_indices_module()
_load_split_indices = _SPLIT_INDICES_MODULE._load_split_indices


def _load_coord_filter_module():
    coord_file = Path(__file__).resolve().parent / "coord_filter.py"
    spec = importlib.util.spec_from_file_location("_pcqm_coord_filter", coord_file)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load coord filter module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_COORD_FILTER_MODULE = _load_coord_filter_module()
filter_indices_with_full_coords = _COORD_FILTER_MODULE.filter_indices_with_full_coords


class PCQMDataloader:
    """Simple batch iterator around `PCQMDataset.batch_collapse`.

    ``shuffle=False`` draws one random permutation at construction and reuses it
    for every pass over the data. ``shuffle=True`` draws a new permutation each
    time the iterator is started (e.g. each training epoch). Use ``seed`` for
    reproducibility.
    """

    def __init__(
        self,
        dataset: "PCQMDataset",
        *,
        indices: Sequence[int] | None = None,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        pad_to_multiple: int | None = None,
        seed: int | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.pad_to_multiple = (
            self.batch_size * 4 if pad_to_multiple is None else int(pad_to_multiple)
        )
        self.seed = seed

        if indices is None:
            self.indices = np.asarray(dataset.get_split_indices(), dtype=np.int64)
        else:
            candidate = np.asarray(indices, dtype=np.int64)
            if np.any(candidate < 0) or np.any(candidate >= dataset.get_graph_count()):
                raise ValueError("All dataset indices must be in [0, num_graphs).")
            self.indices = candidate

        # shuffle=False: one random permutation at construction (fixed across epochs).
        # shuffle=True: fresh permutation on every __iter__ (each epoch).
        if self.shuffle:
            self._epoch_rng = np.random.default_rng(seed)
            self._fixed_order: np.ndarray | None = None
        else:
            self._epoch_rng = None
            order_once = self.indices.copy()
            np.random.default_rng(seed).shuffle(order_once)
            self._fixed_order = order_once

    def __iter__(self) -> Iterable[Dict[str, np.ndarray]]:
        n = self.indices.size
        if n == 0:
            return iter(())

        if self.shuffle:
            order = self.indices.copy()
            self._epoch_rng.shuffle(order)
        else:
            order = self._fixed_order

        def _iter() -> Iterable[Dict[str, np.ndarray]]:
            for start in range(0, n, self.batch_size):
                end = start + self.batch_size
                batch_ids = order[start:end]
                if batch_ids.size == 0:
                    continue
                if self.drop_last and batch_ids.size < self.batch_size:
                    break
                yield self.dataset.batch_collapse(
                    batch_ids,
                    pad_to_multiple=self.pad_to_multiple,
                )

        return _iter()

    def __len__(self) -> int:
        if self.drop_last:
            return self.indices.size // self.batch_size
        return (self.indices.size + self.batch_size - 1) // self.batch_size

    def get_split(self, split_name: str) -> "PCQMDataloader":
        # Check if the dataset already has these indices to avoid reloading from split_dict.h5
        # if possible, but PCQMDataset currently only stores one set of split_indices.
        # We'll stick to _load_split_indices for now as it's cleaner.
        indices = _load_split_indices(self.dataset.dataset_root / "split_dict.h5", split_name, self.dataset.get_graph_count())
        
        # Filter out invalid labels (label < 0)
        valid_mask = self.dataset.labels[indices] >= 0
        indices = indices[valid_mask]

        if split_name == "train" and self.dataset._graph_full_coord_mask is not None:
            indices = filter_indices_with_full_coords(
                indices,
                self.dataset._graph_full_coord_mask,
            )

        return PCQMDataloader(
            self.dataset,
            indices=indices,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            pad_to_multiple=self.pad_to_multiple,
            seed=self.seed,
        )


def batch_collapse(
    dataset: "PCQMDataset",
    graph_ids: Sequence[int],
    *,
    pad_to_multiple: int | None = None,
) -> Dict[str, np.ndarray]:
    """Collapse graphs; default padding is ``4 * len(graph_ids)`` (dataloader batch heuristic)."""
    n = len(graph_ids)
    resolved = (max(1, n) * 4) if pad_to_multiple is None else int(pad_to_multiple)
    return dataset.batch_collapse(graph_ids, pad_to_multiple=resolved)


__all__ = [
    "batch_collapse",
    "PCQMDataloader",
    "_load_split_indices",
]
