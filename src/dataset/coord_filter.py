"""Filter graphs whose atoms lack 3D coordinates in ``node_embd``."""

from __future__ import annotations

import numpy as np

RWPE_DIM = 12
COORD_DIM = 3


def compute_graph_full_coord_mask(node_ptr: np.ndarray, node_embd) -> np.ndarray | None:
    """Return a per-graph mask: True when non-empty and every atom has non-zero coords.

    Coordinates live in ``node_embd[:, RWPE_DIM:RWPE_DIM + COORD_DIM]``. Returns ``None``
    when ``node_embd`` is too narrow (e.g. toy fixtures) so callers can skip filtering.
    """
    try:
        n_cols = int(node_embd.shape[1])
    except (AttributeError, TypeError, ValueError):
        embd = np.asarray(node_embd, dtype=np.float32)
        n_cols = int(embd.shape[1])
        if n_cols < RWPE_DIM + COORD_DIM:
            return None
        coord_block = embd[:, RWPE_DIM : RWPE_DIM + COORD_DIM]
    else:
        if n_cols < RWPE_DIM + COORD_DIM:
            return None
        coord_block = np.asarray(
            node_embd[:, RWPE_DIM : RWPE_DIM + COORD_DIM],
            dtype=np.float32,
        )
    atom_has_coord = (np.abs(coord_block) > 0).any(axis=1)

    lengths = node_ptr[1:] - node_ptr[:-1]
    n_graphs = int(lengths.shape[0])
    mask = np.ones(n_graphs, dtype=bool)
    mask[lengths == 0] = False

    if (~atom_has_coord).any():
        node_graph = np.repeat(np.arange(n_graphs, dtype=np.int64), lengths.astype(np.int64))
        mask[np.unique(node_graph[~atom_has_coord])] = False

    return mask


def filter_indices_with_full_coords(
    indices: np.ndarray,
    graph_full_coord_mask: np.ndarray | None,
) -> np.ndarray:
    """Drop graph IDs that fail :func:`compute_graph_full_coord_mask`."""
    if graph_full_coord_mask is None:
        return indices
    return indices[graph_full_coord_mask[indices]]


__all__ = [
    "COORD_DIM",
    "RWPE_DIM",
    "compute_graph_full_coord_mask",
    "filter_indices_with_full_coords",
]
