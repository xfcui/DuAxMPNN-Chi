"""Core PCQM4Mv2 dataset loading and batch collapsing."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Dict, Sequence

import h5py
import numpy as np

PAD_TO_MULTIPLE = 1024


def _load_feature_vocab() -> Dict[str, list]:
    feature_file = Path(__file__).resolve().parent / "features.py"
    spec = importlib.util.spec_from_file_location("_pcqm_dataset_features", feature_file)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load feature vocabulary module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    feature_vocab = getattr(module, "FEATURE_VOCAB", None)
    if feature_vocab is None:
        raise RuntimeError("FEATURE_VOCAB not found in features module.")
    return feature_vocab


_FEATURE_VOCAB = _load_feature_vocab()

NODE_FEAT_VOCAB_SIZES = [
    len(_FEATURE_VOCAB["possible_atomic_num_list"]),
    len(_FEATURE_VOCAB["possible_chirality_list"]),
    len(_FEATURE_VOCAB["possible_degree_list"]),
    len(_FEATURE_VOCAB["possible_formal_charge_list"]),
    len(_FEATURE_VOCAB["possible_numH_list"]),
    len(_FEATURE_VOCAB["possible_number_radical_e_list"]),
    len(_FEATURE_VOCAB["possible_hybridization_list"]),
    len(_FEATURE_VOCAB["possible_is_aromatic_list"]),
    len(_FEATURE_VOCAB["possible_is_in_ring_list"]),
    len(_FEATURE_VOCAB["possible_ring_size_list"]),
]

EDGE_FEAT_VOCAB_SIZES = {
    "": [
        len(_FEATURE_VOCAB["possible_bond_type_list"]),
        len(_FEATURE_VOCAB["possible_bond_stereo_list"]),
        len(_FEATURE_VOCAB["possible_is_conjugated_list"]),
        len(_FEATURE_VOCAB["possible_is_rotable_list"]),
        len(_FEATURE_VOCAB["possible_neighbor_rank_list"]),
        len(_FEATURE_VOCAB["possible_ring_size_list"]),
    ],
    "_2hop": [len(_FEATURE_VOCAB["possible_path_count_list"])] * 2,
    "_3hop": [len(_FEATURE_VOCAB["possible_path_count_list"])] * 3,
    "_4hop": [len(_FEATURE_VOCAB["possible_path_count_list"])] * 4,
}

NODE_FEAT_TOTAL_VOCAB = int(1 + np.sum(NODE_FEAT_VOCAB_SIZES))
EDGE_FEAT_TOTAL_VOCAB = {
    suffix: int(1 + np.sum(size_list))
    for suffix, size_list in EDGE_FEAT_VOCAB_SIZES.items()
}


def _compute_offsets(sizes: list[int]) -> np.ndarray:
    if not sizes:
        return np.zeros(0, dtype=np.int32)
    return np.asarray([1] + list(np.cumsum(sizes[:-1], dtype=np.int32)), dtype=np.int32)


def _default_dataset_root() -> Path:
    return Path.cwd() / "dataset" / "pcqm4m-v2"


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
compute_graph_full_coord_mask = _COORD_FILTER_MODULE.compute_graph_full_coord_mask
filter_indices_with_full_coords = _COORD_FILTER_MODULE.filter_indices_with_full_coords


def _multiple_of(x: int, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError("`multiple` must be a positive integer.")
    return ((x + multiple - 1) // multiple) * multiple


class PCQMDataset:
    """Load preprocessed PCQM4Mv2 arrays and provide batch collation helpers."""

    _shared_cache: Dict[str, Dict] = {}

    def __init__(
        self,
        dataset_root: Path | str | None = None,
        split: str | None = "train",
        split_file: Path | str | None = None,
        load_in_memory: bool = True,
        processed_h5: str | Path = "data_processed.h5",
    ) -> None:
        self.dataset_root = Path(dataset_root or _default_dataset_root())
        self.data_file = self.dataset_root / "processed" / Path(processed_h5).name
        if not self.data_file.exists():
            raise FileNotFoundError(f"Processed HDF5 not found: {self.data_file}")

        cache_key = f"{self.data_file.resolve()}|{'mem' if load_in_memory else 'disk'}"
        if cache_key not in PCQMDataset._shared_cache:
            if load_in_memory:
                with h5py.File(self.data_file, "r") as h5_file:
                    node_ptr = np.asarray(h5_file["node_ptr"][()], dtype=np.int32)
                    labels = np.asarray(h5_file["labels"][()], dtype=np.float32)
                    node_feat = np.asarray(h5_file["node_feat"][()], dtype=np.int32)
                    node_embd = np.asarray(h5_file["node_embd"][()], dtype=np.float32)

                    # Discover edge kinds once
                    keys = set(h5_file.keys())
                    edge_kinds = []
                    for key in sorted(keys):
                        if key == "edge_index" or key.startswith("edge_index_"):
                            suffix = key[len("edge_index"):]
                            if f"edge_ptr{suffix}" in keys and f"edge_feat{suffix}" in keys:
                                edge_kinds.append(suffix)
                    if "" not in edge_kinds and "edge_index" in keys:
                        edge_kinds.append("")

                    edge_ptrs = {
                        suffix: np.asarray(h5_file[f"edge_ptr{suffix}"][()], dtype=np.int32)
                        for suffix in edge_kinds
                    }
                    edge_indices = {
                        suffix: np.asarray(h5_file[f"edge_index{suffix}"][()], dtype=np.int32)
                        for suffix in edge_kinds
                    }
                    edge_feats = {
                        suffix: np.asarray(h5_file[f"edge_feat{suffix}"][()], dtype=np.int32)
                        for suffix in edge_kinds
                    }
            else:
                h5_file = h5py.File(self.data_file, "r")
                node_ptr = np.asarray(h5_file["node_ptr"][()], dtype=np.int32)
                labels = np.asarray(h5_file["labels"][()], dtype=np.float32)
                node_feat = h5_file["node_feat"]
                node_embd = h5_file["node_embd"]

                # Discover edge kinds once
                keys = set(h5_file.keys())
                edge_kinds = []
                for key in sorted(keys):
                    if key == "edge_index" or key.startswith("edge_index_"):
                        suffix = key[len("edge_index"):]
                        if f"edge_ptr{suffix}" in keys and f"edge_feat{suffix}" in keys:
                            edge_kinds.append(suffix)
                if "" not in edge_kinds and "edge_index" in keys:
                    edge_kinds.append("")

                edge_ptrs = {
                    suffix: np.asarray(h5_file[f"edge_ptr{suffix}"][()], dtype=np.int32)
                    for suffix in edge_kinds
                }
                edge_indices = {
                    suffix: h5_file[f"edge_index{suffix}"]
                    for suffix in edge_kinds
                }
                edge_feats = {
                    suffix: h5_file[f"edge_feat{suffix}"]
                    for suffix in edge_kinds
                }

            PCQMDataset._shared_cache[cache_key] = {
                "h5_file": h5_file if not load_in_memory else None,
                "node_ptr": node_ptr,
                "labels": labels,
                "node_feat": node_feat,
                "node_embd": node_embd,
                "edge_kinds": edge_kinds,
                "edge_ptrs": edge_ptrs,
                "edge_indices": edge_indices,
                "edge_feats": edge_feats,
                "load_in_memory": load_in_memory,
                "graph_full_coord_mask": compute_graph_full_coord_mask(node_ptr, node_embd),
            }

        shared = PCQMDataset._shared_cache[cache_key]
        self.h5_file = shared["h5_file"]
        self.node_ptr = shared["node_ptr"]
        self.labels = shared["labels"]
        self.node_feat = shared["node_feat"]
        self.node_embd = shared["node_embd"]
        self.edge_kinds = shared["edge_kinds"]
        self.edge_ptrs = shared["edge_ptrs"]
        self.edge_indices = shared["edge_indices"]
        self.edge_feats = shared["edge_feats"]
        self._load_in_memory = shared["load_in_memory"]
        self._graph_full_coord_mask = shared["graph_full_coord_mask"]

        self.num_graphs = int(self.node_ptr.shape[0] - 1)
        self.node_feat_vocab_sizes = list(NODE_FEAT_VOCAB_SIZES)
        self.node_feat_total_vocab = NODE_FEAT_TOTAL_VOCAB
        
        self.edge_feat_dtypes = {
            suffix: self.edge_feats[suffix].dtype
            for suffix in self.edge_kinds
        }
        self.edge_feat_dims = {
            suffix: int(self.edge_feats[suffix].shape[1])
            for suffix in self.edge_kinds
        }
        self.edge_feat_vocab_sizes = {
            suffix: EDGE_FEAT_VOCAB_SIZES.get(
                suffix, [len(_FEATURE_VOCAB["possible_path_count_list"])] * self.edge_feat_dims[suffix]
            )
            for suffix in self.edge_kinds
        }
        self.edge_feat_total_vocab = {
            suffix: int(1 + int(np.sum(size_list)))
            for suffix, size_list in self.edge_feat_vocab_sizes.items()
        }
        self._node_feat_offsets = _compute_offsets(self.node_feat_vocab_sizes)
        self._edge_feat_offsets = {
            suffix: _compute_offsets(size_list) for suffix, size_list in self.edge_feat_vocab_sizes.items()
        }

        if split is None:
            self.split_indices = np.arange(self.num_graphs, dtype=np.int64)
        else:
            split_file = Path(split_file or self.dataset_root / "split_dict.h5")
            self.split_indices = _load_split_indices(split_file, split, self.num_graphs)

        # Filter out invalid labels (label < 0)
        valid_mask = self.labels[self.split_indices] >= 0
        self.split_indices = self.split_indices[valid_mask]

        # Train split: drop molecules with empty graphs or any zero-filled coord atoms.
        if split == "train" and self._graph_full_coord_mask is not None:
            self.split_indices = filter_indices_with_full_coords(
                self.split_indices,
                self._graph_full_coord_mask,
            )

    def close(self) -> None:
        # For in-memory datasets there's no open file handle to close.
        if self.h5_file is not None and not self._load_in_memory:
            # Keep current shared-cache semantics: individual close does not evict the cache.
            # This preserves previous behavior while still exposing a close hook.
            pass

    def __del__(self) -> None:
        pass

    def _discover_edge_kinds(self) -> list[str]:
        return self.edge_kinds

    def get_graph_count(self) -> int:
        return self.num_graphs

    def get_split_indices(self) -> np.ndarray:
        return self.split_indices.copy()

    def _get_node_block(self, graph_id: int) -> np.ndarray:
        start, end = self.node_ptr[graph_id], self.node_ptr[graph_id + 1]
        return np.asarray(self.node_feat[start:end], dtype=np.int32)

    def _get_node_emb_block(self, graph_id: int) -> np.ndarray:
        start, end = self.node_ptr[graph_id], self.node_ptr[graph_id + 1]
        return np.asarray(self.node_embd[start:end], dtype=np.float32)

    def _get_edge_blocks(
        self,
        graph_id: int,
        suffix: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        edge_ptr = self.edge_ptrs[suffix]
        start, end = edge_ptr[graph_id], edge_ptr[graph_id + 1]
        edge_index = np.asarray(self.edge_indices[suffix][:, start:end], dtype=np.int32)
        edge_feat = np.asarray(self.edge_feats[suffix][start:end], dtype=np.int32)
        return edge_index, edge_feat

    def batch_collapse(
        self,
        graph_ids: Sequence[int],
        *,
        pad_to_multiple: int = PAD_TO_MULTIPLE,
    ) -> Dict[str, np.ndarray]:
        """
        Collapse a molecule list into one batched graph block.

        Padding rules:
          - prepend one dynamic null graph whose size is chosen so that total nodes/edges
            become multiples of `pad_to_multiple`.
          - no trailing padded rows are appended after valid graphs.
        """
        ids = np.asarray(graph_ids, dtype=np.int64).ravel()
        if np.any(ids < 0) or np.any(ids >= self.num_graphs):
            raise ValueError("All graph IDs must be in [0, num_graphs).")
        n_graphs = int(ids.size)
        total_graphs = n_graphs + 1  # +1 for null graph at the front
        node_counts = np.asarray(self.node_ptr[ids + 1] - self.node_ptr[ids], dtype=np.int32)

        valid_node_count = int(np.sum(node_counts))
        valid_node_counts = np.asarray(node_counts, dtype=np.int32)
        node_target = _multiple_of(valid_node_count + 1, pad_to_multiple)
        null_node_count = node_target - valid_node_count
        node_ptr = np.zeros(total_graphs + 1, dtype=np.int32)
        node_ptr[0] = 0
        node_ptr[1] = null_node_count
        if n_graphs > 0:
            np.cumsum(valid_node_counts, out=node_ptr[2:])
            node_ptr[2:] += null_node_count

        node_feat_padded = np.zeros((node_target, int(self.node_feat.shape[1])), dtype=np.int32)
        node_embd_padded = np.zeros((node_target, int(self.node_embd.shape[1])), dtype=np.float32)

        if valid_node_count > 0:
            node_cursor = null_node_count
            node_starts = self.node_ptr[ids]
            node_ends = self.node_ptr[ids + 1]
            for start, end in zip(node_starts, node_ends):
                n_nodes = int(end - start)
                if n_nodes == 0:
                    continue
                node_feat_block = np.asarray(self.node_feat[start:end], dtype=np.int32)
                node_embd_block = np.asarray(self.node_embd[start:end], dtype=np.float32)
                node_feat_block = node_feat_block + self._node_feat_offsets
                node_feat_padded[node_cursor : node_cursor + n_nodes] = node_feat_block
                node_embd_padded[node_cursor : node_cursor + n_nodes] = node_embd_block
                node_cursor += n_nodes

        node_batch = np.zeros(node_target, dtype=np.int32)
        for batch_gid in range(total_graphs):
            s, e = node_ptr[batch_gid], node_ptr[batch_gid + 1]
            node_batch[s:e] = batch_gid

        out: Dict[str, np.ndarray] = {
            "node_feat": node_feat_padded,
            "node_embd": node_embd_padded,
            "node_ptr": node_ptr,
            "node_batch": node_batch,
            "labels": np.asarray(self.labels[ids], dtype=np.float32),
            "molecule_ids": ids,
            "batch_n_graphs": np.int32(n_graphs),
        }

        node_offsets = np.zeros(n_graphs + 1, dtype=np.int32)
        if n_graphs > 0:
            np.cumsum(node_counts, out=node_offsets[1:])

        for suffix in self.edge_kinds:
            edge_ptr_lookup = self.edge_ptrs[suffix]
            valid_edge_counts = np.asarray(edge_ptr_lookup[ids + 1] - edge_ptr_lookup[ids], dtype=np.int32)
            valid_edge_count = int(np.sum(valid_edge_counts))
            edge_target = _multiple_of(valid_edge_count + 1, pad_to_multiple)
            null_edge_count = edge_target - valid_edge_count

            edge_ptr = np.zeros(total_graphs + 1, dtype=np.int32)
            edge_ptr[0] = 0
            edge_ptr[1] = null_edge_count
            if n_graphs > 0:
                np.cumsum(valid_edge_counts, out=edge_ptr[2:])
                edge_ptr[2:] += null_edge_count

            edge_index_padded = np.zeros((2, edge_target), dtype=np.int32)
            edge_feat_padded = np.zeros((edge_target, self.edge_feat_dims[suffix]), dtype=np.int32)
            edge_batch = np.zeros(edge_target, dtype=np.int32)

            if valid_edge_count > 0:
                edge_cursor = null_edge_count
                edge_offsets = self._edge_feat_offsets[suffix]
                edge_starts = edge_ptr_lookup[ids]
                edge_ends = edge_ptr_lookup[ids + 1]

                for local_idx, (es, ee) in enumerate(zip(edge_starts, edge_ends)):
                    local_count = int(ee - es)
                    if local_count == 0:
                        continue
                    edge_idx_raw = np.asarray(self.edge_indices[suffix][:, es:ee], dtype=np.int32)
                    edge_feat_raw = np.asarray(self.edge_feats[suffix][es:ee], dtype=np.int32)

                    edge_idx_raw = edge_idx_raw + node_offsets[local_idx] + null_node_count
                    edge_feat_raw = edge_feat_raw + edge_offsets

                    edge_index_padded[:, edge_cursor : edge_cursor + local_count] = edge_idx_raw
                    edge_feat_padded[edge_cursor : edge_cursor + local_count] = edge_feat_raw
                    edge_cursor += local_count

            for batch_gid in range(total_graphs):
                s, e = edge_ptr[batch_gid], edge_ptr[batch_gid + 1]
                edge_batch[s:e] = batch_gid

            edge_suffix_name = f"edge{suffix}"
            out[f"{edge_suffix_name}_index"] = edge_index_padded
            out[f"{edge_suffix_name}_feat"] = edge_feat_padded
            out[f"{edge_suffix_name}_ptr"] = edge_ptr
            out[f"{edge_suffix_name}_batch"] = edge_batch

        return out


def batch_collapse(
    dataset: PCQMDataset,
    graph_ids: Sequence[int],
    *,
    pad_to_multiple: int = PAD_TO_MULTIPLE,
) -> Dict[str, np.ndarray]:
    """Collapse graphs using ``PAD_TO_MULTIPLE`` (1024) when padding is unspecified."""
    return dataset.batch_collapse(graph_ids, pad_to_multiple=pad_to_multiple)


__all__ = [
    "batch_collapse",
    "PCQMDataset",
    "PAD_TO_MULTIPLE",
    "NODE_FEAT_VOCAB_SIZES",
    "NODE_FEAT_TOTAL_VOCAB",
    "EDGE_FEAT_VOCAB_SIZES",
    "EDGE_FEAT_TOTAL_VOCAB",
    "_load_split_indices",
]
