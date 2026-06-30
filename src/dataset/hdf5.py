import os
import os.path as osp

import h5py
import numpy as np

from .features import NODE_CONTINUOUS_DIM


def _pack_graphs(graphs: list):
    """Concatenate node/edge arrays and build boundary pointers."""
    node_ptr = [0]
    edge_ptr = [0]
    edge_ptr_2hop = [0]
    edge_ptr_3hop = [0]
    edge_ptr_4hop = [0]
    node_feat_list = []
    node_embd_list = []
    edge_feat_list = []
    edge_indices = []
    edge_indices_2hop = []
    edge_indices_3hop = []
    edge_indices_4hop = []
    edge_feat_2hop_list = []
    edge_feat_3hop_list = []
    edge_feat_4hop_list = []

    node_feat_dim = 0
    node_embd_dim = 0
    edge_feat_dim = 0
    edge_feat_2hop_dim = 0
    edge_feat_3hop_dim = 0
    edge_feat_4hop_dim = 0
    for graph in graphs:
        if node_feat_dim == 0:
            node_feat = np.asarray(graph['node_feat'])
            if node_feat.size > 0 and node_feat.ndim == 2:
                node_feat_dim = node_feat.shape[1]
        if node_embd_dim == 0:
            node_embd = np.asarray(graph.get('node_embd', np.zeros((0, NODE_CONTINUOUS_DIM), dtype=np.float16)))
            if node_embd.size > 0 and node_embd.ndim == 2:
                node_embd_dim = node_embd.shape[1]
        if edge_feat_dim == 0:
            edge_feat = np.asarray(graph['edge_feat'])
            if edge_feat.size > 0 and edge_feat.ndim == 2:
                edge_feat_dim = edge_feat.shape[1]
        if edge_feat_2hop_dim == 0:
            edge_feat_2hop = np.asarray(graph['edge_feat_2hop'])
            if edge_feat_2hop.size > 0 and edge_feat_2hop.ndim == 2:
                edge_feat_2hop_dim = edge_feat_2hop.shape[1]
        if edge_feat_3hop_dim == 0:
            edge_feat_3hop = np.asarray(graph['edge_feat_3hop'])
            if edge_feat_3hop.size > 0 and edge_feat_3hop.ndim == 2:
                edge_feat_3hop_dim = edge_feat_3hop.shape[1]
        if edge_feat_4hop_dim == 0:
            edge_feat_4hop = np.asarray(graph['edge_feat_4hop'])
            if edge_feat_4hop.size > 0 and edge_feat_4hop.ndim == 2:
                edge_feat_4hop_dim = edge_feat_4hop.shape[1]

        if node_feat_dim > 0 and node_embd_dim > 0 and edge_feat_dim > 0:
            if edge_feat_2hop_dim > 0 and edge_feat_3hop_dim > 0 and edge_feat_4hop_dim > 0:
                break

    if node_embd_dim == 0:
        node_embd_dim = NODE_CONTINUOUS_DIM
    if edge_feat_2hop_dim == 0:
        edge_feat_2hop_dim = 2
    if edge_feat_3hop_dim == 0:
        edge_feat_3hop_dim = 3
    if edge_feat_4hop_dim == 0:
        edge_feat_4hop_dim = 4

    for graph in graphs:
        node_feat = np.asarray(graph['node_feat'])
        node_embd = np.asarray(graph.get('node_embd', np.zeros((node_feat.shape[0], node_embd_dim), dtype=np.float16)))
        edge_feat = np.asarray(graph['edge_feat'])
        edge_index = np.asarray(graph['edge_index'])
        edge_index_2hop = np.asarray(graph['edge_index_2hop'])
        edge_feat_2hop = np.asarray(graph['edge_feat_2hop'])
        edge_index_3hop = np.asarray(graph['edge_index_3hop'])
        edge_feat_3hop = np.asarray(graph['edge_feat_3hop'])
        edge_index_4hop = np.asarray(graph['edge_index_4hop'])
        edge_feat_4hop = np.asarray(graph['edge_feat_4hop'])

        if node_feat.size == 0:
            node_feat = np.zeros((0, node_feat_dim), dtype=np.uint8)
        if node_embd.size == 0:
            node_embd = np.zeros((0, node_embd_dim), dtype=np.float16)
        if edge_feat.size == 0:
            edge_feat = np.zeros((0, edge_feat_dim), dtype=np.uint8)
        if edge_feat_2hop.size == 0:
            edge_feat_2hop = np.zeros((0, edge_feat_2hop_dim), dtype=np.uint8)
        if edge_feat_3hop.size == 0:
            edge_feat_3hop = np.zeros((0, edge_feat_3hop_dim), dtype=np.uint8)
        if edge_feat_4hop.size == 0:
            edge_feat_4hop = np.zeros((0, edge_feat_4hop_dim), dtype=np.uint8)

        node_feat_list.append(node_feat)
        node_embd_list.append(node_embd)
        edge_feat_list.append(edge_feat)
        edge_feat_2hop_list.append(edge_feat_2hop)
        edge_feat_3hop_list.append(edge_feat_3hop)
        edge_feat_4hop_list.append(edge_feat_4hop)

        num_nodes = node_feat.shape[0]
        num_edges = edge_index.shape[1] if edge_index.size else 0
        num_edges_2hop = edge_index_2hop.shape[1] if edge_index_2hop.size else 0
        num_edges_3hop = edge_index_3hop.shape[1] if edge_index_3hop.size else 0
        num_edges_4hop = edge_index_4hop.shape[1] if edge_index_4hop.size else 0

        if num_edges > 0:
            edge_indices.append(np.asarray(edge_index))
        else:
            edge_indices.append(np.zeros((2, 0), dtype=np.int32))
        if num_edges_2hop > 0:
            edge_indices_2hop.append(np.asarray(edge_index_2hop))
        else:
            edge_indices_2hop.append(np.zeros((2, 0), dtype=np.int32))
        if num_edges_3hop > 0:
            edge_indices_3hop.append(np.asarray(edge_index_3hop))
        else:
            edge_indices_3hop.append(np.zeros((2, 0), dtype=np.int32))
        if num_edges_4hop > 0:
            edge_indices_4hop.append(np.asarray(edge_index_4hop))
        else:
            edge_indices_4hop.append(np.zeros((2, 0), dtype=np.int32))

        node_ptr.append(node_ptr[-1] + num_nodes)
        edge_ptr.append(edge_ptr[-1] + num_edges)
        edge_ptr_2hop.append(edge_ptr_2hop[-1] + num_edges_2hop)
        edge_ptr_3hop.append(edge_ptr_3hop[-1] + num_edges_3hop)
        edge_ptr_4hop.append(edge_ptr_4hop[-1] + num_edges_4hop)

    if node_feat_list:
        node_feat_arr = np.concatenate(node_feat_list, axis=0)
    else:
        node_feat_arr = np.zeros((0, 0))

    if edge_feat_list:
        edge_feat_arr = np.concatenate(edge_feat_list, axis=0)
    else:
        edge_feat_arr = np.zeros((0, 0))

    if node_embd_list:
        node_embd_arr = np.concatenate(node_embd_list, axis=0)
    else:
        node_embd_arr = np.zeros((0, node_embd_dim), dtype=np.float16)

    if edge_indices:
        edge_index = np.concatenate(edge_indices, axis=1) if edge_indices else np.zeros((2, 0), dtype=np.int32)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int32)

    if edge_feat_2hop_list:
        edge_feat_2hop_arr = np.concatenate(edge_feat_2hop_list, axis=0)
    else:
        edge_feat_2hop_arr = np.zeros((0, edge_feat_2hop_dim), dtype=np.uint8)
    if edge_feat_3hop_list:
        edge_feat_3hop_arr = np.concatenate(edge_feat_3hop_list, axis=0)
    else:
        edge_feat_3hop_arr = np.zeros((0, edge_feat_3hop_dim), dtype=np.uint8)
    if edge_feat_4hop_list:
        edge_feat_4hop_arr = np.concatenate(edge_feat_4hop_list, axis=0)
    else:
        edge_feat_4hop_arr = np.zeros((0, edge_feat_4hop_dim), dtype=np.uint8)

    if edge_indices_2hop:
        edge_index_2hop = np.concatenate(edge_indices_2hop, axis=1) if edge_indices_2hop else np.zeros((2, 0), dtype=np.int32)
    else:
        edge_index_2hop = np.zeros((2, 0), dtype=np.int32)
    if edge_indices_3hop:
        edge_index_3hop = np.concatenate(edge_indices_3hop, axis=1) if edge_indices_3hop else np.zeros((2, 0), dtype=np.int32)
    else:
        edge_index_3hop = np.zeros((2, 0), dtype=np.int32)
    if edge_indices_4hop:
        edge_index_4hop = np.concatenate(edge_indices_4hop, axis=1) if edge_indices_4hop else np.zeros((2, 0), dtype=np.int32)
    else:
        edge_index_4hop = np.zeros((2, 0), dtype=np.int32)

    return (
        np.asarray(node_feat_arr, dtype=np.uint8),
        np.asarray(node_embd_arr, dtype=np.float16),
        np.asarray(edge_feat_arr, dtype=np.uint8),
        np.asarray(edge_index, dtype=np.uint8),
        np.asarray(node_ptr, dtype=np.int32),
        np.asarray(edge_ptr, dtype=np.int32),
        np.asarray(edge_index_2hop, dtype=np.uint8),
        np.asarray(edge_feat_2hop_arr, dtype=np.uint8),
        np.asarray(edge_ptr_2hop, dtype=np.int32),
        np.asarray(edge_index_3hop, dtype=np.uint8),
        np.asarray(edge_feat_3hop_arr, dtype=np.uint8),
        np.asarray(edge_ptr_3hop, dtype=np.int32),
        np.asarray(edge_index_4hop, dtype=np.uint8),
        np.asarray(edge_feat_4hop_arr, dtype=np.uint8),
        np.asarray(edge_ptr_4hop, dtype=np.int32),
    )


def save_graphs(path: str, graphs: list, labels: np.ndarray):
    """Persist concatenated graph tensors and labels in HDF5 format.

    Rule: save compact (uint8/float16, *_ptr as int32); load standard (int32/float32).
    Arrays from _pack_graphs already carry the correct compact dtypes.
    """
    (
        node_feat,
        node_embd,
        edge_feat,
        edge_index,
        node_ptr,
        edge_ptr,
        edge_index_2hop,
        edge_feat_2hop,
        edge_ptr_2hop,
        edge_index_3hop,
        edge_feat_3hop,
        edge_ptr_3hop,
        edge_index_4hop,
        edge_feat_4hop,
        edge_ptr_4hop,
    ) = _pack_graphs(graphs)

    os.makedirs(osp.dirname(path), exist_ok=True)
    with h5py.File(path, 'w') as f:
        f.create_dataset('labels', data=np.asarray(labels))
        f.create_dataset('node_feat', data=node_feat)
        f.create_dataset('node_embd', data=node_embd)
        f.create_dataset('edge_feat', data=edge_feat)
        f.create_dataset('edge_index', data=edge_index)
        f.create_dataset('edge_index_2hop', data=edge_index_2hop)
        f.create_dataset('edge_feat_2hop', data=edge_feat_2hop)
        f.create_dataset('edge_ptr_2hop', data=edge_ptr_2hop)
        f.create_dataset('edge_index_3hop', data=edge_index_3hop)
        f.create_dataset('edge_feat_3hop', data=edge_feat_3hop)
        f.create_dataset('edge_ptr_3hop', data=edge_ptr_3hop)
        f.create_dataset('edge_index_4hop', data=edge_index_4hop)
        f.create_dataset('edge_feat_4hop', data=edge_feat_4hop)
        f.create_dataset('edge_ptr_4hop', data=edge_ptr_4hop)
        f.create_dataset('node_ptr', data=node_ptr)
        f.create_dataset('edge_ptr', data=edge_ptr)


def load_graphs(path: str):
    """Load concatenated graph arrays and boundary pointers from HDF5."""
    with h5py.File(path, 'r') as f:
        labels = np.asarray(f['labels'][()], dtype=np.float32)
        node_feat = np.asarray(f['node_feat'][()], dtype=np.int32)
        node_embd = np.asarray(f['node_embd'][()], dtype=np.float32)
        edge_feat = np.asarray(f['edge_feat'][()], dtype=np.int32)
        edge_index = np.asarray(f['edge_index'][()], dtype=np.int32)
        edge_index_2hop = np.asarray(f['edge_index_2hop'][()], dtype=np.int32)
        edge_feat_2hop = np.asarray(f['edge_feat_2hop'][()], dtype=np.int32)
        edge_ptr_2hop = np.asarray(f['edge_ptr_2hop'][()], dtype=np.int32)
        edge_index_3hop = np.asarray(f['edge_index_3hop'][()], dtype=np.int32)
        edge_feat_3hop = np.asarray(f['edge_feat_3hop'][()], dtype=np.int32)
        edge_ptr_3hop = np.asarray(f['edge_ptr_3hop'][()], dtype=np.int32)
        edge_index_4hop = np.asarray(f['edge_index_4hop'][()], dtype=np.int32)
        edge_feat_4hop = np.asarray(f['edge_feat_4hop'][()], dtype=np.int32)
        edge_ptr_4hop = np.asarray(f['edge_ptr_4hop'][()], dtype=np.int32)
        node_ptr = np.asarray(f['node_ptr'][()], dtype=np.int32)
        edge_ptr = np.asarray(f['edge_ptr'][()], dtype=np.int32)

    return (
        labels,
        node_feat,
        node_embd,
        edge_feat,
        edge_index,
        node_ptr,
        edge_ptr,
        edge_index_2hop,
        edge_feat_2hop,
        edge_ptr_2hop,
        edge_index_3hop,
        edge_feat_3hop,
        edge_ptr_3hop,
        edge_index_4hop,
        edge_feat_4hop,
        edge_ptr_4hop,
    )
