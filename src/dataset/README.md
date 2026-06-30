# Dataset and Dataloader for Training

This directory contains the dataset and dataloader implementation for PCQM4Mv2 graph training with chirality-aware features.

## Overview

The dataset loads preprocessed molecular graphs from HDF5 files and batches them into a "collapsed" format suitable for JAX/Equinox training. The Chi variant extends DuAxMPNN with chiral node columns and train-split coordinate filtering.

### Key Components

- `PCQMDataset`: Handles loading from HDF5, managing splits, coord filtering on train, and graph collation.
- `PCQMDataloader`: Iterator that yields batched graph blocks from a `PCQMDataset`.
- `batch_collapse`: Merges multiple molecular graphs into one padded graph block with a prepended null graph.
- `split_indices.py`: Loads train/valid/test indices from `split_dict.h5`.
- `coord_filter.py`: Drops train graphs whose atoms lack 3D coordinates (required for chiral aux loss).
- `dataprocess.py`, `features.py`, `graph.py`, `hdf5.py`: Preprocessing from raw SMILES/SDF to HDF5 tensors.

## Usage

### 1. Initialize the Dataset

By default, it looks for data in `Path.cwd() / "dataset" / "pcqm4m-v2"`. The dataset uses `h5py` to access data on disk. Metadata like `node_ptr` and `labels` are loaded as NumPy arrays.

**Optimization**: The `PCQMDataset` class implements a shared cache. Multiple instances pointing to the same processed file share metadata arrays and the HDF5 file handle.

```python
from src.dataset.dataset import PCQMDataset

train_dataset = PCQMDataset(split="train")
valid_dataset = PCQMDataset(split="valid")
```

### 2. Create a Dataloader

```python
from src.dataset.dataset import PCQMDataset
from src.dataset.dataloader import PCQMDataloader

train_dataset = PCQMDataset(split="train")
valid_dataset = PCQMDataset(split="valid")

train_loader = PCQMDataloader(train_dataset, batch_size=256, shuffle=True)
valid_loader = PCQMDataloader(valid_dataset, batch_size=256)

for batch in train_loader:
    node_feat = batch["node_feat"]   # [total_nodes, 10]
    node_embd = batch["node_embd"]   # [total_nodes, 19]
    edge_index = batch["edge_index"] # [2, total_edges]
    labels = batch["labels"]         # [batch_size]
    break
```

### 3. Collapsed Batch Format

To support JIT-compiled JAX functions with fixed shapes, we use a "collapsed" batching strategy:

- Multiple graphs are concatenated into a single large set of nodes and edges.
- A **null graph** is prepended to the batch to act as padding.
- The total number of nodes and edges is padded to a multiple of `PAD_TO_MULTIPLE` (default 1024).

#### Batch Dictionary Keys

- `node_feat`: Node features with offsets applied.
- `node_embd`: Continuous node embeddings (19 floats per atom in Chi).
- `node_ptr`: Pointers to the start of each graph in the node array.
- `node_batch`: Graph index for each node.
- `edge_index`: Edge indices (source, target).
- `edge_feat`: Edge features with offsets applied.
- `labels`: Training labels for the molecules in the batch.
- `molecule_ids`: Original dataset indices for the molecules.

Additional edge types (e.g., `edge_2hop_index`, `edge_2hop_feat`) are included if present in the processed data.

## Topic-Specific Details

### Chiral node columns

`graph.py` appends two extra continuous columns to `node_embd` during preprocessing:

- `chiral_sign` (column 17): CIP R/S scalar in `{-1, 0, 1}` from numbering-invariant stereochemistry.
- `chiral_vol` (column 18): Signed tetrahedral volume from SDF 3D coordinates (pseudoscalar).

Layout: `[RWPE(0:12), coord(12:15), en(15), gc(16), chiral_sign(17), chiral_vol(18)]`.

### Train-split coord filtering

`coord_filter.filter_indices_with_full_coords` removes train graphs where any atom lacks non-zero 3D coordinates in `node_embd[:, 12:15]`. This ensures the chiral auxiliary volume loss has valid targets. Valid and test splits are not filtered.

### Extended `node_embd`

DuAxMPNN uses 17 floats per atom; DuAxMPNN-Chi uses 19 (adds chiral sign and volume).

## Feature Offsets

Node and edge features are stored as raw category indices. `PCQMDataset` automatically applies offsets so they can be used directly with a single large embedding table (e.g., `node_feat_total_vocab`).
