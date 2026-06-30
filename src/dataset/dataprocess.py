#!/usr/bin/env python3
"""Convert SMILES strings into the project graph HDF5 format."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import SDMolSupplier
from tqdm.auto import tqdm

# Make repository root importable when running from any working directory.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.dataset import mol_to_graph, save_graphs
from src.dataset.features import NODE_CONTINUOUS_DIM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess SMILES into processed HDF5 graphs.")
    parser.add_argument(
        "--dataset-root",
        default="dataset",
        help="Dataset base directory (default: dataset).",
    )
    parser.add_argument(
        "--dataset-name",
        default="pcqm4m-v2",
        help="Dataset folder name under --dataset-root (default: pcqm4m-v2).",
    )
    parser.add_argument(
        "--raw-csv",
        default=None,
        help="Optional raw SMILES csv path; defaults to <dataset-root>/<dataset-name>/raw/data.csv.gz",
    )
    parser.add_argument(
        "--sdf",
        default=None,
        help="Optional SDF path used for 3D coordinates; defaults to <dataset-root>/<dataset-name>/raw/pcqm4m-v2-train.sdf",
    )
    parser.add_argument(
        "--smiles-col",
        default="smiles",
        help="Name of the SMILES column in the raw CSV (default: smiles).",
    )
    parser.add_argument(
        "--label-col",
        default="homolumogap",
        help="Name of the target column in the raw CSV (default: homolumogap).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Output HDF5 path; defaults to "
            "<dataset-root>/<dataset-name>/processed/data_processed.h5"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output even if it already exists.",
    )
    parser.add_argument(
        "--h-mode",
        choices=("active", "all", "heavy"),
        default="active",
        help="Hydrogen handling for graph nodes: active (default), all H, or heavy-only.",
    )
    return parser.parse_args()


def _build_empty_graph() -> dict:
    return {
        "edge_index": np.empty((2, 0), dtype=np.int64),
        "edge_feat": np.empty((0, 5), dtype=np.uint8),
        "edge_index_2hop": np.empty((2, 0), dtype=np.int64),
        "edge_feat_2hop": np.empty((0, 2), dtype=np.uint8),
        "edge_index_3hop": np.empty((2, 0), dtype=np.int64),
        "edge_feat_3hop": np.empty((0, 3), dtype=np.uint8),
        "edge_index_4hop": np.empty((2, 0), dtype=np.int64),
        "edge_feat_4hop": np.empty((0, 4), dtype=np.uint8),
        "node_feat": np.empty((0, 0), dtype=np.int64),
        "node_embd": np.zeros((0, NODE_CONTINUOUS_DIM), dtype=np.float16),
        "num_nodes": 0,
    }


def _load_graphs_from_smiles(
    csv_path: Path,
    smiles_col: str,
    label_col: str,
    sdf_path: Path | None,
    *,
    h_mode: str,
) -> tuple[list[dict], np.ndarray]:
    data = pd.read_csv(csv_path)
    if smiles_col not in data.columns:
        raise ValueError(f"Missing smiles column: {smiles_col}")
    if label_col not in data.columns:
        raise ValueError(
            f"Missing label column: {label_col}. Set --label-col to a valid column name."
        )

    smiles_series = data[smiles_col]
    labels_series = pd.to_numeric(data[label_col], errors="coerce")

    supplier = None
    sdf_len = -1
    if sdf_path is not None and sdf_path.exists():
        RDLogger.DisableLog("rdApp.*")
        supplier = SDMolSupplier(str(sdf_path), removeHs=False)
        sdf_len = len(supplier) if supplier is not None else -1
        if sdf_len >= 0:
            print(f"Loaded SDF with {sdf_len} entries from {sdf_path}")

    graphs: list[dict] = []
    labels: list[float] = []
    failed = 0
    skipped = 0

    for i in tqdm(range(len(smiles_series)), desc="Converting SMILES -> graph"):
        smiles = smiles_series.iloc[i]
        if pd.isna(smiles):
            skipped += 1
            graphs.append(_build_empty_graph())
            labels.append(float("nan"))
            continue

        if Chem.MolFromSmiles(str(smiles)) is None:
            failed += 1
            graphs.append(_build_empty_graph())
            labels.append(-1.0)
            continue

        try:
            sdf_mol = supplier[i] if (supplier is not None and i < sdf_len) else None
            graph = mol_to_graph(str(smiles), sdf_mol=sdf_mol, h_mode=h_mode)
            if graph is None:
                failed += 1
                graph = _build_empty_graph()
                label = -1.0
            else:
                label = float(labels_series.iloc[i]) if pd.notna(labels_series.iloc[i]) else -1.0
        except Exception:
            failed += 1
            graph = _build_empty_graph()
            label = -1.0

        graphs.append(graph)
        labels.append(label)

    print(f"Total invalid/failed SMILES: {failed}")
    print(f"Total missing SMILES entries: {skipped}")
    return graphs, np.asarray(labels, dtype=np.float32)


def _print_dataset_info(out_path: Path) -> None:
    with h5py.File(out_path, "r") as f:
        print(f"Processed dataset file: {out_path}")
        for key in f:
            obj = f[key]
            if isinstance(obj, h5py.Dataset):
                print(f"  {key}: dtype={obj.dtype}, shape={obj.shape}")
            else:
                print(f"  {key}: <group>")


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    raw_csv = Path(args.raw_csv) if args.raw_csv else dataset_root / args.dataset_name / "raw" / "data.csv.gz"
    sdf_path = Path(args.sdf) if args.sdf else dataset_root / args.dataset_name / "raw" / "pcqm4m-v2-train.sdf"
    if args.out:
        out_path = Path(args.out)
    else:
        sub = "data_processed.h5" if args.h_mode == "active" else f"data_processed_{args.h_mode}.h5"
        out_path = dataset_root / args.dataset_name / "processed" / sub

    if not raw_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {raw_csv}")

    if out_path.exists() and not args.overwrite:
        print(f"Output exists; skip regeneration. Use --overwrite to replace it.")
        _print_dataset_info(out_path)
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    graphs, labels = _load_graphs_from_smiles(
        raw_csv, args.smiles_col, args.label_col, sdf_path, h_mode=args.h_mode
    )
    save_graphs(out_path, graphs, labels)
    print(f"Saved processed dataset -> {out_path}")
    _print_dataset_info(out_path)


if __name__ == "__main__":
    main()
