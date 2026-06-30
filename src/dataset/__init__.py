from .features import *
from .features import _implicit_h_count, _is_rotatable
from .graph import *
from .graph import (
    _keep_atom_mask,
    _is_polar_hydrogen,
    _centered_electronegativity,
    _gasteiger_charge,
    _rwpe,
    _rotatable_bonds,
    _khop_edges,
)
from .hdf5 import *
from .hdf5 import _pack_graphs

# Legacy aliases for older scripts and notebooks; prefer the canonical names above.
allowable_features = FEATURE_VOCAB
safe_index = vocab_index
atom_to_feature_vector = atom_features
bond_to_feature_vector = bond_features
smiles2graph = mol_to_graph
_is_rotable_bond = _is_rotatable
_non_active_hydrogen_count = _implicit_h_count
_is_active_hydrogen = _is_polar_hydrogen
_active_hydrogen_mask = _keep_atom_mask
_get_centered_en = _centered_electronegativity
_get_gasteiger_charge = _gasteiger_charge
_compute_rwpe = _rwpe
_compute_khop_edges = _khop_edges
_rotatable_bond_indices = _rotatable_bonds
_concat_graph_blocks = _pack_graphs
_save_hdf5 = save_graphs
_load_hdf5 = load_graphs

# Import the compatibility dataset API shipped as src/dataset.py.
from pathlib import Path
import importlib.util

_compat_path = Path(__file__).resolve().parent.parent / "dataset.py"
_spec = importlib.util.spec_from_file_location("_pcqm_dataset_compat", _compat_path)
if _spec is not None and _spec.loader is not None:
    _compat = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_compat)
    PCQMDataset = getattr(_compat, "PCQMDataset", None)
    PCQMDataloader = getattr(_compat, "PCQMDataloader", None)
    batch_collapse = getattr(_compat, "batch_collapse", None)
    PAD_TO_MULTIPLE = getattr(_compat, "PAD_TO_MULTIPLE", None)
    NODE_FEAT_VOCAB_SIZES = getattr(_compat, "NODE_FEAT_VOCAB_SIZES", None)
    NODE_FEAT_TOTAL_VOCAB = getattr(_compat, "NODE_FEAT_TOTAL_VOCAB", None)
    EDGE_FEAT_VOCAB_SIZES = getattr(_compat, "EDGE_FEAT_VOCAB_SIZES", None)
    EDGE_FEAT_TOTAL_VOCAB = getattr(_compat, "EDGE_FEAT_TOTAL_VOCAB", None)
    del _compat
else:
    PCQMDataset = None
    PCQMDataloader = None
    batch_collapse = None

__all__ = [
    'FEATURE_VOCAB',
    'vocab_index',
    'atom_features',
    'bond_features',
    '_implicit_h_count',
    '_is_rotatable',
    '_is_polar_hydrogen',
    '_keep_atom_mask',
    '_centered_electronegativity',
    '_gasteiger_charge',
    '_rwpe',
    '_rotatable_bonds',
    '_khop_edges',
    '_pack_graphs',
    'save_graphs',
    'load_graphs',
    'mol_to_graph',
    'allowable_features',
    'safe_index',
    'atom_to_feature_vector',
    'bond_to_feature_vector',
    'smiles2graph',
    '_is_rotable_bond',
    '_non_active_hydrogen_count',
    '_is_active_hydrogen',
    '_active_hydrogen_mask',
    '_get_centered_en',
    '_get_gasteiger_charge',
    '_compute_rwpe',
    '_compute_khop_edges',
    '_rotatable_bond_indices',
    '_concat_graph_blocks',
    '_save_hdf5',
    '_load_hdf5',
]

if PCQMDataset is not None:
    __all__ += [
        "batch_collapse",
        "PCQMDataset",
        "PCQMDataloader",
        "PAD_TO_MULTIPLE",
        "NODE_FEAT_VOCAB_SIZES",
        "NODE_FEAT_TOTAL_VOCAB",
        "EDGE_FEAT_VOCAB_SIZES",
        "EDGE_FEAT_TOTAL_VOCAB",
    ]
