import numpy as np
from rdkit import Chem
from rdkit.Chem import rdPartialCharges

from .features import (
    FEATURE_VOCAB,
    _CARBON_EN,
    COORD_DIM,
    NODE_CONTINUOUS_DIM,
    RWPE_DIM,
    _PAULING_EN,
    _ROTATABLE_BOND_SMARTS,
    vocab_index,
    atom_features,
    bond_features,
    chiral_sign_from_cip,
    signed_tetra_volume,
    _is_rotatable,
)


def _is_polar_hydrogen(atom):
    if atom.GetAtomicNum() != 1:
        return False

    neighbors = atom.GetNeighbors()
    return len(neighbors) == 1 and neighbors[0].GetAtomicNum() in {7, 8, 16}


def _keep_atom_mask(mol_with_hs):
    keep_atom = np.zeros(mol_with_hs.GetNumAtoms(), dtype=np.bool_)

    for atom in mol_with_hs.GetAtoms():
        idx = atom.GetIdx()
        if atom.GetAtomicNum() != 1:
            keep_atom[idx] = True
            continue

        if _is_polar_hydrogen(atom):
            keep_atom[idx] = True

    return keep_atom


def _centered_electronegativity(atom):
    en = _PAULING_EN.get(atom.GetAtomicNum())
    if en is None:
        en = _CARBON_EN
    return float(en - _CARBON_EN)


def _gasteiger_charge(atom) -> float:
    """Return the Gasteiger partial charge for an atom, falling back to 0.0 on failure."""
    try:
        gc = atom.GetDoubleProp('_GasteigerCharge')
        if gc != gc or abs(gc) > 4.0:
            return 0.0
        return float(gc)
    except KeyError:
        return 0.0


def _rwpe(num_nodes: int, edge_index: np.ndarray, rwpe_dim: int = RWPE_DIM) -> np.ndarray:
    """Compute random-walk positional encoding using return probabilities."""
    if num_nodes <= 0:
        return np.zeros((0, rwpe_dim), dtype=np.float16)

    adjacency = np.eye(num_nodes, dtype=np.float32)
    if edge_index.size > 0:
        src = edge_index[0]
        dst = edge_index[1]
        adjacency[src, dst] = 1.0

    degree = adjacency.sum(axis=1)
    transition = np.zeros_like(adjacency)
    nonzero_degree = degree > 0
    transition[nonzero_degree] = adjacency[nonzero_degree] / degree[nonzero_degree, None]

    rwpe = np.zeros((num_nodes, rwpe_dim), dtype=np.float32)
    power = transition.copy()
    for k in range(rwpe_dim):
        rwpe[:, k] = np.diag(power)
        power = power @ transition

    return rwpe.astype(np.float16)


def _extract_3d_coords(mol, sdf_mol, keep_atom):
    """
    Extract 3D coordinates for atoms that are kept in the returned graph.

    :param mol: RDKit molecule object with hydrogens added (and optional reordering)
    :param sdf_mol: RDKit molecule loaded from the training SDF, or None
    :param keep_atom: boolean mask over mol atoms indicating nodes to retain
    :return: np.ndarray with shape (num_kept_atoms, 3), dtype float16
    """
    if sdf_mol is None:
        return np.zeros((int(keep_atom.sum()), COORD_DIM), dtype=np.float16)

    try:
        if sdf_mol.GetNumConformers() == 0:
            return np.zeros((int(keep_atom.sum()), COORD_DIM), dtype=np.float16)
        match = sdf_mol.GetSubstructMatch(mol)
    except Exception:
        return np.zeros((int(keep_atom.sum()), COORD_DIM), dtype=np.float16)

    if len(match) != mol.GetNumAtoms() or any(m < 0 for m in match):
        return np.zeros((int(keep_atom.sum()), COORD_DIM), dtype=np.float16)

    conf = sdf_mol.GetConformer()
    coords = []
    for smi_idx in range(mol.GetNumAtoms()):
        sdf_idx = int(match[smi_idx])
        pos = conf.GetAtomPosition(sdf_idx)
        coords.append([pos.x, pos.y, pos.z])

    node_coords = np.asarray(coords, dtype=np.float32)
    node_coords = node_coords[keep_atom]
    if node_coords.shape[0] == 0:
        return np.zeros((0, COORD_DIM), dtype=np.float16)

    centroid = node_coords.mean(axis=0, keepdims=True)
    node_coords = node_coords - centroid
    return node_coords.astype(np.float16)


def _rotatable_bonds(mol) -> set[int]:
    """Return bond indices of rotatable bonds using an RDKit substructure pattern."""
    if _ROTATABLE_BOND_SMARTS is None:
        return set()

    matches = mol.GetSubstructMatches(_ROTATABLE_BOND_SMARTS)
    bond_indices = set()
    for atom_idx_a, atom_idx_b in matches:
        bond = mol.GetBondBetweenAtoms(atom_idx_a, atom_idx_b)
        if bond is not None:
            bond_indices.add(int(bond.GetIdx()))
    return bond_indices


def _khop_edges(num_nodes, edge_index):
    """Return 2-hop, 3-hop, and 4-hop directed edge lists with acyclic path counts."""
    empty_index = np.empty((2, 0), dtype=np.uint8)
    empty_2hop_feat = np.empty((0, 2), dtype=np.uint8)
    empty_3hop_feat = np.empty((0, 3), dtype=np.uint8)
    empty_4hop_feat = np.empty((0, 4), dtype=np.uint8)

    if num_nodes <= 1:
        return (
            empty_index,
            empty_2hop_feat,
            empty_index,
            empty_3hop_feat,
            empty_index,
            empty_4hop_feat,
        )

    adjacency = [set() for _ in range(num_nodes)]
    if edge_index.size > 0:
        for src, dst in zip(edge_index[0], edge_index[1]):
            adjacency[int(src)].add(int(dst))

    count_1 = {}
    for u in range(num_nodes):
        for v in adjacency[u]:
            if v > u:
                count_1[(u, v)] = 1

    count_2 = {}
    for u in range(num_nodes):
        for n1 in adjacency[u]:
            for v in adjacency[n1]:
                if v == u or v <= u:
                    continue
                count_2[(u, v)] = count_2.get((u, v), 0) + 1

    count_3 = {}
    for u in range(num_nodes):
        for n1 in adjacency[u]:
            for n2 in adjacency[n1]:
                if n2 == u:
                    continue
                for v in adjacency[n2]:
                    if v in (u, n1, n2) or v <= u:
                        continue
                    count_3[(u, v)] = count_3.get((u, v), 0) + 1

    count_4 = {}
    for u in range(num_nodes):
        for n1 in adjacency[u]:
            for n2 in adjacency[n1]:
                if n2 in (u, n1):
                    continue
                for n3 in adjacency[n2]:
                    if n3 in (u, n1, n2):
                        continue
                    for v in adjacency[n3]:
                        if v in (u, n1, n2, n3) or v <= u:
                            continue
                        count_4[(u, v)] = count_4.get((u, v), 0) + 1

    edge_index_2hop = []
    edge_feat_2hop = []
    for (u, v), c2 in count_2.items():
        if c2 <= 0:
            continue
        c1 = vocab_index(FEATURE_VOCAB['possible_path_count_list'], count_1.get((u, v), 0))
        c2_idx = vocab_index(FEATURE_VOCAB['possible_path_count_list'], c2)
        edge_index_2hop.append((u, v))
        edge_feat_2hop.append([c1, c2_idx])
        edge_index_2hop.append((v, u))
        edge_feat_2hop.append([c1, c2_idx])

    edge_index_3hop = []
    edge_feat_3hop = []
    for (u, v), c3 in count_3.items():
        if c3 <= 0:
            continue
        c1 = vocab_index(FEATURE_VOCAB['possible_path_count_list'], count_1.get((u, v), 0))
        c2_idx = vocab_index(FEATURE_VOCAB['possible_path_count_list'], count_2.get((u, v), 0))
        c3_idx = vocab_index(FEATURE_VOCAB['possible_path_count_list'], c3)
        edge_index_3hop.append((u, v))
        edge_feat_3hop.append([c1, c2_idx, c3_idx])
        edge_index_3hop.append((v, u))
        edge_feat_3hop.append([c1, c2_idx, c3_idx])

    edge_index_4hop = []
    edge_feat_4hop = []
    for (u, v), c4 in count_4.items():
        if c4 <= 0:
            continue
        c1 = vocab_index(FEATURE_VOCAB['possible_path_count_list'], count_1.get((u, v), 0))
        c2_idx = vocab_index(FEATURE_VOCAB['possible_path_count_list'], count_2.get((u, v), 0))
        c3_idx = vocab_index(FEATURE_VOCAB['possible_path_count_list'], count_3.get((u, v), 0))
        c4_idx = vocab_index(FEATURE_VOCAB['possible_path_count_list'], c4)
        edge_index_4hop.append((u, v))
        edge_feat_4hop.append([c1, c2_idx, c3_idx, c4_idx])
        edge_index_4hop.append((v, u))
        edge_feat_4hop.append([c1, c2_idx, c3_idx, c4_idx])

    if len(edge_index_2hop) == 0:
        edge_index_2hop = empty_index
        edge_feat_2hop = empty_2hop_feat
    else:
        edge_index_2hop = np.asarray(edge_index_2hop, dtype=np.uint8).T
        edge_feat_2hop = np.asarray(edge_feat_2hop, dtype=np.uint8)

    if len(edge_index_3hop) == 0:
        edge_index_3hop = empty_index
        edge_feat_3hop = empty_3hop_feat
    else:
        edge_index_3hop = np.asarray(edge_index_3hop, dtype=np.uint8).T
        edge_feat_3hop = np.asarray(edge_feat_3hop, dtype=np.uint8)

    if len(edge_index_4hop) == 0:
        edge_index_4hop = empty_index
        edge_feat_4hop = empty_4hop_feat
    else:
        edge_index_4hop = np.asarray(edge_index_4hop, dtype=np.uint8).T
        edge_feat_4hop = np.asarray(edge_feat_4hop, dtype=np.uint8)

    return (
        edge_index_2hop,
        edge_feat_2hop,
        edge_index_3hop,
        edge_feat_3hop,
        edge_index_4hop,
        edge_feat_4hop,
    )


def mol_to_graph(smiles_string, sdf_mol=None, *, h_mode="active"):
    """
    Converts SMILES string to graph dictionary.

    ``h_mode`` controls which atoms become nodes after ``AddHs``:
      - ``\"active\"``: heavy atoms + polar H (N/O/S-bonded); default, matches selective H handling.
      - ``\"all\"``: all atoms including every hydrogen.
      - ``\"heavy\"``: heavy atoms only (no explicit hydrogens).
    """
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)
    # Populate the per-atom `_CIPCode` property (R/S) used for chiral_sign.
    # Numbering-invariant and reproducible from SMILES at inference (no 3D).
    try:
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    except Exception:
        pass
    try:
        rdPartialCharges.ComputeGasteigerCharges(mol)
    except Exception:
        pass

    if h_mode == "active":
        keep_atom = _keep_atom_mask(mol)
    elif h_mode == "all":
        keep_atom = np.ones(mol.GetNumAtoms(), dtype=np.bool_)
    elif h_mode == "heavy":
        keep_atom = np.array([a.GetAtomicNum() != 1 for a in mol.GetAtoms()], dtype=np.bool_)
    else:
        raise ValueError(f"h_mode must be active|all|heavy, got {h_mode!r}")

    keep_idx = np.flatnonzero(keep_atom)
    if len(keep_idx) == 0:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_feat = np.empty((0, 6), dtype=np.int64)
        return {
            'edge_index': edge_index,
            'edge_feat': edge_feat,
            'edge_index_2hop': np.empty((2, 0), dtype=np.int64),
            'edge_feat_2hop': np.empty((0, 2), dtype=np.uint8),
            'edge_index_3hop': np.empty((2, 0), dtype=np.int64),
            'edge_feat_3hop': np.empty((0, 3), dtype=np.uint8),
            'edge_index_4hop': np.empty((2, 0), dtype=np.int64),
            'edge_feat_4hop': np.empty((0, 4), dtype=np.uint8),
            'node_feat': np.empty((0, 0), dtype=np.int64),
            'node_embd': np.zeros((0, NODE_CONTINUOUS_DIM), dtype=np.float16),
            'num_nodes': 0,
        }

    old_to_new = -np.ones(mol.GetNumAtoms(), dtype=np.int32)
    old_to_new[keep_idx] = np.arange(len(keep_idx))
    rotatable_bond_indices = _rotatable_bonds(mol)

    num_nodes = len(keep_idx)
    neighbor_lists: list[list[int]] = [[] for _ in range(num_nodes)]
    direct_bonds: list[tuple[int, int, object]] = []
    if len(mol.GetBonds()) > 0:
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            if not (keep_atom[i] and keep_atom[j]):
                continue

            ni, nj = int(old_to_new[i]), int(old_to_new[j])
            neighbor_lists[ni].append(nj)
            neighbor_lists[nj].append(ni)
            direct_bonds.append((ni, nj, bond))

    atom_feat_list = []
    node_en_list = []
    node_gc_list = []
    node_chiral_sign_list = []
    for atom in mol.GetAtoms():
        if keep_atom[atom.GetIdx()]:
            atom_feat_list.append(atom_features(atom))
            node_en_list.append(_centered_electronegativity(atom))
            node_gc_list.append(_gasteiger_charge(atom))
            node_chiral_sign_list.append(chiral_sign_from_cip(atom))

    node_coords = _extract_3d_coords(mol, sdf_mol, keep_atom)

    x = np.array(atom_feat_list, dtype=np.int64)
    node_en = np.array(node_en_list, dtype=np.float16).reshape(-1, 1)
    node_gc = np.array(node_gc_list, dtype=np.float16).reshape(-1, 1)
    node_chiral_sign = np.array(node_chiral_sign_list, dtype=np.float16).reshape(-1, 1)
    node_coords = np.asarray(node_coords, dtype=np.float16)
    node_chiral_vol = np.zeros((num_nodes, 1), dtype=np.float16)

    _TETRAHEDRAL_CHIRAL_TAGS = {'CHI_TETRAHEDRAL_CW', 'CHI_TETRAHEDRAL_CCW'}
    tetrahedral_chiral = set()
    for atom in mol.GetAtoms():
        if keep_atom[atom.GetIdx()] and str(atom.GetChiralTag()) in _TETRAHEDRAL_CHIRAL_TAGS:
            tetrahedral_chiral.add(int(old_to_new[atom.GetIdx()]))

    _RANK_MISC = len(FEATURE_VOCAB['possible_neighbor_rank_list']) - 1

    num_bond_features = 6  # bond type, bond stereo, is_conjugated, is_rotable, ring_size, rank_in_neighbor_order
    if direct_bonds:
        for neighbors in neighbor_lists:
            neighbors.sort()

        rank_lookup = [
            {neighbor: rank for rank, neighbor in enumerate(neighbors)}
            for neighbors in neighbor_lists
        ]

        # Train-only 3D grounding: signed tetrahedral volume (pseudoscalar) at
        # each tetrahedral centre, using the same rank-ordered neighbour
        # convention as `neighbor_rank`. Zero when 3D coords are unavailable.
        coords_present = bool(np.abs(node_coords).any())
        if coords_present:
            for center in tetrahedral_chiral:
                neighbors = neighbor_lists[center]
                if len(neighbors) >= 3:
                    node_chiral_vol[center, 0] = signed_tetra_volume(
                        node_coords[center],
                        [node_coords[n] for n in neighbors[:3]],
                    )

        edges_list = []
        edge_feat_list = []
        for ni, nj, bond in direct_bonds:
            edge_feature = bond_features(bond, rotatable_bond_indices)

            if nj in tetrahedral_chiral:
                rank_fwd = vocab_index(FEATURE_VOCAB['possible_neighbor_rank_list'], rank_lookup[nj].get(ni, _RANK_MISC))
            else:
                rank_fwd = _RANK_MISC
            if ni in tetrahedral_chiral:
                rank_rev = vocab_index(FEATURE_VOCAB['possible_neighbor_rank_list'], rank_lookup[ni].get(nj, _RANK_MISC))
            else:
                rank_rev = _RANK_MISC
            edge_feature_fwd = edge_feature + [rank_fwd]
            edge_feature_rev = edge_feature + [rank_rev]

            edges_list.append((ni, nj))
            edge_feat_list.append(edge_feature_fwd)
            edges_list.append((nj, ni))
            edge_feat_list.append(edge_feature_rev)

        if len(edges_list) == 0:
            edge_index = np.empty((2, 0), dtype=np.int64)
            edge_feat = np.empty((0, num_bond_features), dtype=np.int64)
        else:
            edge_index = np.array(edges_list, dtype=np.int64).T
            edge_feat = np.array(edge_feat_list, dtype=np.int64)
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_feat = np.empty((0, num_bond_features), dtype=np.int64)

    edge_index_2hop, edge_feat_2hop, edge_index_3hop, edge_feat_3hop, edge_index_4hop, edge_feat_4hop = (
        _khop_edges(len(x), edge_index)
    )

    graph = {
        'edge_index': edge_index,
        'edge_feat': edge_feat,
        'edge_index_2hop': edge_index_2hop,
        'edge_feat_2hop': edge_feat_2hop,
        'edge_index_3hop': edge_index_3hop,
        'edge_feat_3hop': edge_feat_3hop,
        'edge_index_4hop': edge_index_4hop,
        'edge_feat_4hop': edge_feat_4hop,
        'node_feat': x,
        'node_embd': np.concatenate(
            [
                _rwpe(len(x), edge_index, RWPE_DIM),
                node_coords,
                node_en,
                node_gc,
                node_chiral_sign,
                node_chiral_vol,
            ],
            axis=1,
        ),
        'num_nodes': len(x),
    }
    return graph
