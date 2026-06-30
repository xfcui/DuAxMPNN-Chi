from rdkit import Chem


FEATURE_VOCAB = {
    'possible_atomic_num_list': list(range(1, 35)) + ['misc'],
    'possible_chirality_list': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
        'misc',
    ],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list': [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc',
    ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring_list': [False, True],
    'possible_bond_type_list': [
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
        'misc',
    ],
    'possible_bond_stereo_list': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ],
    'possible_is_conjugated_list': [False, True],
    'possible_is_rotable_list': [False, True],
    'possible_neighbor_rank_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_path_count_list': [0, 1, 2, 'misc'],
    'possible_ring_size_list': [3, 4, 5, 6, 7, 8, 'misc'],
}


_CARBON_EN = 2.55
_HBOND_DONOR_ATOMIC_NUMBERS = {7, 8, 16}
# Strict rotatable bond SMARTS (Oprea definition).
# Acyclic single bond where neither end is terminal, triple-bonded, methyl,
# trihalomethyl (CF3/CCl3/CBr3), or t-butyl.  Amide-like C(=X)–Y bonds
# (X,Y in {N,O,S}) are excluded via one-sided constraints that SMARTS
# matching applies symmetrically over both atom orderings.
_ROT_BASE = (                       # filters applied to BOTH end-atoms
    '!$(*#*)'                       # not in a triple bond
    '&!D1'                          # not terminal
    '&!$(C(F)(F)F)'                 # not CF3
    '&!$(C(Cl)(Cl)Cl)'             # not CCl3
    '&!$(C(Br)(Br)Br)'             # not CBr3
    '&!$(C([CH3])([CH3])[CH3])'    # not t-butyl centre
    '&!$([CH3])'                    # not methyl
)
_ROT_AMIDE = (                      # extra filters on ONE end-atom
    '&!$([CD3](=[N,O,S])-!@[#7,O,S!D1])'   # amide-like C→heteroatom
    '&!$([#7,O,S!D1]-!@[CD3]=[N,O,S])'     # heteroatom→amide-like C
    '&!$([CD3](=[N+])-!@[#7!D1])'           # guanidinium-like C→N
    '&!$([#7!D1]-!@[CD3]=[N+])'             # N→guanidinium-like C
)
_ROTATABLE_BOND_SMARTS = Chem.MolFromSmarts(
    f'[{_ROT_BASE}{_ROT_AMIDE}]'
    '-,:;!@'                        # acyclic single / aromatic bond
    f'[{_ROT_BASE}]'
)

RWPE_DIM = 12
COORD_DIM = 3
EN_DIM = 1
GC_DIM = 1
CHIRAL_SIGN_DIM = 1
CHIRAL_VOL_DIM = 1
NODE_CONTINUOUS_DIM = (
    RWPE_DIM + COORD_DIM + EN_DIM + GC_DIM + CHIRAL_SIGN_DIM + CHIRAL_VOL_DIM
)
# Pauling electronegativity by atomic number; unlisted elements default to _CARBON_EN.
_PAULING_EN = {
    1: 2.20,
    3: 0.98,
    4: 1.57,
    5: 2.04,
    6: 2.55,
    7: 3.04,
    8: 3.44,
    9: 3.98,
    11: 0.93,
    12: 1.00,
    13: 1.61,
    14: 1.90,
    15: 2.19,
    16: 2.58,
    17: 3.16,
    19: 0.82,
    20: 1.00,
    21: 1.36,
    22: 1.54,
    23: 1.63,
    24: 1.66,
    25: 1.55,
    26: 1.83,
    27: 1.88,
    28: 1.91,
    29: 1.90,
    30: 1.65,
    31: 1.81,
    32: 2.01,
    33: 2.18,
    34: 2.55,
    35: 2.96,
}


def vocab_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index.
    """
    try:
        return l.index(e)
    except:  # noqa: BLE001
        return len(l) - 1


def _implicit_h_count(atom):
    non_active_hydrogen_count = 0
    for neighbor in atom.GetNeighbors():
        if neighbor.GetAtomicNum() != 1:
            continue

        if (
            len(neighbor.GetNeighbors()) == 1
            and neighbor.GetNeighbors()[0].GetAtomicNum() in _HBOND_DONOR_ATOMIC_NUMBERS
        ):
            continue
        non_active_hydrogen_count += 1

    return non_active_hydrogen_count


def _is_rotatable(bond, rotatable_bond_indices: set[int] | None = None):
    if rotatable_bond_indices is not None:
        return bond.GetIdx() in rotatable_bond_indices
    return False


def chiral_sign_from_cip(atom) -> float:
    """Return the canonical CIP signed-chirality scalar for an atom.

    ``R -> +1``, ``S -> -1``, anything else (no/undefined stereo) ``-> 0``.
    Read from RDKit's ``_CIPCode`` property, which is atom-numbering invariant
    and reproducible from SMILES at inference without any 3D conformer. Call
    ``Chem.AssignStereochemistry(mol, cleanIt=True, force=True)`` beforehand so
    the property is populated.
    """
    try:
        if atom.HasProp('_CIPCode'):
            code = atom.GetProp('_CIPCode')
            if code == 'R':
                return 1.0
            if code == 'S':
                return -1.0
    except Exception:  # noqa: BLE001
        return 0.0
    return 0.0


def signed_tetra_volume(center, neighbors_in_rank_order) -> float:
    """Signed tetrahedral volume (a reflection-antisymmetric pseudoscalar).

    Computes the scalar triple product ``u0 . (u1 x u2)`` of the **unit**
    vectors from ``center`` to the first three rank-ordered neighbor positions.
    The result lies in ``[-1, 1]``, flips sign under mirror reflection, and is
    ~0 for coplanar (achiral) arrangements. Returns ``0.0`` when fewer than
    three neighbors are available or any vector is degenerate.
    """
    import numpy as _np

    pts = [_np.asarray(p, dtype=_np.float64) for p in neighbors_in_rank_order[:3]]
    if len(pts) < 3:
        return 0.0
    c = _np.asarray(center, dtype=_np.float64)
    units = []
    for p in pts:
        v = p - c
        norm = _np.linalg.norm(v)
        if norm < 1e-6:
            return 0.0
        units.append(v / norm)
    vol = float(_np.dot(units[0], _np.cross(units[1], units[2])))
    return vol


def atom_features(atom):
    """
    Convert an RDKit atom object to a feature list of vocabulary indices.
    """
    return [
        vocab_index(FEATURE_VOCAB['possible_atomic_num_list'], atom.GetAtomicNum()),
        vocab_index(FEATURE_VOCAB['possible_chirality_list'], str(atom.GetChiralTag())),
        vocab_index(FEATURE_VOCAB['possible_degree_list'], atom.GetTotalDegree()),
        vocab_index(FEATURE_VOCAB['possible_formal_charge_list'], atom.GetFormalCharge()),
        vocab_index(FEATURE_VOCAB['possible_numH_list'], _implicit_h_count(atom)),
        vocab_index(FEATURE_VOCAB['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
        vocab_index(FEATURE_VOCAB['possible_hybridization_list'], str(atom.GetHybridization())),
        FEATURE_VOCAB['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
        FEATURE_VOCAB['possible_is_in_ring_list'].index(atom.IsInRing()),
        vocab_index(
            FEATURE_VOCAB['possible_ring_size_list'],
            atom.GetOwningMol().GetRingInfo().MinAtomRingSize(atom.GetIdx()),
        ),
    ]


def bond_features(bond, rotatable_bond_indices: set[int] | None = None):
    """
    Convert an RDKit bond object to a feature list of vocabulary indices.
    """
    return [
        vocab_index(FEATURE_VOCAB['possible_bond_type_list'], str(bond.GetBondType())),
        FEATURE_VOCAB['possible_bond_stereo_list'].index(str(bond.GetStereo())),
        FEATURE_VOCAB['possible_is_conjugated_list'].index(bond.GetIsConjugated()),
        FEATURE_VOCAB['possible_is_rotable_list'].index(
            _is_rotatable(bond, rotatable_bond_indices),
        ),
        vocab_index(
            FEATURE_VOCAB['possible_ring_size_list'],
            bond.GetOwningMol().GetRingInfo().MinBondRingSize(bond.GetIdx()),
        ),
    ]
