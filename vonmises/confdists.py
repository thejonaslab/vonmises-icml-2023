""" Conformation distributions and utilities for evaluation. """
from typing import Dict

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdchem, rdmolops
from rdkit.Chem.Lipinski import RotatableBondSmarts

from vonmises import mol_utils


class SampleDistribution:
    """
    A distribution of conformations with metadata.
    """

    def __init__(self, mol: rdchem.Mol, meta: Dict = None):
        """
        :param mol: RDKit mol object.
        :param meta: Metadata dictionary.
        """
        self.mol = Chem.Mol(mol)
        self.meta = {} if meta is None else meta

    def get_dist_mats(self) -> np.ndarray:
        """
        Get distance matrix from conformations.

        :return: Distance matrix.
        """
        return mol_utils.get_conf_dist(mol_utils.get_all_conf_pos(self.mol))


def get_atom_pair_restrict_to_vm_preds(mol, atom_index_a: int, atom_index_b: int, shortest_paths: Dict) -> bool:
    """
    Determine if distances between a particular pair of atoms are included in VM predictions.

    :param mol: RDKit mol object.
    :param atom_index_a: First atom index.
    :param atom_index_b: Second atom index.
    :param shortest_paths: Shortest paths dictionary.
    :return: Boolean indicating inclusion.
    """
    add = True
    rotatable_bonds = [mol.GetBondBetweenAtoms(x[0], x[1]).GetIdx() for x in
                       mol.GetSubstructMatches(RotatableBondSmarts)]
    shortest_path = shortest_paths[(atom_index_a, atom_index_b)]
    if len(shortest_path) == 1:
        if mol.GetBondWithIdx(int(shortest_path[0])).IsInRing():
            add = False
    elif len(shortest_path) == 2:
        if shortest_path[0] not in rotatable_bonds and shortest_path[1] not in rotatable_bonds:
            add = False
    else:
        for i in range(1, len(shortest_path) - 1):
            if shortest_path[i] not in rotatable_bonds:
                add = False
                break

    return add


def get_atom_pair_restrict_to_non_ring(mol, atom_index_a: int, atom_index_b: int, shortest_paths: Dict) -> bool:
    """
    Determine if distances between a particular pair of atoms with non-ring restriction.

    :param mol: RDKit mol object.
    :param atom_index_a: First atom index.
    :param atom_index_b: Second atom index.
    :param shortest_paths: Shortest paths dictionary.
    :return: Boolean indicating inclusion.
    """
    add = True
    shortest_path = shortest_paths[(atom_index_a, atom_index_b)]
    if len(shortest_path) == 1:
        if mol.GetBondWithIdx(int(shortest_path[0])).IsInRing():
            add = False
    elif len(shortest_path) == 2:
        if mol.GetBondWithIdx(int(shortest_path[0])).IsInRing() and \
                mol.GetBondWithIdx(int(shortest_path[1])).IsInRing():
            add = False
    else:
        for i in range(1, len(shortest_path) - 1):
            if mol.GetBondWithIdx(int(shortest_path[i])).IsInRing():
                add = False
                break

    return add


def get_atom_pair_restrict_to_non_aromatic_ring(mol, atom_index_a: int, atom_index_b: int,
                                                shortest_paths: Dict) -> bool:
    """
    Determine if distances between a particular pair of atoms with non-aromatic ring restriction.

    :param mol: RDKit mol object.
    :param atom_index_a: First atom index.
    :param atom_index_b: Second atom index.
    :param shortest_paths: Shortest paths dictionary.
    :return: Boolean indicating inclusion.
    """
    non_aromatic_ring_bonds = [mol.GetBondBetweenAtoms(x[0], x[1]).GetIdx() for x in
                               mol_utils.compute_non_aromatic_ring_bonds(mol)]
    shortest_path = shortest_paths[(atom_index_a, atom_index_b)]
    add = True
    if len(shortest_path) == 1:
        if shortest_path[0] in non_aromatic_ring_bonds:
            add = False
    elif len(shortest_path) == 2:
        if shortest_path[0] in non_aromatic_ring_bonds and shortest_path[1] in non_aromatic_ring_bonds:
            add = False
    else:
        for i in range(1, len(shortest_path) - 1):
            if shortest_path[i] in non_aromatic_ring_bonds:
                add = False
    return add


def compute_dihedral_hist(dist: SampleDistribution, bins=32, filler_value: float = 1e-10) -> Dict:
    """
    Compute histograms of rotatable bond torsion angles.

    :param dist: SampleDistribution object with conformations.
    :param bins: Number of histogram bins to use.
    :param filler_value: Non-zero value to support histogram computation.
    :return: Dictionary mapping bond index to histogram.
    """
    torsion_targets = mol_utils.compute_torsions(dist.mol, True)
    angles, bond_indices = torsion_targets['angles'], torsion_targets['bond_indices']

    theta_bins = np.linspace(-np.pi, np.pi, bins + 1)

    out_dict = {}
    for i, v in zip(bond_indices, angles):
        assert np.max(v) <= np.pi
        assert np.min(v) >= -np.pi
        hist = np.histogram(v, bins=theta_bins)[0]
        hist = hist / np.sum(hist)
        hist = np.where(hist < filler_value, filler_value, hist)
        out_dict[int(i)] = hist
    return out_dict


def compute_dihedral_list(dist: SampleDistribution) -> Dict:
    """
    Compute sets of rotatable bond torsion angles.

    :param dist: SampleDistribution object with conformations.
    :return: Dictionary mapping bond index to list of torsion angles.
    """
    torsion_targets = mol_utils.compute_torsions(dist.mol, True)
    angles, bond_indices = torsion_targets['angles'], torsion_targets['bond_indices']

    out_dict = {}
    for i, v in zip(bond_indices, angles):
        assert np.max(v) <= np.pi
        assert np.min(v) >= -np.pi
        out_dict[int(i)] = v
    return out_dict


def compute_distance_avg_path_lim(dist: SampleDistribution, min_atom_path_len: int = 2, max_atom_path_len: int = 2,
                                  restrict_to_vm_preds: bool = False, restrict_to_non_ring: bool = False,
                                  restrict_to_non_aromatic_ring: bool = False) -> Dict:
    """
    Compute average distances between pairs of atoms whose topological distance is within given bounds.

    :param dist: SampleDistribution object with conformations.
    :param min_atom_path_len: Minimum topological pairwise distance between atoms, measured in # atoms along the path.
    :param max_atom_path_len: Maximum topological pairwise distance between atoms, measured in # atoms along the path.
    :param restrict_to_vm_preds: Whether or not to restrict computations to VonMisesNet predictions.
    :param restrict_to_non_ring: Whether or not to restrict computations to non-ring components.
    :param restrict_to_non_aromatic_ring: Whether or not to restrict computations to exclude non-aromatic ring.
    :return: Dictionary mapping from atom indices and path length tuple to average pairwise distance.
    """
    # Topological path lengths dictionary, measured in # bonds along the path (equivalent to # atoms + 1)
    path_length = Chem.rdmolops.GetDistanceMatrix(dist.mol)

    # Shortest paths (in bond indices) between pairs of atoms dictionary
    shortest_paths = mol_utils.get_bond_index_paths(dist.mol)

    # Compute average distance matrix
    dm = dist.get_dist_mats()
    mean_dist = dm.mean(axis=0)

    out = {}
    for i in range(mean_dist.shape[0]):
        for j in range(i + 1, mean_dist.shape[1]):
            if min_atom_path_len <= path_length[i, j] + 1 <= max_atom_path_len:
                add = True
                if restrict_to_vm_preds:
                    add = get_atom_pair_restrict_to_vm_preds(dist.mol, i, j, shortest_paths)
                elif restrict_to_non_ring:
                    add = get_atom_pair_restrict_to_non_ring(dist.mol, i, j, shortest_paths)
                elif restrict_to_non_aromatic_ring:
                    add = get_atom_pair_restrict_to_non_aromatic_ring(dist.mol, i, j, shortest_paths)
                if add:
                    out[(i, j, int(path_length[i, j]))] = mean_dist[i, j]

    return out


def compute_distance_lists(dist: SampleDistribution, min_atom_path_len: int = 4, max_atom_path_len: int = 4,
                           restrict_to_vm_preds: bool = False, restrict_to_non_ring: bool = False,
                           restrict_to_non_aromatic_ring: bool = False):
    """
    Compute all distances between pairs of atoms whose topological distance is within given bounds.

    :param dist: SampleDistribution object with conformation.
    :param min_atom_path_len: Minimum topological pairwise distance between atoms, measured in # atoms along the path.
    :param max_atom_path_len: Maximum topological pairwise distance between atoms, measured in # atoms along the path.
    :param restrict_to_vm_preds: Whether or not to restrict computations to VonMisesNet predictions.
    :param restrict_to_non_ring: Whether or not to restrict computations to non-ring components.
    :param restrict_to_non_aromatic_ring: Whether or not to restrict computations to exclude non-aromatic ring.
    :return: Dictionary mapping from atom indices and path length tuple to list of pairwise distance.
    """
    # Topological path lengths dictionary, measured in # bonds along the path (equivalent to # atoms + 1)
    path_length = Chem.rdmolops.GetDistanceMatrix(dist.mol)

    # Shortest paths (in bond indices) between pairs of atoms dictionary
    shortest_paths = mol_utils.get_bond_index_paths(dist.mol)

    dm = dist.get_dist_mats()
    out = {}
    for i in range(dm[0].shape[0]):
        for j in range(i + 1, dm[0].shape[1]):
            if min_atom_path_len <= path_length[i, j] + 1 <= max_atom_path_len:
                add = True
                if restrict_to_vm_preds:
                    add = get_atom_pair_restrict_to_vm_preds(dist.mol, i, j, shortest_paths)
                elif restrict_to_non_ring:
                    add = get_atom_pair_restrict_to_non_ring(dist.mol, i, j, shortest_paths)
                elif restrict_to_non_aromatic_ring:
                    add = get_atom_pair_restrict_to_non_aromatic_ring(dist.mol, i, j, shortest_paths)
                if add:
                    distances = []
                    for k in range(dm.shape[0]):
                        distances.append(dm[k][i, j])
                    out[(i, j, int(path_length[i, j]))] = distances
    return out
