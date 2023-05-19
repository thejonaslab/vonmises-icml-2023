""" RDKit auxiliary functions. """
from enum import Enum
import itertools
import math

import matplotlib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from rdkit import Chem
from rdkit.Chem import rdchem, rdMolTransforms, rdForceFieldHelpers
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Lipinski import RotatableBondSmarts
from rdkit.ForceField import rdForceField
from rdkit.Geometry.rdGeometry import Point3D
from scipy.spatial import distance_matrix
from scipy.stats import gaussian_kde
import seaborn as sns
from skspatial.objects import Plane
from networkx.algorithms import isomorphism
import tinygraph as tg
import tinygraph.io.rdkit


def neighbor_search(mol: rdchem.Mol, tied_neighbors_dict, visited) -> Tuple:
    """
    Search through neighbors of neighbors to determine a priority index.

    :param mol: RDKit mol object.
    :param tied_neighbors_dict: Dictionary mapping tied neighbor to a dictionary of neighbors ordered by atomic number.
    :param visited: Atom indices that we have visited.
    :return: Information about the search.
    """
    tied_neighbors = list(tied_neighbors_dict.keys())
    neighbors_lists = list(tied_neighbors_dict.values())

    for i in range(len(tied_neighbors_dict[tied_neighbors[0]])):
        atomic_nums = []
        for d in neighbors_lists:
            atomic_nums.append(list(d.items())[i][1])
        atomic_nums = np.array(atomic_nums)
        max_atomic_num = np.max(atomic_nums)
        max_select = atomic_nums == max_atomic_num
        if sum(max_select) == 1:
            return True, tied_neighbors[np.argmax(atomic_nums)], None, None
        else:
            continue

    # Create new tied_neighbors_dict
    for i in range(len(tied_neighbors)):
        keys = list(neighbors_lists[i].keys())
        visited += keys

    tmp = dict()
    for i in range(len(tied_neighbors)):
        tmp2 = dict()
        keys = list(neighbors_lists[i].keys())
        for j in keys:
            if type(j) == int:
                for y in [n for n in mol.GetAtomWithIdx(int(j)).GetNeighbors() if n.GetIdx() not in visited]:
                    tmp2[y.GetIdx()] = y.GetAtomicNum()
        tmp[tied_neighbors[i]] = tmp2

    # Make the dictionaries all the same length with zero padding
    neighbor_lists = list(tmp.values())
    max_length = max([len(d) for d in neighbor_lists])

    # If there are no neighbors of neighbors, then just select the first tied neighbor
    if max_length == 0:
        return True, tied_neighbors[0], None, None

    # Make the dictionaries all the same length with zero padding
    for d in neighbor_lists:
        if len(d) < max_length:
            diff = max_length - len(d)
            for i in range(diff):
                d[f'null_{i}'] = 0

    # Sort the dictionaries in order of decreasing atomic number
    for i in range(len(neighbor_lists)):
        neighbor_lists[i] = {k: v for k, v in
                             sorted(neighbor_lists[i].items(), key=lambda item: item[1], reverse=True)}

    # Create a dictionary that maps tied neighbor index to neighbors dictionary
    tied_neighbors_dict = {}
    for i in range(len(tied_neighbors)):
        tied_neighbors_dict[tied_neighbors[i]] = neighbor_lists[i]

    return False, None, tied_neighbors_dict, visited


# noinspection PyUnresolvedReferences
def select_atom_priority_neighbor(mol: rdchem.Mol, neighbors: np.ndarray, source: int) -> int:
    """
    Select one neighbor of an atom in a consistent fashion using rules based on CIP ranking.

    :param mol: RDKit molecule.
    :param neighbors: Neighbor atomic indices to select from.
    :param source: Atom index for which we want to select a neighbor.
    :return: Index of selected neighbor.
    """
    # Compute the atomic numbers of the neighbors
    atomic_numbers = np.array([mol.GetAtomWithIdx(int(x)).GetAtomicNum() for x in neighbors])
    max_atomic_num = np.max(atomic_numbers)
    max_select = atomic_numbers == max_atomic_num

    # If there is a unique largest atomic number, select that neighbor
    if sum(max_select) == 1:
        return neighbors[np.argmax(atomic_numbers)]

    # Otherwise, perform a recursive procedure to select the neighbor
    else:
        # Set a boolean that indicates whether or not the atomic number tie has been broken yet
        tie_broken = False

        # Select the neighbors that are tied in terms of atomic number
        tied_neighbors = neighbors[max_select]

        # Keep track of which atom indexes we have examined so far
        visited = list(tied_neighbors) + [source]

        # For each tied neighbor, create a dictionary containing neighbors of neighbors, being careful to exclude
        # any visited indices
        neighbor_lists = []
        for x in tied_neighbors:
            tmp = dict()
            for y in [i for i in mol.GetAtomWithIdx(int(x)).GetNeighbors() if i.GetIdx() not in visited]:
                tmp[y.GetIdx()] = y.GetAtomicNum()
            neighbor_lists.append(tmp)

        # Check the maximum length of the dictionaries
        max_length = max([len(d) for d in neighbor_lists])

        # If there are no neighbors of neighbors, then just select the first tied neighbor
        if max_length == 0:
            return tied_neighbors[0]

        # Make the dictionaries all the same length with zero padding
        for d in neighbor_lists:
            if len(d) < max_length:
                diff = max_length - len(d)
                for i in range(diff):
                    d[f'null_{i}'] = 0

        # Sort the dictionaries in order of decreasing atomic number
        for i in range(len(neighbor_lists)):
            neighbor_lists[i] = {k: v for k, v in
                                 sorted(neighbor_lists[i].items(), key=lambda item: item[1], reverse=True)}

        # Create a dictionary that maps tied neighbor index to neighbors dictionary
        tied_neighbors_dict = {}
        for i in range(len(tied_neighbors)):
            tied_neighbors_dict[tied_neighbors[i]] = neighbor_lists[i]

        index_choice = None
        while not tie_broken:
            tie_broken, index_choice, tied_neighbors_dict, visited = neighbor_search(mol, tied_neighbors_dict, visited)

        return index_choice


# noinspection PyUnresolvedReferences
def select_dihedral_atom_indices(mol: rdchem.Mol, bond_index: int) -> Tuple[int, int, int, int]:
    """
    Compute the four atom indices defining an RDKit bond's torsion angle.

    :param mol: RDKit molecule.
    :param bond_index: Bond index.
    :return: List of atoms defining torsion angle.
    """
    bond = mol.GetBondWithIdx(bond_index)
    atom_a_index = bond.GetBeginAtomIdx()
    atom_b_index = bond.GetEndAtomIdx()
    atom_a_neighbors = mol.GetAtomWithIdx(atom_a_index).GetNeighbors()
    atom_a_neighbors = np.array([x.GetIdx() for x in atom_a_neighbors if x.GetIdx() != atom_b_index])
    atom_b_neighbors = mol.GetAtomWithIdx(atom_b_index).GetNeighbors()
    atom_b_neighbors = np.array([x.GetIdx() for x in atom_b_neighbors if x.GetIdx() != atom_a_index])

    # Check if the bond is in a ring
    if bond.IsInRing():
        # Get all ring tuples
        ring_info = list(mol.GetRingInfo().AtomRings())

        # Special case
        ring_info = [x for x in ring_info if atom_a_index in x and atom_b_index in x]

        if len(ring_info) == 2 and len(ring_info[0]) == 3 and len(ring_info[1]) == 3:
            # Extract atom a neighbors which are not also atom b neighbors
            atom_a_external_choices = [x for x in atom_a_neighbors if x not in atom_b_neighbors]
            if len(atom_a_external_choices) > 0:
                # If there is a valid choice from the list above, pick it as the neighbor
                atom_a_neighbor_index = atom_a_external_choices[0]
            else:
                # Otherwise, just select a priority neighbor
                atom_a_neighbor_index = select_atom_priority_neighbor(mol, atom_a_neighbors, atom_a_index)

            # Extract atom b neighbors which are not also atom a neighbors
            atom_b_external_choices = [x for x in atom_b_neighbors if x not in atom_a_neighbors]
            if len(atom_b_external_choices) > 0:
                # If there is a valid choice from the list above, pick it as the neighbor
                atom_b_neighbor_index = atom_b_external_choices[0]
            else:
                # Otherwise, select a priority neighbor, excluding the atom that was just selected as atom a's neighbor
                atom_b_neighbor_index = \
                    select_atom_priority_neighbor(mol,
                                                  np.array([x for x in atom_b_neighbors if
                                                            x != atom_a_neighbor_index]), atom_b_index)
        else:
            # Select the (first) ring containing both atom_a and atom_b
            ring_info = ring_info[0]

            if len(ring_info) > 3:
                # Pick neighboring atoms within this ring if the ring has more than 3 members
                atom_a_neighbor_index = [x for x in atom_a_neighbors if x in ring_info][0]
                atom_b_neighbor_index = [x for x in atom_b_neighbors if x in ring_info][0]
            else:
                # If the ring has 3 members...
                if len(atom_a_neighbors) > 1:
                    # If atom a has more than one neighbor not including atom b, then restrict the neighbor list to
                    # atoms outside of the ring; this avoids selecting the third member of the ring as the neighbor
                    # for both atom a and atom b
                    atom_a_neighbors = np.array([x for x in atom_a_neighbors if x not in ring_info])
                if len(atom_b_neighbors) > 1:
                    atom_b_neighbors = np.array([x for x in atom_b_neighbors if x not in ring_info])
                atom_a_neighbor_index = select_atom_priority_neighbor(mol, atom_a_neighbors, atom_a_index)
                atom_b_neighbor_index = select_atom_priority_neighbor(mol, atom_b_neighbors, atom_b_index)

        return int(atom_a_neighbor_index), int(atom_a_index), int(atom_b_index), int(atom_b_neighbor_index)

    # Select a neighbor for each bond atom in order to form a dihedral in the case that the bond is not in a ring
    atom_a_neighbor_index = select_atom_priority_neighbor(mol, atom_a_neighbors, atom_a_index)
    atom_b_neighbor_index = select_atom_priority_neighbor(mol, atom_b_neighbors, atom_b_index)

    return int(atom_a_neighbor_index), int(atom_a_index), int(atom_b_index), int(atom_b_neighbor_index)


# noinspection PyUnresolvedReferences
def compute_torsions_data_frame(mol: rdchem.Mol, bonds: np.ndarray) -> pd.DataFrame:
    """
    Compute torsion angles for a set of bonds defined by pairs of atoms.

    :param mol: RDKit mol object containing conformations.
    :param bonds: Bonds defined by begin and end atoms.
    :return: Dataframe.
    """
    atom_indices = []
    column_names = dict()
    for i, bond in enumerate(bonds):
        # Get atom indices for the ith bond
        atom_a_idx = int(bond[0])
        atom_b_idx = int(bond[1])
        atom_a_symbol = mol.GetAtomWithIdx(atom_a_idx).GetSymbol()
        atom_b_symbol = mol.GetAtomWithIdx(atom_b_idx).GetSymbol()

        b = mol.GetBondBetweenAtoms(int(bond[0]), int(bond[1])).GetIdx()
        atom_a_neighbor_index, atom_a_idx, atom_b_idx, atom_b_neighbor_index = select_dihedral_atom_indices(mol, b)

        atom_indices.append([atom_a_neighbor_index, atom_a_idx, atom_b_idx, atom_b_neighbor_index])
        column_names[i] = f'{bond[0]}-{bond[1]} | {atom_a_symbol} {atom_b_symbol}'

    results = None
    for i in range(len(bonds)):
        angles = []
        confs = mol.GetConformers()
        for c in confs:
            angles.append(rdMolTransforms.GetDihedralRad(c, atom_indices[i][0], atom_indices[i][1],
                                                         atom_indices[i][2], atom_indices[i][3]))
        angles = np.array(angles)
        if i == 0:
            results = angles[:, np.newaxis]
        else:
            results = np.concatenate((results, angles[:, np.newaxis]), axis=1)

    df = pd.DataFrame(results)
    df = df.rename(columns=column_names)

    return df


def compute_rotatable_bond_torsions(mol: rdchem.Mol) -> pd.DataFrame:
    """
    Compute torsion angles for rotatable bonds.

    :param mol: RDKit mol object containing conformations.
    :return: Dataframe.
    """
    rotatable_bonds = mol.GetSubstructMatches(RotatableBondSmarts)
    df = compute_torsions_data_frame(mol, np.array(rotatable_bonds))

    return df


def compute_aromatic_ring_bonds(mol: rdchem.Mol) -> List:
    """
    Compute aromatic ring bonds.

    :param mol: RDKit mol object containing conformations.
    :return: List of begin and end atom indices for the bonds.
    """
    rotatable_bonds = mol.GetSubstructMatches(RotatableBondSmarts)
    aromatic_bonds = []
    for bond in mol.GetBonds():
        if (bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()) not in rotatable_bonds and \
                (bond.GetEndAtom().GetIdx(), bond.GetBeginAtom().GetIdx()) not in rotatable_bonds:
            if bond.IsInRing():
                if bond.GetBeginAtom().GetIsAromatic() and bond.GetEndAtom().GetIsAromatic():
                    aromatic_bonds.append([bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()])
    return aromatic_bonds


def compute_aromatic_ring_bond_torsions(mol: rdchem.Mol) -> pd.DataFrame:
    """
    Compute torsion angles for aromatic ring bonds.

    :param mol: RDKit mol object containing conformations.
    :return: Dataframe.
    """
    bonds = compute_aromatic_ring_bonds(mol)
    df = compute_torsions_data_frame(mol, np.array(bonds))

    return df


def compute_non_aromatic_ring_bonds(mol: rdchem.Mol) -> List:
    """
    Compute non-aromatic ring bonds.

    :param mol: RDKit mol object containing conformations.
    :return: List of begin and end atom indices for the bonds.
    """
    rotatable_bonds = mol.GetSubstructMatches(RotatableBondSmarts)
    non_aromatic_ring_bonds = []
    for bond in mol.GetBonds():
        if (bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()) not in rotatable_bonds and \
                (bond.GetEndAtom().GetIdx(), bond.GetBeginAtom().GetIdx()) not in rotatable_bonds:
            if bond.IsInRing():
                if not bond.GetBeginAtom().GetIsAromatic() or not bond.GetEndAtom().GetIsAromatic():
                    non_aromatic_ring_bonds.append([bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()])
    return non_aromatic_ring_bonds


def compute_non_aromatic_ring_bond_torsions(mol: rdchem.Mol) -> pd.DataFrame:
    """
    Compute torsion angles for non-aromatic ring bonds.

    :param mol: RDKit mol object containing conformations.
    :return: Dataframe.
    """
    bonds = compute_non_aromatic_ring_bonds(mol)
    df = compute_torsions_data_frame(mol, np.array(bonds))

    return df


def compute_non_rotatable_non_ring_bonds(mol: rdchem.Mol) -> List:
    """
    Compute non-rotatable non-ring bonds.

    :param mol: RDKit mol object containing conformations.
    :return: List of begin and end atom indices for the bonds.
    """
    rotatable_bonds = mol.GetSubstructMatches(RotatableBondSmarts)
    non_rotatable_non_ring_bonds = []
    for bond in mol.GetBonds():
        if (bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()) not in rotatable_bonds and \
                (bond.GetEndAtom().GetIdx(), bond.GetBeginAtom().GetIdx()) not in rotatable_bonds:
            if not bond.IsInRing() and len(bond.GetBeginAtom().GetNeighbors()) > 1 and \
                    len(bond.GetEndAtom().GetNeighbors()) > 1:
                non_rotatable_non_ring_bonds.append([bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()])
    return non_rotatable_non_ring_bonds


def compute_non_rotatable_non_ring_bond_torsions(mol: rdchem.Mol) -> pd.DataFrame:
    """
    Compute torsion angles for non-rotatable non-ring bonds.

    :param mol: RDKit mol object containing conformations.
    :return: Dataframe.
    """
    bonds = compute_non_rotatable_non_ring_bonds(mol)
    df = compute_torsions_data_frame(mol, np.array(bonds))

    return df


def compute_angle_triplets(mol: rdchem.Mol) -> Tuple[List, Dict]:
    """
    Compute triples of atom indices that define all angles in a molecule.

    :param mol: RDKit mol object containing conformations.
    :return: Angle triplets list as well as column names for angles dataframe, used in compute_angles function.
    """
    angle_triplets = []
    column_names = dict()
    count = 0
    for a, b in itertools.combinations(mol.GetBonds(), 2):
        begin_a, end_a = a.GetBeginAtomIdx(), a.GetEndAtomIdx()
        begin_b, end_b = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        begin_a_symbol, end_a_symbol = mol.GetAtomWithIdx(begin_a).GetSymbol(), mol.GetAtomWithIdx(end_a).GetSymbol()
        begin_b_symbol, end_b_symbol = mol.GetAtomWithIdx(begin_b).GetSymbol(), mol.GetAtomWithIdx(end_b).GetSymbol()
        if begin_a == begin_b:
            angle_triplets.append((end_a, begin_a, end_b))
            column_names[count] = f'{end_a}-{begin_a}-{end_b} | {end_a_symbol}-{begin_a_symbol}-{end_b_symbol}'
            count += 1
        elif begin_a == end_b:
            angle_triplets.append((end_a, begin_a, begin_b))
            column_names[count] = f'{end_a}-{begin_a}-{begin_b} | {end_a_symbol}-{begin_a_symbol}-{begin_b_symbol}'
            count += 1
        elif end_a == begin_b:
            angle_triplets.append((begin_a, end_a, end_b))
            column_names[count] = f'{begin_a}-{end_a}-{end_b} | {begin_a_symbol}-{end_a_symbol}-{end_b_symbol}'
            count += 1
        elif end_a == end_b:
            angle_triplets.append((begin_a, end_a, begin_b))
            column_names[count] = f'{begin_a}-{end_a}-{begin_b} | {begin_a_symbol}-{end_a_symbol}-{begin_b_symbol}'
            count += 1

    return angle_triplets, column_names


def compute_bond_idxs(mol: rdchem.Mol, bond_list) -> List:
    """
    Compute bond idxs for pair of begin and end atom indices in a list.

    :param mol: RDKit mol object.
    :param bond_list: List of begin and end atom RDKit indices.
    :return: RDKit bond indices.
    """
    bond_idxs = []
    for bond in bond_list:
        atom_a_idx = int(bond[0])
        atom_b_idx = int(bond[1])
        bond_idx = mol.GetBondBetweenAtoms(atom_a_idx, atom_b_idx).GetIdx()
        bond_idxs.append(bond_idx)

    return bond_idxs


def plot_torsion_joint_histograms(df: pd.DataFrame, weights: np.ndarray = None, bin_width: float = 0.1,
                                  joint_hist_bw_adjust: float = 0.25, plot_diag: bool = True) -> \
        matplotlib.figure.Figure:
    """
    Plot pairwise joint histogram of all torsion distributions in the given DataFrame.

    :param df: DataFrame of torsion angles for a set of conformations and bonds (# conformations x # bonds).
    :param weights: Histogram weights.
    :param bin_width: Histogram bin width.
    :param joint_hist_bw_adjust: bw_adjust value for kernel density estimate in lower triangle of grid.
    :param plot_diag: Whether or not to plot diagonals.
    :return: Figure.
    """
    g = sns.PairGrid(df)
    g.set(ylim=(-math.pi - 1., math.pi + 1.), xlim=(-math.pi - 1., math.pi + 1.))
    if plot_diag:
        g.map_upper(sns.histplot, bins=list(np.arange(-math.pi - 1., math.pi + 1., bin_width)), weights=weights)
        g.map_lower(sns.kdeplot, fill=True, weights=weights, bw_adjust=joint_hist_bw_adjust)
    g.map_diag(sns.histplot, bins=list(np.arange(-math.pi - 1., math.pi + 1., bin_width)), weights=weights)

    return g.fig


def compute_num_torsion_modes(df: pd.DataFrame, shift: float = 0.1, bw_method: float = 0.1) -> pd.DataFrame:
    """
    Compute # torsion modes for a set of torsion distributions by counting maxima in a kernel density estimate.

    :param df: DataFrame containing torsion angles (# confs x # bonds).
    :param shift: Amount (radians) by which to do incremental modular shifts of the distribution.
    :param bw_method: Estimator bandwidth (kde.factor).
    :return: DataFrame of shape (# confs x 2). Column 0 is bond name, column 1 is mode count.
    """
    positions = np.arange(0.0, 2 * math.pi, shift)
    mode_counts = []
    for i in range(df.shape[1]):
        min_count = float('inf')
        for k in positions:
            count = 0

            # Compute the kernel estimate
            kernel = gaussian_kde((df.iloc[:, i].to_numpy() + math.pi + k) % (2 * math.pi), bw_method=bw_method)

            # Compute the kernel value at points between 0 and 2\pi
            Z = kernel(positions)

            # Compute the first derivative and its sign
            diff = np.gradient(Z)
            s_diff = np.sign(diff)

            # Locate zero crossings and check where the crossing corresponds to a local maximum of the kernel estimate
            zc = np.where(s_diff[:-1] != s_diff[1:])[0]
            for j in zc:
                if s_diff[:-1][j] == 1.0 and s_diff[1:][j] == -1.0:
                    count += 1

            # Record the smallest mode counts
            if count < min_count:
                min_count = count

        mode_counts.append([df.columns[i], min_count])

    df = pd.DataFrame(mode_counts)
    df = df.rename(columns={0: "Bond", 1: "Mode Count"})

    return df


def draw_mol(mol, output_path):
    """
    Draw a molecule as a PNG file.

    :param mol: RDKit mol object.
    :param output_path: File output path.
    """
    # noinspection PyUnresolvedReferences
    tmp = Chem.Mol(mol)
    tmp.RemoveAllConformers()  # Remove conformers in order to make molecule representation clearer in image
    d = rdMolDraw2D.MolDraw2DCairo(500, 500)
    # noinspection PyArgumentList
    d.drawOptions().addAtomIndices = True
    rdMolDraw2D.PrepareAndDrawMolecule(d, tmp)

    with open(output_path, 'wb') as f:
        # noinspection PyArgumentList
        f.write(d.GetDrawingText())


def reorder_mol_atoms_like_smiles(mol: rdchem.Mol) -> Tuple[rdchem.Mol, np.ndarray]:
    """
    Reorder the atoms in a molecule using the ordering from a SMILES parsing.

    :param mol: RDKit Mol object (with explicit H atoms).
    """
    mol = Chem.Mol(mol)
    g_mol = tg.io.rdkit.from_rdkit_mol(mol)
    nx_mol = tg.io.to_nx(g_mol, weight_prop='bond')

    clean_mol = Chem.AddHs(Chem.MolFromSmiles(Chem.MolToSmiles(mol)))
    g_clean_mol = tg.io.rdkit.from_rdkit_mol(clean_mol)
    nx_clean_mol = tg.io.to_nx(g_clean_mol, weight_prop='bond')       

    def node_match(a, b) -> bool:
        """
        Determine if nodes in a molecular graph match for isomorphism check.

        :param a: Molecular graph from tiny graph.
        :param b: Molecular graph from tiny graph.
        :return: Whether or not there is a match.
        """
        return a['atomicno'] == b['atomicno']

    def edge_match(a, b):
        """
        Determine if edges in a molecular graph match for isomorphism check.

        :param a: Molecular graph from tiny graph.
        :param b: Molecular graph from tiny graph.
        :return: Whether or not there is a match.
        """
        return a['bond'] == b['bond']

    GM = isomorphism.GraphMatcher(nx_mol, nx_clean_mol, node_match=node_match, edge_match=edge_match)
    assert GM.is_isomorphic()

    atom_mapping_array = np.zeros(mol.GetNumAtoms(), dtype=int)
    for k, v in GM.mapping.items():
        atom_mapping_array[v] = k

    mol_renumbered = Chem.rdmolops.RenumberAtoms(mol, [int(a) for a in atom_mapping_array])    
    return mol_renumbered, atom_mapping_array


def get_all_conf_pos(mol: rdchem.Mol) -> np.ndarray:
    """
    Create a numpy array of N confs x M atoms x 3 conformation positions.

    :param mol: RDKit mol object.
    :return: Array.
    """
    pos = np.stack([c.GetPositions() for c in mol.GetConformers()], axis=0)
    return pos


def get_conf_dist(pos) -> np.ndarray:
    """
    Create distance matrix from conformation position matrix.

    :param pos: Conformation position matrix.
    :return: Distance matrix.
    """
    dists = np.stack([distance_matrix(a, a) for a in pos], axis=0)
    return dists


def calc_energy_grad(pos: np.ndarray, force_field: rdForceField.ForceField) -> Tuple[float, np.ndarray]:
    """
    Compute MMFF energy and gradient.

    :param pos: Atomic coordinates.
    :param force_field: RDKit force field.
    :return: MMFF energy and gradient, where the energy is kcal/mol and the gradient is kcal/mol/Angstrom.
    """
    pos = tuple(pos.flatten())

    energy = force_field.CalcEnergy(pos)
    grad = force_field.CalcGrad(pos)
    grad = np.reshape(np.array(grad), [int(len(grad) / 3), 3])

    return energy, grad


def compute_angle(d1: float, d2: float, angle12: float) -> float:
    """
    Compute angle given bond lengths and 13 distance using Law of Cosines. This function uses soft cutoffs in order to
    compensate for numerical precision issues.

    :param d1: First bond length.
    :param d2: Second bond length.
    :param angle12: Angle distance.
    :return: Angle in radians.
    """
    argument = (angle12 ** 2 - d1 ** 2 - d2 ** 2) / (-2 * d1 * d2)
    if -1.1 <= argument < -1.0:
        argument = -1.0
    if 1.0 < argument <= 1.1:
        argument = 1.0
    if argument < -1.1 or argument > 1.1:
        print("Angle error! The input to math.acos is way outside of [-1, 1].")
        exit()
    result = math.acos(argument)
    return result


def count_lone_pairs(mol: rdchem.Mol, atom_idx: int) -> int:
    """
    Count the number of lone pairs on an atom.

    :param mol: RDKit mol object.
    :param atom_idx: Index of atom of interest.
    :return: Number of lone pairs
    """
    pt = Chem.GetPeriodicTable()
    atom = mol.GetAtomWithIdx(atom_idx)
    num_outer_electrons = pt.GetNOuterElecs(atom.GetAtomicNum())
    default_valence = pt.GetDefaultValence(atom.GetAtomicNum())
    formal_charge = atom.GetFormalCharge()

    num_lone_pair_electrons = num_outer_electrons - default_valence
    num_lone_pair_electrons = max(num_lone_pair_electrons - formal_charge, 0)

    num_lone_pairs = num_lone_pair_electrons // 2

    return num_lone_pairs


def compute_volume(conformer, indices: List[int]) -> int:
    """
    Compute oriented volume given four points.

    :return: +1 or -1 oriented volume.
    """
    p1 = conformer.GetPositions()[indices[0]]
    p2 = conformer.GetPositions()[indices[1]]
    p3 = conformer.GetPositions()[indices[2]]
    p4 = conformer.GetPositions()[indices[3]]
    matrix = np.array([[1, p1[0], p1[1], p1[2]],
                       [1, p2[0], p2[1], p2[2]],
                       [1, p3[0], p3[1], p3[2]],
                       [1, p4[0], p4[1], p4[2]]])

    return np.sign(np.linalg.det(matrix))


def select_atom_priority_indices_for_chirality(mol: rdchem.Mol, atom_index: int) -> List[int]:
    """
    Select atom indices in order of priority for computing chirality consistently.

    :param mol: RDKit mol object.
    :param atom_index: Index of atom to compute chirality for.
    :return: List of atom indices in order of priority.
    """
    priority_indices = []
    neighbors = [x.GetIdx() for x in mol.GetAtomWithIdx(atom_index).GetNeighbors()]
    while neighbors:
        priority_index = select_atom_priority_neighbor(mol, np.array(neighbors), atom_index)
        priority_indices.append(priority_index)
        neighbors.remove(priority_index)

    return priority_indices


def compute_chirality(mol: rdchem.Mol, conformer: rdchem.Conformer, atom_index: int) -> int:
    """
    Compute chirality of a single conformer in a consistent fashion.

    :param mol: RDKit mol object.
    :param conformer: RDKit conformer object.
    :param atom_index: Index of atom to compute chirality for.
    :return: Chirality (oriented volume value of +/- 1)
    """
    priority_indices = select_atom_priority_indices_for_chirality(mol, atom_index)
    if len(priority_indices) == 3:
        priority_indices.append(atom_index)

    return compute_volume(conformer, priority_indices)


def compute_chirality_all_confs(mol: rdchem.Mol, atom_index) -> List[int]:
    """
    Compute chirality for all conformers.

    :param mol: RDKit mol object.
    :param atom_index: Atom index.
    :return: List of chiralities (oriented volumes)
    """
    res = []
    for conf in mol.GetConformers():
        res.append(compute_chirality(mol, conf, atom_index))

    return res


def compute_chirality_probability_targets(mol: rdchem.Mol) -> Dict[str, np.ndarray]:
    """
    Compute chirality probability targets.

    :param mol: RDKit mol object containing conformations.
    :return: Dict of chiralities and atom indices.
    """
    atom_idxs = []
    out_chiralities = []
    bond_classes = classify_bonds_torsion_type(mol)

    for bond in mol.GetBonds():
        if bond_classes[bond.GetIdx()] == TorsionBondType.ROTATABLE.value:
            begin_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()
            if len(begin_atom.GetNeighbors()) == 3 and count_lone_pairs(mol, begin_atom.GetIdx()) == 1 and \
                    begin_atom.GetIdx() not in atom_idxs:
                atom_idxs.append(begin_atom.GetIdx())
                out_chiralities.append(compute_chirality_all_confs(mol, begin_atom.GetIdx()))
            if len(end_atom.GetNeighbors()) == 3 and count_lone_pairs(mol, end_atom.GetIdx()) == 1 and \
                    end_atom.GetIdx() not in atom_idxs:
                atom_idxs.append(end_atom.GetIdx())
                out_chiralities.append(compute_chirality_all_confs(mol, end_atom.GetIdx()))

    out_chiralities = np.array(out_chiralities, dtype=np.float32)

    return {'chiralities': out_chiralities,
            'atom_indices': np.array(atom_idxs).astype(np.int32)}


def compute_lengths(mol: rdchem.Mol) -> Dict:
    """
    Compute all bond lengths.

    :param mol: RDKit mol object containing conformations.
    :return: Numpy array and list of bond indices.
    """
    bond_idxs = []
    conf_n = mol.GetNumConformers()
    bond_n = mol.GetNumBonds()
    out_bond_lengths = np.zeros((bond_n, conf_n), dtype=np.float32)

    for i, bond in enumerate(mol.GetBonds()):
        bond_idx = bond.GetIdx()
        bond_idxs.append(bond_idx)
        bond_lengths = []
        confs = mol.GetConformers()
        for j in range(len(confs)):
            c = confs[j]
            bond_lengths.append(rdMolTransforms.GetBondLength(c,
                                                              bond.GetBeginAtom().GetIdx(),
                                                              bond.GetEndAtom().GetIdx()))
        out_bond_lengths[i] = np.array(bond_lengths)

    return {'lengths': out_bond_lengths,
            'bond_indices': np.array(bond_idxs).astype(np.int32)}


def compute_angles(mol: rdchem.Mol) -> Dict:
    """
    Compute all 1-3 bond angles.

    :param mol: RDKit mol object containing conformations.
    :return: Numpy array containing all 1-3 angles for each conformation as well as corresponding angle index triplets.
    """
    angle_triplets, column_names = compute_angle_triplets(mol)

    conf_n = mol.GetNumConformers()
    angles_n = len(angle_triplets)

    out_angles = np.zeros((angles_n, conf_n), dtype=np.float32)

    confs = list(mol.GetConformers())
    for i, angle_triplet in enumerate(angle_triplets):
        angles = []
        for j, c in enumerate(confs):
            angle = rdMolTransforms.GetAngleRad(c, angle_triplet[0], angle_triplet[1],
                                                angle_triplet[2])
            angles.append(angle)
        angles = np.array(angles)
        assert len(angles) > 0

        out_angles[i] = angles

    return {'angles': out_angles,
            'angle_triplets': np.array(angle_triplets).astype(np.int32)}


def compute_torsions_for_bond(mol: rdchem.Mol, bond: Tuple, restrict_to_first_conf: bool = False):
    """
    Compute torsion angles for all conformations for a given bond.

    :param mol: RDKit mol object.
    :param bond: Bond indices.
    :param restrict_to_first_conf: Whether or not to only use the first conformation in the mol object.
    :return: Bond index, torsions, and atoms defining the torsion angle.
    """
    b = mol.GetBondBetweenAtoms(int(bond[0]), int(bond[1])).GetIdx()
    atom_a_neighbor_index, atom_a_idx, atom_b_idx, atom_b_neighbor_index = select_dihedral_atom_indices(mol, b)

    angles = []
    confs = mol.GetConformers()
    for c in confs:
        angles.append(rdMolTransforms.GetDihedralRad(c,
                                                     atom_a_neighbor_index,
                                                     atom_a_idx, atom_b_idx,
                                                     atom_b_neighbor_index))
        if restrict_to_first_conf:
            break

    angles = np.array(angles)
    return {'bond_index': b,
            'angles': angles,
            'atom_index_tuple': (atom_a_neighbor_index,
                                 atom_a_idx, atom_b_idx,
                                 atom_b_neighbor_index)}


class TorsionBondType(Enum):
    """
    Class defining torsion bond types.
    """
    NOT_SET = -1
    ROTATABLE = 0
    AROMATIC_RING = 1
    NON_AROMATIC_RING = 2
    NON_AROMATIC_NON_RING = 3


def get_bond_sorted_atoms(bond) -> Tuple:
    """
    Get begin and end atoms for a bond in consistent fashion.

    :param bond: RDKit bond object.
    :return: Begin and end atom indices.
    """
    a = bond.GetBeginAtom().GetIdx()
    b = bond.GetEndAtom().GetIdx()

    return min(a, b), max(a, b)


def classify_bonds_torsion_type(mol: rdchem.Mol) -> np.ndarray:
    """
    Compute torsion bond types for each torsion bond in a molecule.

    :param mol: RDKit molecule object.
    :return: Array of types.
    """
    rotatable_bonds = mol.GetSubstructMatches(RotatableBondSmarts)
    aromatic_ring_bonds = compute_aromatic_ring_bonds(mol)
    non_aromatic_ring_bonds = compute_non_aromatic_ring_bonds(mol)
    non_rotatable_non_ring_bonds = compute_non_rotatable_non_ring_bonds(mol)

    out = np.ones(mol.GetNumBonds()) * int(TorsionBondType.NOT_SET.value)

    for bond_class, bonds in [(TorsionBondType.ROTATABLE, rotatable_bonds),
                              (TorsionBondType.AROMATIC_RING, aromatic_ring_bonds),
                              (TorsionBondType.NON_AROMATIC_RING, non_aromatic_ring_bonds),
                              (TorsionBondType.NON_AROMATIC_NON_RING, non_rotatable_non_ring_bonds)]:
        for bond in bonds:
            bond_idx = mol.GetBondBetweenAtoms(bond[0], bond[1]).GetIdx()
            assert out[bond_idx] == int(TorsionBondType.NOT_SET.value)
            out[bond_idx] = int(bond_class.value)
    return out


def compute_torsions(mol: rdchem.Mol, restrict_to_rotatable: bool = False, restrict_to_first_conf: bool = False) -> \
        Dict:
    """
    Compute torsion angles for torsion bonds and all conformations.

    :param mol: RDKit mol object.
    :param restrict_to_rotatable: Whether or not to only consider rotatable bond torsions.
    :param restrict_to_first_conf: Whether or not to only use the first conformation in the mol object.
    :return: Dictionary with information on torsion angles.
    """
    angles = []
    bond_indices = []
    atom_index_tuples = []
    bond_indices_with_chirality = []

    bond_classes = classify_bonds_torsion_type(mol)

    for bond in mol.GetBonds():
        if restrict_to_rotatable:
            if bond_classes[bond.GetIdx()] != TorsionBondType.ROTATABLE.value:
                continue

        elif bond_classes[bond.GetIdx()] == TorsionBondType.NOT_SET.value:
            continue

        result = compute_torsions_for_bond(mol, get_bond_sorted_atoms(bond), restrict_to_first_conf)
        angles.append(result['angles'])
        bond_indices.append(result['bond_index'])
        atom_index_tuples.append(result['atom_index_tuple'])

        chirality_indicator = 0
        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()
        if len(begin_atom.GetNeighbors()) == 3 and count_lone_pairs(mol, begin_atom.GetIdx()) == 1:
            chirality_indicator = 1
        elif len(end_atom.GetNeighbors()) == 3 and count_lone_pairs(mol, end_atom.GetIdx()) == 1:
            chirality_indicator = 1
        bond_indices_with_chirality.append(chirality_indicator)

    if len(angles) == 0:
        return {'angles': np.array([]),
                'bond_classes': np.array([]),
                'bond_indices': np.array([]),
                'atom_index_tuples': np.array([]),
                'bond_indices_with_chirality': np.array([])}

    angles = np.stack(angles).astype(np.float16)
    bond_classes = bond_classes.astype(np.int32)
    bond_indices = np.array(bond_indices).astype(np.int32)
    atom_index_tuples = np.stack(atom_index_tuples).astype(np.int32)
    bond_indices_with_chirality = np.array(bond_indices_with_chirality).astype(np.int32)

    subset_bond_classes = bond_classes[bond_indices]

    # sort everything by bond index
    sorted_idx = np.argsort(bond_indices)

    return {'angles': angles[sorted_idx],
            'bond_classes': subset_bond_classes[sorted_idx],
            'bond_indices': bond_indices[sorted_idx],
            'atom_index_tuples': atom_index_tuples[sorted_idx],
            'bond_indices_with_chirality': bond_indices_with_chirality[sorted_idx]}


def get_bond_index_paths(mol: rdchem.Mol) -> Dict:
    """
    For an atom pair (i, j) where i < j, compute the bond indices of the shortest path between them.

    :param mol: RDKit mol object.
    :return: Dictionary mapping atom pair tuples to list of bond indices.
    """
    N = mol.GetNumAtoms()

    g = tg.io.rdkit.from_rdkit_mol(mol)
    shortest_path_lens, next_mat = tg.algorithms.get_shortest_paths(g, False, True)
    shortest_paths = tg.algorithms.construct_all_shortest_paths(next_mat)

    # construct lut
    bond_idx_lut = np.ones((N, N), dtype=np.int32) * -1
    for b in mol.GetBonds():
        a1 = b.GetBeginAtom().GetIdx()
        a2 = b.GetEndAtom().GetIdx()
        a1, a2 = min(a1, a2), max(a1, a2)
        bond_idx_lut[a1, a2] = b.GetIdx()
        bond_idx_lut[a2, a1] = b.GetIdx()

    bond_idx_sps = {}
    for i in range(N):
        for j in range(i + 1, N):
            sp_len = shortest_path_lens[i, j]
            bond_idx_path = []
            for k in range(int(sp_len)):
                a1, a2 = int(shortest_paths[i, j, k]), int(shortest_paths[i, j, k + 1])
                bond_idx_path.append(bond_idx_lut[a1, a2])
            bond_idx_sps[(i, j)] = bond_idx_path

    return bond_idx_sps


def rdkit_get_bond_index_paths_slow(mol: rdchem.Mol):
    """
    For an atom pair (i, j) where i < j, compute the bond indices of the shortest path between them. Slow version.

    :param mol: RDKit mol object.
    :return: Dictionary mapping atom pair tuples to list of bond indices.
    """
    N = mol.GetNumAtoms()
    sps = {}
    for i in range(N):
        for j in range(i + 1, N):
            atom_sp = Chem.rdmolops.GetShortestPath(mol, i, j)
            bonds_sp = [mol.GetBondBetweenAtoms(atom_sp[k], atom_sp[k + 1]) for k in range(len(atom_sp) - 1)]
            sps[(i, j)] = [b.GetIdx() for b in bonds_sp]

    return sps


def flip_chirality(mol: rdchem.Mol, conformer_index, atom_index) -> None:
    """
    Flip the chirality of a chirality inversion atom.

    :param mol: RDKit molecule object.
    :param conformer_index: Index of conformer in list of RDKit conformers.
    :param atom_index: Atom index.
    """
    # Make sure that the atom has exactly three neighbors
    neighbors = mol.GetAtomWithIdx(atom_index).GetNeighbors()
    assert len(mol.GetAtomWithIdx(atom_index).GetNeighbors()) == 3

    conformer = mol.GetConformers()[conformer_index]

    original_chirality = compute_chirality(mol, conformer, atom_index)

    n1, n2, n3 = neighbors[0], neighbors[1], neighbors[2]

    # Compute the plane formed by N's three neighbors
    p1 = conformer.GetPositions()[n1.GetIdx()]
    p2 = conformer.GetPositions()[n2.GetIdx()]
    p3 = conformer.GetPositions()[n3.GetIdx()]

    plane = Plane.from_points(p1, p2, p3)

    # Compute the projection of N onto the plane
    N = conformer.GetPositions()[atom_index]
    point_projected = plane.project_point(N)

    # Compute the plane formed by N, one of its neighbors, and N's projection onto the neighbor plane
    plane_new = Plane.from_points(N, p3, point_projected)

    # Reflect all atoms across this plane
    for i in range(mol.GetNumAtoms()):
        point = conformer.GetPositions()[i]
        point_projected = np.array(plane_new.project_point(point))
        v = 2.0*(point_projected - point)
        new_point = point + v
        mol.GetConformers()[conformer_index].SetAtomPosition(i, Point3D(new_point[0], new_point[1], new_point[2]))

    new_chirality = compute_chirality(mol, conformer, atom_index)

    assert (int(original_chirality) != int(new_chirality))


def filter_conformation(mol: rdchem.Mol, conf_idx: int) -> bool:
    """
    Determine whether a conformation should be filtered based on the average of van Der Waals radii.

    :param mol: Molecule object.
    :param conf_idx: Conformation index to check for filtering.
    :return: Boolean indicating filtering.
    """
    filter_c = False
    pt = Chem.GetPeriodicTable()
    dist_mat = Chem.rdmolops.Get3DDistanceMatrix(mol, confId=conf_idx)
    path_length = Chem.rdmolops.GetDistanceMatrix(mol)
    upper_triangle_indices = np.triu_indices_from(dist_mat, k=1)
    for i in range(upper_triangle_indices[0].shape[0]):
        atom_idx_1 = int(upper_triangle_indices[0][i])
        atom_idx_2 = int(upper_triangle_indices[1][i])
        # Only perform distance check for molecules separated by more than 5 bonds
        if path_length[atom_idx_1, atom_idx_2] > 5:
            atom_1_vdw = pt.GetRvdw(mol.GetAtomWithIdx(atom_idx_1).GetAtomicNum())
            atom_2_vdw = pt.GetRvdw(mol.GetAtomWithIdx(atom_idx_2).GetAtomicNum())
            # Set the lower bound based on vDw distances
            lower_bound = (atom_1_vdw + atom_2_vdw) / 2
            distance = dist_mat[atom_idx_1, atom_idx_2]
            if distance < lower_bound:
                filter_c = True
                break
    return filter_c
