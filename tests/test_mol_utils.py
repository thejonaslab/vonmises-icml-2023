""" Test functions for mol_utils.py. """
import os
import time

import torch
torch.manual_seed(0)
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import pytest

from vonmises.mol_utils import *

DIRNAME = os.path.dirname(os.path.abspath(__file__))


def test_draw_mol() -> None:
    """
    Test creating an image of a molecule. Make sure that the image file is created.
    """
    # noinspection PyUnresolvedReferences
    mol = Chem.Mol(open(f'{DIRNAME}/butane.bin', "rb").read())
    file_name = f'draw_mol_test.{int(time.time()) % 100000000}{os.getpid() % 10000}.png'
    draw_mol(mol, file_name)
    assert (os.path.exists(file_name))
    os.remove(file_name)


def test_compute_num_torsion_modes() -> None:
    """
    Test computing number of torsion modes for torsions in a molecule. We will make sure that 10K ETKDG conformations
    results in the C-C bond of ethane having 3 modes.
    """
    # noinspection PyUnresolvedReferences
    mol = Chem.Mol(open(f'{DIRNAME}/ethane.bin', "rb").read())
    df = compute_rotatable_bond_torsions(mol)
    df2 = compute_num_torsion_modes(df)
    assert (df2.iloc[0, 1] == 3)


def test_plot_torsion_joint_histograms() -> None:
    """
    Test plot torsion joint histograms.
    """
    # noinspection PyUnresolvedReferences
    mol = Chem.Mol(open(f'{DIRNAME}/ethane.bin', "rb").read())
    df = compute_rotatable_bond_torsions(mol)
    g = plot_torsion_joint_histograms(df)
    file_name = f'plot_torsion_joint_histograms_test.{int(time.time()) % 100000000}{os.getpid() % 10000}.png'
    g.savefig(file_name)
    assert (os.path.exists(file_name))
    os.remove(file_name)


def test_compute_bond_idxs() -> None:
    """
    Test computing bond indices.
    """
    # noinspection PyUnresolvedReferences
    mol = Chem.Mol(open(f'{DIRNAME}/butane.bin', "rb").read())
    rotatable_bonds = mol.GetSubstructMatches(RotatableBondSmarts)
    assert (compute_bond_idxs(mol, rotatable_bonds) == [0, 1, 2])


def test_compute_non_rotatable_non_ring_bond_torsions() -> None:
    """
    Test computing non-rotatable non-ring bond torsions.
    """
    # noinspection PyUnresolvedReferences
    mol = Chem.Mol(open(f'{DIRNAME}/nmrshiftdb_example.bin', "rb").read())
    df = compute_non_rotatable_non_ring_bond_torsions(mol)
    assert (df.shape == (10, 1))


def test_compute_non_rotatable_non_ring_bonds() -> None:
    """
    Test computing non-rotatable non-ring bonds.
    """
    # noinspection PyUnresolvedReferences
    mol = Chem.Mol(open(f'{DIRNAME}/nmrshiftdb_example.bin', "rb").read())
    res = compute_non_rotatable_non_ring_bonds(mol)
    assert (res == [[3, 4]])


def test_compute_non_aromatic_ring_bond_torsions() -> None:
    """
    Test computing non-aromatic ring bond torsions.
    """
    # noinspection PyUnresolvedReferences
    mol = Chem.Mol(open(f'{DIRNAME}/cholesterol.bin', "rb").read())
    df = compute_non_aromatic_ring_bond_torsions(mol)
    assert (df.shape == (10, 20))


def test_compute_non_aromatic_ring_bonds() -> None:
    """
    Test computing non-aromatic ring bonds.
    """
    # noinspection PyUnresolvedReferences
    mol = Chem.Mol(open(f'{DIRNAME}/cholesterol.bin', "rb").read())
    res = compute_non_aromatic_ring_bonds(mol)
    assert (res == [[8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18],
                    [18, 19], [19, 20], [20, 21], [21, 22], [22, 23], [23, 24], [12, 8], [16, 11], [20, 15], [24, 19]])


def test_compute_aromatic_ring_bond_torsions() -> None:
    """
    Test computing aromatic ring bond torsions.
    """
    # noinspection PyUnresolvedReferences
    mol = Chem.Mol(open(f'{DIRNAME}/ibuprofen.bin', "rb").read())
    df = compute_aromatic_ring_bond_torsions(mol)
    assert (df.shape == (10, 6))


def test_compute_aromatic_ring_bonds() -> None:
    """
    Test computing aromatic ring bonds.
    """
    # noinspection PyUnresolvedReferences
    mol = Chem.Mol(open(f'{DIRNAME}/ibuprofen.bin', "rb").read())
    res = compute_aromatic_ring_bonds(mol)
    assert (res == [[4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 4]])


def test_compute_rotatable_bond_torsions() -> None:
    """
    Test computing rotatable bond torsions.
    """
    # noinspection PyUnresolvedReferences
    mol = Chem.Mol(open(f'{DIRNAME}/ibuprofen.bin', "rb").read())
    df = compute_rotatable_bond_torsions(mol)
    assert (df.shape == (10, 8))


@pytest.mark.xfail(reason="'correct' answer generated from previous output with bug")
def test_compute_angle_triplets() -> None:
    """
    Test compute angle triplets.
    """
    # noinspection PyUnresolvedReferences
    mol = Chem.Mol(open(f'{DIRNAME}/ibuprofen.bin', "rb").read())
    res = compute_angle_triplets(mol)
    assert (res[0] == [(0, 1, 2), (0, 1, 3), (1, 0, 15), (1, 0, 16), (1, 0, 17), (0, 1, 18), (2, 1, 3), (2, 1, 18),
                       (1, 2, 19), (1, 2, 20), (1, 2, 21), (1, 3, 4), (3, 1, 18), (1, 3, 22), (1, 3, 23), (3, 4, 5),
                       (3, 4, 9), (4, 3, 22), (4, 3, 23), (4, 5, 6), (5, 4, 9), (4, 5, 24), (5, 6, 7), (6, 5, 24),
                       (5, 6, 25), (6, 7, 8), (6, 7, 10), (7, 6, 25), (7, 8, 9), (8, 7, 10), (7, 8, 26), (8, 9, 4),
                       (9, 8, 26), (8, 9, 27), (7, 10, 11), (7, 10, 12), (7, 10, 28), (11, 10, 12), (11, 10, 28),
                       (10, 11, 29), (10, 11, 30), (10, 11, 31), (10, 12, 13), (10, 12, 14), (12, 10, 28),
                       (13, 12, 14), (12, 14, 32), (4, 9, 27), (15, 0, 16), (15, 0, 17), (16, 0, 17), (19, 2, 20),
                       (19, 2, 21), (20, 2, 21), (22, 3, 23), (29, 11, 30), (29, 11, 31), (30, 11, 31)])
    assert (res[1] == {0: '0-1-2 | C-C-C', 1: '0-1-3 | C-C-C', 2: '1-0-15 | C-C-H', 3: '1-0-16 | C-C-H',
                       4: '1-0-17 | C-C-H', 5: '0-1-18 | C-C-C', 6: '2-1-3 | C-C-C', 7: '2-1-18 | C-C-H',
                       8: '1-2-19 | C-C-C', 9: '1-2-20 | C-C-C', 10: '1-2-21 | C-C-C', 11: '1-3-4 | C-C-C',
                       12: '3-1-18 | C-C-H', 13: '1-3-22 | C-C-C', 14: '1-3-23 | C-C-C', 15: '3-4-5 | C-C-C',
                       16: '3-4-9 | C-C-C', 17: '4-3-22 | C-C-H', 18: '4-3-23 | C-C-H', 19: '4-5-6 | C-C-C',
                       20: '5-4-9 | C-C-C', 21: '4-5-24 | C-C-C', 22: '5-6-7 | C-C-C', 23: '6-5-24 | C-C-H',
                       24: '5-6-25 | C-C-C', 25: '6-7-8 | C-C-C', 26: '6-7-10 | C-C-C', 27: '7-6-25 | C-C-H',
                       28: '7-8-9 | C-C-C', 29: '8-7-10 | C-C-C', 30: '7-8-26 | C-C-C', 31: '8-9-4 | C-C-C',
                       32: '9-8-26 | C-C-H', 33: '8-9-27 | C-C-C', 34: '7-10-11 | C-C-C', 35: '7-10-12 | C-C-C',
                       36: '7-10-28 | C-C-C', 37: '11-10-12 | C-C-C', 38: '11-10-28 | C-C-H', 39: '10-11-29 | C-C-C',
                       40: '10-11-30 | C-C-C', 41: '10-11-31 | C-C-C', 42: '10-12-13 | C-C-C', 43: '10-12-14 | C-C-C',
                       44: '12-10-28 | C-C-H', 45: '13-12-14 | O-C-O', 46: '12-14-32 | C-O-O', 47: '4-9-27 | C-C-H',
                       48: '15-0-16 | H-C-H', 49: '15-0-17 | H-C-H', 50: '16-0-17 | H-C-H', 51: '19-2-20 | H-C-H',
                       52: '19-2-21 | H-C-H', 53: '20-2-21 | H-C-H', 54: '22-3-23 | H-C-H', 55: '29-11-30 | H-C-H',
                       56: '29-11-31 | H-C-H', 57: '30-11-31 | H-C-H'})


def test_compute_torsions() -> None:
    """
    Test computing torsions.
    """
    # noinspection PyUnresolvedReferences
    mol = Chem.Mol(open(f'{DIRNAME}/ibuprofen.bin', "rb").read())
    res = compute_torsions_data_frame(mol, np.array([[0, 1], [1, 2], [1, 3]]))
    assert (res.shape == (10, 3))


def test_select_dihedral_atom_indices() -> None:
    """
    Test select dihedral atom indices.
    """
    # noinspection PyUnresolvedReferences
    mol = Chem.Mol(open(f'{DIRNAME}/ibuprofen.bin', "rb").read())

    # Case: rotatable bonds
    assert (select_dihedral_atom_indices(mol, 13) == (13, 12, 14, 32))
    assert (select_dihedral_atom_indices(mol, 3) == (1, 3, 4, 5))

    # Case: not exactly two shared rings, and first shared ring has more than 3 members
    assert (select_dihedral_atom_indices(mol, 4) == (9, 4, 5, 6))

    # Case: not exactly two shared rings, first shared ring has 3 members, atom a has only 1 neighbor that is not atom
    # b, atom b has more than 1 neighbor that is not atom a
    # noinspection PyUnresolvedReferences
    mol = Chem.Mol(open(f'{DIRNAME}/mol_33334.bin', "rb").read())
    assert (select_dihedral_atom_indices(mol, 0) == (4, 0, 2, 3))

    # Case: not exactly two shared rings, first shared ring has 3 members, atom a has more than 1 neighbor that is not
    # atom b, atom b has only 1 neighbor that is not atom a
    # noinspection PyUnresolvedReferences
    mol = Chem.Mol(open(f'{DIRNAME}/mol_1136.bin', "rb").read())
    assert (select_dihedral_atom_indices(mol, 13) == (9, 10, 23, 11))

    # Case: not exactly two shared rings, first shared ring has 3 members, atom a has more than 1 neighbor that is not
    # atom b, atom b has more than 1 neighbor that is not atom a
    # noinspection PyUnresolvedReferences
    mol = Chem.Mol(open(f'{DIRNAME}/mol_13183.bin', "rb").read())
    assert (select_dihedral_atom_indices(mol, 0) == (10, 0, 16, 2))

    # Case: exactly two shared rings that each are 3-membered, both atom a and atom b have non-zero external choices
    # noinspection PyUnresolvedReferences
    mol = Chem.Mol(open(f'{DIRNAME}/mol_2346.bin', "rb").read())
    assert (select_dihedral_atom_indices(mol, 2) == (7, 0, 6, 10))

    # Case: exactly two shared rings that are each 3-membered, atom a has non-zero external choices but atom b does not
    # noinspection PyUnresolvedReferences
    mol = Chem.Mol(open(f'{DIRNAME}/mol_25788.bin', "rb").read())
    assert (select_dihedral_atom_indices(mol, 0) == (6, 1, 0, 2))


def test_select_atom_priority_neighbor() -> None:
    """
    Test selecting priority neighbor.
    """
    # noinspection PyUnresolvedReferences
    mol = Chem.Mol(open(f'{DIRNAME}/ibuprofen.bin', "rb").read())

    # Case: there is an immediate neighbor which has priority (highest atomic number)
    atom_idx = 0
    neighbors = np.array([x.GetIdx() for x in mol.GetAtomWithIdx(atom_idx).GetNeighbors()])
    assert (select_atom_priority_neighbor(mol, neighbors, atom_idx) == 1)

    atom_idx = 15
    neighbors = np.array([x.GetIdx() for x in mol.GetAtomWithIdx(atom_idx).GetNeighbors()])
    assert (select_atom_priority_neighbor(mol, neighbors, atom_idx) == 0)

    atom_idx = 3
    neighbors = np.array([x.GetIdx() for x in mol.GetAtomWithIdx(atom_idx).GetNeighbors()])
    assert (select_atom_priority_neighbor(mol, neighbors, atom_idx) == 1)

    atom_idx = 4
    neighbors = np.array([x.GetIdx() for x in mol.GetAtomWithIdx(atom_idx).GetNeighbors()])
    assert (select_atom_priority_neighbor(mol, neighbors, atom_idx) == 3)

    # noinspection PyUnresolvedReferences
    mol = Chem.Mol(open(f'{DIRNAME}/cholesterol.bin', "rb").read())

    atom_idx = 20
    neighbors = np.array([x.GetIdx() for x in mol.GetAtomWithIdx(atom_idx).GetNeighbors()])
    assert (select_atom_priority_neighbor(mol, neighbors, atom_idx) == 15)


def test_compute_angle() -> None:
    """
    Test compute angle.
    """
    angle = compute_angle(1, 1, 1)
    assert (1.04 < angle < 1.05)

    angle = compute_angle(1, 1, 2)
    assert (3.14 < angle < 3.15)

    angle = compute_angle(1, 1, 2.01)
    assert (3.14 < angle < 3.15)

    with pytest.raises(SystemExit) as pytest_wrapped_e:
        compute_angle(1, 1, 2.1)
    assert pytest_wrapped_e.type == SystemExit


def test_reorder_mol_atoms_like_smiles() -> None:
    """
    Simple test for reordering mol atoms
    """

    # two round trips to get canonical smiles ordering
    init_mol = Chem.AddHs(Chem.MolFromSmiles("CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O"))
    init_mol = Chem.AddHs(Chem.MolFromSmiles(Chem.MolToSmiles(init_mol)))

    ordering = np.random.permutation(init_mol.GetNumAtoms())
    init_mol_perm = Chem.rdmolops.RenumberAtoms(init_mol,
                                               [int(a) for a in ordering])

    # init atom element order
    orig_elt = [a.GetSymbol() for a in init_mol.GetAtoms()]
    perm_elt = [a.GetSymbol() for a in init_mol_perm.GetAtoms()]

    assert orig_elt != perm_elt

    reordered_mol, reorder = reorder_mol_atoms_like_smiles(init_mol_perm)
    reordered_elt = [a.GetSymbol() for a in reordered_mol.GetAtoms()]
    assert reordered_elt == orig_elt
    
    
def validate_path(rdmol, start_atom, end_atom, bond_idx_path):
    
    cur_atom = start_atom
    for bond_idx in bond_idx_path:
        b = rdmol.GetBondWithIdx(int(bond_idx))
        a1, a2 = b.GetBeginAtom().GetIdx(), b.GetEndAtom().GetIdx()
        if a1 == cur_atom:
            next_atom = a2
        elif a2 == cur_atom:
            next_atom = a1
        else:
            return False
        cur_atom = next_atom

    if cur_atom != end_atom:
        return False
    return True

def test_get_bond_index_paths():
    """

    """


    smiles1 = "CCCCC"
    smiles2 = "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O"
    smiles3 = 'CC1=C2[C@@]([C@]([C@H]([C@@H]3[C@]4([C@H](OC4)C[C@@H]([C@]3(C(=O)[C@@H]2OC(=O)C)C)O)OC(=O)C)OC(=O)c5ccccc5)(C[C@@H]1OC(=O)[C@H](O)[C@@H](NC(=O)c6ccccc6)c7ccccc7)O)(C)C'


    for s in [smiles1, smiles2, smiles3]:
        
        rdmol = Chem.AddHs(Chem.MolFromSmiles(s))


        ref_bond_idx = rdkit_get_bond_index_paths_slow(rdmol)
        fast_bond_idx = get_bond_index_paths(rdmol)
        
        
        assert len(ref_bond_idx) == len(fast_bond_idx)        
    



        N = rdmol.GetNumAtoms()
        for i in range(N):
            for j in range(i+1, N):
                #print(i, j, ref_bond_idx[i, j], fast_bond_idx[i, j])

                if not np.allclose(ref_bond_idx[i, j], fast_bond_idx[i, j]):
                    # possible multiple paths? 
                    assert len(ref_bond_idx[i, j]) == len(fast_bond_idx[i, j])
                    assert validate_path(rdmol, i, j, ref_bond_idx[i, j])
                    assert validate_path(rdmol, i, j, fast_bond_idx[i, j])        
