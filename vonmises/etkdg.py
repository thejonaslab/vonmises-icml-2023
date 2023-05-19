"""
We have identified several places where etkdg does not maintain atomic identities. These are our workarounds.
"""
from typing import Dict, List

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdchem, AllChem

from vonmises import mol_utils


def copy_confs(conf_source_mol: rdchem.Mol, target_mol: rdchem.Mol) -> rdchem.Mol:
    """
    Copy conformations to a new mol object. This provides a way to transfer conformations in the event that the
    conformation-generation modifies properties of the input molecule.

    :param conf_source_mol: Source RDKit mol object.
    :param target_mol: Output RDKit mol object.
    :return: New mol object.
    """
    out_mol = Chem.Mol(target_mol)
    out_mol.RemoveAllConformers()
    for c in conf_source_mol.GetConformers():
        out_mol.AddConformer(Chem.Conformer(c))
    return out_mol

        
def generate_etkdg_confs(orig_mol: rdchem.Mol, num: int = 100, seed: int = -1, max_embed_attempts: int = 0) -> \
        rdchem.Mol:
    """
    ETKDG wrapper.

    :param orig_mol: RDKit mol object.
    :param num: Number of conformations to generate.
    :param seed: Random seed.
    :param max_embed_attempts: Number of times to re-try failed ETKDG embedding before giving up.
    :return: New mol object with generated conformations.
    """
    mol = Chem.Mol(orig_mol)
    AllChem.EmbedMultipleConfs(mol, numConfs=num, randomSeed=seed, maxAttempts=max_embed_attempts)

    return copy_confs(mol, orig_mol)


def get_double_bonds(mol: rdchem.Mol) -> List:
    """
    Get double bonds of a molecule.

    :param mol: RDKit mol object.
    :return: List of double bonds.
    """
    return [b for b in mol.GetBonds() if b.GetBondTypeAsDouble() == 2.0]


def get_atom_det(p: np.ndarray) -> np.array:
    """
    For the 3 atoms in the position matrix, return the determinant.

    :param p: Position matrix of 3 atoms.
    :return: Determinant.
    """
    p = np.concatenate([p, np.ones((4, 1))], axis=1)
    return np.linalg.det(p.T)


def get_det_values_for_conf(mol: rdchem.Mol, conf_idx: int = 0) -> Dict:
    """
    Get the determinant for all atoms with 4 neighbors in a molecule for a specified conformation.

    :param mol: RDKit mol object.
    :param conf_idx: Conformation index.
    :return: Dictionary mapping from atom index to determinant.
    """
    pos = mol_utils.get_all_conf_pos(mol)
    tgt_pos = pos[conf_idx]
    out = {}
    
    for a in mol.GetAtoms():
        n = a.GetNeighbors()
        if len(n) == 4:
            atom_idx = [b.GetIdx() for b in n]
            out[a.GetIdx()] = get_atom_det(tgt_pos[atom_idx])
    return out


def force_stereo_for_double_bonds(mol: rdchem.Mol) -> None:
    """
    Force a fixed stereo for double bonds in the molecule that do not already have stereo set. This mutates the
    input molecule.

    :param mol: RDKit mol object.
    """
    for b in get_double_bonds(mol):
        if b.GetStereo() == Chem.BondStereo.STEREONONE:
            for begin_stereo in b.GetBeginAtom().GetNeighbors():
                if begin_stereo.GetIdx() != b.GetEndAtom().GetIdx():
                    break
            
            for end_stereo in b.GetEndAtom().GetNeighbors():
                if end_stereo.GetIdx() != b.GetBeginAtom().GetIdx():
                    break
            
            b.SetStereoAtoms(int(begin_stereo.GetIdx()), int(end_stereo.GetIdx()))
            b.SetStereo(Chem.BondStereo.STEREOCIS)
    return None


def force_chiral_for_atoms_with_4_neighbors(mol: rdchem.Mol) -> None:
    """
    If a molecule has 4 neighbors and does not have chirality set, force it.

    :param mol: RDKit mol object.
    """
    for a in mol.GetAtoms():
        if len(a.GetNeighbors()) == 4:
            if a.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED:
                a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
    
    return None


def set_chiral_for_all_unset_atoms_based_on_pos(mol: rdchem.Mol, conf_idx: int = 0) -> None:
    """
    For all atoms that don't currently have a chirality tag set, set it based on the geometry in a way that future
    runs of ETKDG will generate the same chirality. This mutates the input molecule.

    :param mol: RDKit mol object.
    :param conf_idx: Conformation index
    """
    dv = get_det_values_for_conf(mol, conf_idx)

    for atom_idx, det_val in dv.items():
        if det_val > 0:
            mol.GetAtomWithIdx(atom_idx).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
        else:
            mol.GetAtomWithIdx(atom_idx).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)    

    return None


def set_stereo_for_double_bonds_based_on_pos(mol: rdchem.Mol, conf_idx: int = 0) -> None:
    """
    For all bonds that don't currently have a stereochem tag set, set it based on the geometry in a way that future
    runs of ETKDG will generate the same geometry. This mutates the input molecule.

    :param mol: RDKit mol object.
    :param conf_idx: Conformation index.
    """
    dm = AllChem.Get3DDistanceMatrix(mol, confId = mol.GetConformers()[conf_idx].GetId())

    double_bonds = get_double_bonds(mol)

    for b in double_bonds:
        if not ((b.GetStereo() == Chem.BondStereo.STEREONONE )
                or (b.GetStereo() == Chem.BondStereo.STEREOANY )):
            continue

        bond_atom_1 = b.GetBeginAtom()
        bond_atom_2 = b.GetEndAtom()

        if len(bond_atom_1.GetNeighbors()) == 1 or \
           len(bond_atom_2.GetNeighbors()) == 1:
            continue
        
        swapped_12 = False
        
        if len(bond_atom_1.GetNeighbors()) > len(bond_atom_2.GetNeighbors()):
            bond_atom_1, bond_atom_2 = bond_atom_2, bond_atom_1
            swapped_12 = True
        assert len(bond_atom_1.GetNeighbors()) <= len(bond_atom_2.GetNeighbors())

        atom_1 = [n1 for n1 in bond_atom_1.GetNeighbors() if
                  n1.GetIdx() != bond_atom_2.GetIdx()][0]
        
        atom_2s = [a for a in bond_atom_2.GetNeighbors() \
                   if a.GetIdx() != bond_atom_1.GetIdx()]

        if len(atom_2s) == 1:
            continue
        
        atom_2_1_d = dm[atom_1.GetIdx(), atom_2s[0].GetIdx() ] 
        atom_2_2_d = dm[atom_1.GetIdx(), atom_2s[1].GetIdx() ]
        if atom_2_1_d < atom_2_2_d :
            atom_2 = atom_2s[0]
        else:
            atom_2 = atom_2s[1]

        if swapped_12:
            b.SetStereoAtoms(int(atom_2.GetIdx()), int(atom_1.GetIdx()))
        
        else:
            b.SetStereoAtoms(int(atom_1.GetIdx()), int(atom_2.GetIdx()))
        b.SetStereo(Chem.BondStereo.STEREOCIS)

    return None
    

def generate_clean_etkdg_confs(orig_mol: rdchem.Mol,
                               num: int = 100,
                               assign_chi_to_tet: bool = True,
                               assign_stereo_to_double: bool = True,
                               seed: int = -1,
                               conform_to_existing_conf_idx: int = -1,
                               max_embed_attempts: int = 0,
                               exception_for_num_failure: bool = True) -> rdchem.Mol:
    """
    Generate ETKDG tags in a way that preserves chirality and double bond stereochemistry.
    Return a new molecule with the generated conformations, having removed the old ones.

    :param orig_mol: RDKit mol object.
    :param num: Number of conformations to generate.
    :param assign_chi_to_tet: Whether or not to preserve chirality.
    :param assign_stereo_to_double: Whether or not to preserve double bond stereochemistry.
    :param seed: Random seed.
    :param conform_to_existing_conf_idx: If >= 0, the chirality of the existing conformation at this index will be used.
    :param max_embed_attempts: Maximum number of ETKDG embeddings to try before giving up.
    :param exception_for_num_failure: Raise an exception if fail to generate requested number of conformations.
    :return: New RDKit mol object.
    """
    mol = Chem.Mol(orig_mol)

    if conform_to_existing_conf_idx >= 0:
        assert mol.GetNumConformers() > 0
        set_chiral_for_all_unset_atoms_based_on_pos(mol, conform_to_existing_conf_idx)
        set_stereo_for_double_bonds_based_on_pos(mol, conform_to_existing_conf_idx)
    else:
        if assign_chi_to_tet:
            force_chiral_for_atoms_with_4_neighbors(mol)
        if assign_stereo_to_double:
            force_stereo_for_double_bonds(mol)

    AllChem.EmbedMultipleConfs(mol, numConfs=num, randomSeed=seed, maxAttempts=max_embed_attempts)

    if exception_for_num_failure:
        if mol.GetNumConformers() < num:
            raise Exception(f"Requested {num} conformers but only generated {mol.GetNumConformers()}")

    return copy_confs(mol, orig_mol)
