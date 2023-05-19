""" Functions to featurize molecules. """
from typing import Dict, List

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdchem, rdForceFieldHelpers, rdmolops, rdPartialCharges
from rdkit.Chem.Lipinski import RotatableBondSmarts
from rdkit.Chem.rdDistGeom import GetMoleculeBoundsMatrix

from vonmises.mol_utils import compute_angle
from vonmises import etkdg


def to_one_hot(x: int, vals: List) -> List:
    """
    Return a one-hot vector.

    :param x: Data integer.
    :param vals: List of possible data values.
    :return: One-hot vector as list.
    """
    return [x == v for v in vals]


class FeaturizeMol:
    """ 
    Generate molecular featurization.
    """
    def __init__(self, atom_feat_config: Dict, bond_feat_config: Dict, preprocess_config: Dict):
        """
        :param atom_feat_config: Dictionary of atom parameters.
        :param bond_feat_config: Dictionary of bond parameters.
        :param preprocess_config: Dictionary of preprocess parameters
        """
        self.featurize_bonds = FeaturizeBonds(**bond_feat_config)
        self.featurize_atoms = FeaturizeAtoms(**atom_feat_config)
        self.preprocess_mol = PreprocessMol(**preprocess_config)
        self.featurize_angles = FeaturizeAngles()

    def __call__(self, mol: rdchem.Mol, angle_triplets: np.ndarray) -> Dict:
        """
        Get features for a molecule.

        :param mol: RDKit mol object.
        :param angle_triplets: Angle triplet info from targets.
        """
        num_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()

        mol = self.preprocess_mol(mol)

        rotatable_bonds = mol.GetSubstructMatches(RotatableBondSmarts)

        atom_features = self.featurize_atoms(mol, num_atoms)
        bond_features = self.featurize_bonds(mol, rotatable_bonds)
        angle_features = self.featurize_angles(mol, angle_triplets)

        features = {
            'angle_features': angle_features,
            'atom_features': atom_features,
            'num_atoms': num_atoms,
            'bond_features': bond_features,
            'bond_idx': mol.GetBonds(),
            'num_bonds': num_bonds,
            'rotatable_bonds': rotatable_bonds
        }

        return features


class FeaturizeAtoms:
    """
    Generate the atom features for a molecule.
    """
    def __init__(self,
                 mmff_atom_types_one_hot: bool = True,
                 assign_stereo: bool = True,
                 atomic_num: bool = True,
                 atom_types: List = None,
                 valence: bool = True,
                 valence_types: List = None,
                 aromatic: bool = True,
                 hybridization: bool = True,
                 hybridization_types: List = None,
                 partial_charge: bool = True,
                 formal_charge: bool = True,
                 charge_types: List = None,
                 r_covalent: bool = True,
                 r_vanderwals: bool = True,
                 default_valence: bool = True,
                 max_ring_size: int = 8,
                 rings: bool = True,
                 chirality: bool = True,
                 chi_types: List = None,
                 mmff94_atom_types: List = None,
                 degree_types: List = None,
                 degree: bool = True,
                 num_hydrogen: bool = True,
                 num_hydrogen_types: bool = None,
                 num_radical_electron: bool = True,
                 num_radical_electron_types: bool = None):
        """
        :param mmff_atom_types_one_hot: Whether or not to include MMFF94 atom types as vertex features.
        :param assign_stereo: Whether or not to include stereochemistry information.
        :param atomic_num: Whether or not to include atomic number as a vertex feature.
        :param atom_types: List of allowed atomic numbers.
        :param valence: Whether or not to include total valence as a vertex feature.
        :param valence_types: List of allowed total valence numbers.
        :param aromatic: Whether or not to include aromaticity as a vertex feature.
        :param hybridization: Whether or not to include hybridization as a vertex feature.
        :param hybridization_types: Hybridization types.
        :param partial_charge: Whether or not to include Gasteiger Charge as a vertex feature.
        :param formal_charge: Whether or not to include formal charge as a vertex feature.
        :param charge_types: Formal charge types.
        :param r_covalent: Whether or not to include covalent radius as a vertex feature.
        :param r_vanderwals: Whether or not to include vanderwals radius as a vertex feature.
        :param default_valence: Whether or not to include default valence as a vertex feature.
        :param max_ring_size: Maximum ring size.
        :param rings: Whether or not to include ring size as a vertex feature.
        :param chirality: Whether or not to include chirality as a vertex feature.
        :param chi_types: Chiral tag types.
        :param mmff94_atom_types: MMFF94 atom types.
        :param degree_types: Atomic degree types.
        :param degree: Whether or not to include degree as a vertex feature.
        :param num_hydrogen: Whether or not to include number of (neighboring) H atoms as a vertex feature.
        :param num_hydrogen_types: List of allowed number of H atoms (including neighbors).
        :param num_radical_electron: Whether or not to include number of radical electrons as a vertex feature.
        :param num_radical_electron_types: List of allowed number of radical electrons.
        """
        if num_radical_electron_types is None:
            num_radical_electron_types = [0, 1, 2]
        if num_hydrogen_types is None:
            num_hydrogen_types = [0, 1, 2, 3]
        if degree_types is None:
            degree_types = [1, 2, 3, 4]
        if chi_types is None:
            chi_types = list(rdchem.ChiralType.values.values())
        if mmff94_atom_types is None:
            mmff94_atom_types = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26,
                                 27, 28, 29, 30, 31, 32, 33, 37, 38, 39, 40, 42, 43, 44, 46, 48, 59, 62, 63, 64,
                                 65, 66, 70, 71, 72, 74, 75, 78]
        if charge_types is None:
            charge_types = [-1, 0, 1]
        if hybridization_types is None:
            hybridization_types = [Chem.HybridizationType.S,
                                   Chem.HybridizationType.SP,
                                   Chem.HybridizationType.SP2,
                                   Chem.HybridizationType.SP3,
                                   Chem.HybridizationType.SP3D,
                                   Chem.HybridizationType.SP3D2,
                                   Chem.HybridizationType.UNSPECIFIED]
        if valence_types is None:
            valence_types = [1, 2, 3, 4, 5, 6]
        if atom_types is None:
            atom_types = [1, 6, 7, 8, 9]
        self.partial_charge = partial_charge
        self.mmff_atom_types_one_hot = mmff_atom_types_one_hot
        self.assign_stereo = assign_stereo
        self.atomic_num = atomic_num
        self.atom_types = atom_types
        self.valence_types = valence_types
        self.valence = valence
        self.aromatic = aromatic
        self.hybridization_types = hybridization_types
        self.hybridization = hybridization
        self.partial_charge = partial_charge
        self.charge_types = charge_types
        self.formal_charge = formal_charge
        self.r_covalent = r_covalent
        self.r_vanderwals = r_vanderwals
        self.default_valence = default_valence
        self.max_ring_size = max_ring_size
        self.rings = rings
        self.chi_types = chi_types
        self.chirality = chirality
        self.mmff94_atom_types = mmff94_atom_types
        self.degree_types = degree_types
        self.degree = degree
        self.num_hydrogen_types = num_hydrogen_types
        self.num_hydrogen = num_hydrogen
        self.num_radical_electron_types = num_radical_electron_types
        self.num_radical_electron = num_radical_electron

    def __call__(self, mol: rdchem.Mol, num_atoms: int) -> List:
        """
        Get features for each atom in a molecule.

        :param mol: RDKit mol object.
        :param num_atoms: Number of atoms in the molecule.
        :return: List of atom features.
        """
        atom_features = []

        pt = Chem.GetPeriodicTable()

        if self.partial_charge:
            rdPartialCharges.ComputeGasteigerCharges(mol)

        mmff_p = None
        if self.mmff_atom_types_one_hot:
            # Make a copy of mol since this function can modify the aromaticity model
            mmff_p = rdForceFieldHelpers.MMFFGetMoleculeProperties(Chem.Mol(mol))

        if self.assign_stereo:
            rdmolops.AssignStereochemistryFrom3D(mol)

        for i in range(num_atoms):
            atom = mol.GetAtomWithIdx(i)
            atom_feature = []

            if self.atomic_num:
                atom_feature += to_one_hot(atom.GetAtomicNum(), self.atom_types)

            if self.valence:
                atom_feature += to_one_hot(atom.GetTotalValence(), self.valence_types)

            if self.aromatic:
                atom_feature += [atom.GetIsAromatic()]

            if self.hybridization:
                atom_feature += to_one_hot(atom.GetHybridization(), self.hybridization_types)

            if self.partial_charge:
                gc = float(atom.GetProp('_GasteigerCharge'))
                if not np.isfinite(gc):
                    gc = 0.0
                atom_feature += [gc]

            if self.formal_charge:
                atom_feature += to_one_hot(atom.GetFormalCharge(), self.charge_types)

            if self.r_covalent:
                atom_feature += [pt.GetRcovalent(atom.GetAtomicNum())]

            if self.r_vanderwals:
                atom_feature += [pt.GetRvdw(atom.GetAtomicNum())]

            if self.default_valence:
                atom_feature += to_one_hot(pt.GetDefaultValence(atom.GetAtomicNum()), self.valence_types)

            if self.rings:
                atom_feature += [atom.IsInRingSize(r) for r in range(3, self.max_ring_size + 1)]

            if self.chirality:
                atom_feature += to_one_hot(atom.GetChiralTag(), self.chi_types)

            if self.mmff_atom_types_one_hot:
                if mmff_p is None:
                    atom_feature += [0] * len(self.mmff94_atom_types)
                else:
                    atom_feature += to_one_hot(mmff_p.GetMMFFAtomType(i), self.mmff94_atom_types)

            if self.degree:
                atom_feature += to_one_hot(atom.GetDegree(), self.degree_types)

            if self.num_hydrogen:
                atom_feature += to_one_hot(atom.GetTotalNumHs(), self.num_hydrogen_types)

            if self.num_radical_electron:
                atom_feature += to_one_hot(atom.GetNumRadicalElectrons(), self.num_radical_electron_types)

            atom_features.append(atom_feature)
        return atom_features


class FeaturizeAngles:
    """
    Generate the bond angle features for a molecule.
    """

    def __init__(self):
        # If choices about featurization of angles emerge, initialize here.
        pass

    def __call__(self, mol: rdchem.Mol, angle_triplets: np.ndarray) -> List:
        """
        Get features for each angle in a molecule.

        :param mol: RDKit mol object.
        :param angle_triplets: Triplets of atom indices defining bond angles.
        :return: List of bond angle features.
        """
        bounds_matrix = GetMoleculeBoundsMatrix(mol)
        angle_features = []
        for i, (a, b, c) in enumerate(angle_triplets):
            d12 = (bounds_matrix[a][b] + bounds_matrix[b][a]) / 2.
            d23 = (bounds_matrix[b][c] + bounds_matrix[c][b]) / 2.
            d123 = (bounds_matrix[a][c] + bounds_matrix[c][a]) / 2.
            angle = compute_angle(d12, d23, d123)

            angle_feature = [angle]
            angle_features.append(angle_feature)

        return angle_features


class FeaturizeBonds:
    """
    Generate the bond features for a molecule.
    """

    def __init__(self,
                 bond_types: List = None,
                 bond_type: bool = True,
                 conjugated: bool = True,
                 bond_ring: bool = True,
                 bond_stereo: bool = True,
                 bond_stereo_types: List = None,
                 shortest_path: bool = True,
                 max_path_length: int = 10,
                 same_ring: bool = True,
                 rot_bond: bool = True):
        """
        :param bond_types: List of allowed bond types.
        :param bond_type: Whether or not to include bond type as an edge feature.
        :param conjugated: Whether or not to include conjugated as an edge feature.
        :param bond_ring: Whether or not to include bond being in ring as an edge feature.
        :param bond_stereo: Whether or not to include bond stereo as an edge feature.
        :param bond_stereo_types: List of bond stereo types.
        :param shortest_path: Whether or not to include shortest path length as a bond feature.
        :param max_path_length: Maximum shortest path length between any two atoms in a molecule in the dataset.
        :param same_ring: Whether or not to include same ring as bond feature.
        :param rot_bond: Whether or not a bond is rotatable.
        """
        if bond_stereo_types is None:
            bond_stereo_types = list(rdchem.BondStereo.values.values())
        if bond_types is None:
            bond_types = [0., 1., 1.5, 2., 3.]
        self.bond_types = bond_types
        self.bond_type = bond_type
        self.conjugated = conjugated
        self.bond_ring = bond_ring
        self.bond_stereo = bond_stereo
        self.bond_stereo_types = bond_stereo_types
        self.shortest_path = shortest_path
        self.max_path_length = max_path_length
        self.same_ring = same_ring
        self.rot_bond = rot_bond

    def __call__(self, mol: rdchem.Mol, rotatable_bonds: List = None) -> List:
        """
        Get features for each bond in a molecule.

        :param mol: RDKit mol object.
        :param rotatable_bonds: list of rotatable bonds in the molecule.
        :return: Bond features.
        """
        if rotatable_bonds is None:
            rotatable_bonds = []
        bond_features = []
        ring_info = list(mol.GetRingInfo().AtomRings())
        distances = rdmolops.GetDistanceMatrix(mol)
        for bond in mol.GetBonds():
            bond_feature = []
            a = bond.GetBeginAtom().GetIdx()
            b = bond.GetEndAtom().GetIdx()
            if self.bond_type:
                bond_feature += to_one_hot(bond.GetBondTypeAsDouble(), self.bond_types)

            if self.conjugated:
                bond_feature += [bond.GetIsConjugated()]

            if self.bond_ring:
                bond_feature += [bond.IsInRing()]

            if self.bond_stereo:
                bond_feature += to_one_hot(bond.GetStereo(), self.bond_stereo_types)

            if self.shortest_path:
                bond_feature += to_one_hot(distances[a, b] - 1, list(range(self.max_path_length)))

            if self.same_ring:
                membership = [int(a) in r and int(b) in r for r in ring_info]
                if sum(membership) > 0:
                    bond_feature += [1]
                else:
                    bond_feature += [0]

            if self.rot_bond:
                if (a, b) in rotatable_bonds or (b, a) in rotatable_bonds:
                    bond_feature += [1]
                else:
                    bond_feature += [0]

            bond_features.append(bond_feature)

        return bond_features


class PreprocessMol:
    """
    Any mol clean-up/sanity check that we want as part of the featurization process.
    """
    def __init__(self,
                 force_chi_to_tet: bool = True,
                 force_stereo_to_double: bool = True,
                 conform_to_existing_conf_idx: int = -1):
        self.force_chi_to_tet = force_chi_to_tet
        self.force_stereo_to_double = force_stereo_to_double
        self.conform_to_existing_conf_idx = conform_to_existing_conf_idx

    def __call__(self, mol):
        """
        Preprocess molecule. 
        """
        mol = Chem.Mol(mol)
        
        if self.conform_to_existing_conf_idx >= 0:
            assert mol.GetNumConformers() > 0
            etkdg.set_chiral_for_all_unset_atoms_based_on_pos(mol, self.conform_to_existing_conf_idx)
            etkdg.set_stereo_for_double_bonds_based_on_pos(mol, self.conform_to_existing_conf_idx)

        else:
            if self.force_chi_to_tet:
                etkdg.force_chiral_for_atoms_with_4_neighbors(mol)
            if self.force_stereo_to_double:
                etkdg.force_stereo_for_double_bonds(mol)

        return mol
