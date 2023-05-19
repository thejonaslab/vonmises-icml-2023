""" Generate independence analysis. """
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Lipinski import RotatableBondSmarts
import scipy
from sqlalchemy import select
from tqdm import tqdm

from generate_distributions_pipeline import get_molid_random_subset
from vonmises import mol_utils
from vonmises.sqlalchemy_model import Molecule, create_session

MOL_DB_FILENAME = "data/mol-data/nmrshiftdb-pt-conf-mols.db"
MAX_MOL = 5000
MAX_NUM_ATOMS = 64
SMALLEST_ELEMENT_SUBSET = "HCONFSPCl"
MORGAN_DIGITS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
RANKS = [1, 2, 3]
OUTPUT_FILENAME = "results/independence_analysis.pkl"
BINS = 32


def is_methyl(mol, atom_index) -> bool:
    """
    Check whether atom belongs to a methyl group.

    :param mol: RDKit mol object.
    :param atom_index: Atom index.
    :return: Boolean.
    """
    if mol.GetAtomWithIdx(atom_index).GetSymbol() == "C":
        neighbors = mol.GetAtomWithIdx(atom_index).GetNeighbors()
        neighbor_types = [n.GetSymbol() for n in neighbors]
        if len([n for n in neighbor_types if n == "H"]) == 3:
            return True
    return False


def get_svd(mol, rank: int = 2, bins: int = 32, mol_id: int = 0) -> List[Dict]:
    """
    Get SVD reconstruction error analysis.

    :param mol: RDKit mol object.
    :param rank: Rank of reconstruction.
    :param bins: Numbero f histogram bins.
    :param mol_id: Mol ID.
    :return:
    """
    assert rank > 0, "Rank must be > 0"

    path_length = Chem.rdmolops.GetDistanceMatrix(mol)
    rotatable_bonds = mol.GetSubstructMatches(RotatableBondSmarts)
    theta_bins = np.linspace(-np.pi, np.pi, bins + 1)
    df = mol_utils.compute_rotatable_bond_torsions(mol)
    out = []
    for i in range(df.shape[1]):
        for j in range(i + 1, df.shape[1]):
            hist = np.histogram2d(df.iloc[:, i], df.iloc[:, j], bins=theta_bins)[0]
            U, sigma, VT = scipy.linalg.svd(hist)
            for r in range(rank):
                approx_inc = np.outer(U[:, r], VT[r])*sigma[r]
                if r == 0:
                    X = approx_inc
                else:
                    X += approx_inc
            atom_indices_1 = rotatable_bonds[i]
            atom_indices_2 = rotatable_bonds[j]
            path_len = min(path_length[atom_indices_1[0], atom_indices_2[0]],
                           path_length[atom_indices_1[0], atom_indices_2[1]],
                           path_length[atom_indices_1[1], atom_indices_2[0]],
                           path_length[atom_indices_1[1], atom_indices_2[1]],
                           )
            mse = (np.square(hist - X)).mean()
            methyl_pair = (is_methyl(mol, atom_indices_1[0]) or is_methyl(mol, atom_indices_1[1])) and \
                          (is_methyl(mol, atom_indices_2[0]) or is_methyl(mol, atom_indices_2[1]))
            out.append({"bonds": (atom_indices_1, atom_indices_2),
                        "path_length": path_len,
                        "mse": mse,
                        "mol_id": mol_id,
                        "rank": rank,
                        "methyl_pair": methyl_pair})

    return out


def generate_independence_analysis(mol_db_filename, max_mol, max_num_atoms, smallest_element_subset, morgan_digits,
                                   output_filename, bins) -> None:
    """
    Generate independence analysis.

    :param mol_db_filename: Database filename containing mols.
    :param max_mol: Maximum number of mols.
    :param max_num_atoms: Maximum number of atoms.
    :param smallest_element_subset: Atom type subset allowed.
    :param morgan_digits: Morgan digits allowed.
    :param output_filename: Output filename.
    :param bins: Number of histogram bins.
    """
    mol_db_filename = mol_db_filename
    assert os.path.exists(mol_db_filename)
    db_session = create_session(mol_db_filename)

    # Get molecule ids
    all_molids = [a[0] for a in db_session.execute(select([Molecule.id])).all()]
    all_molids = get_molid_random_subset(all_molids, max_mol)

    out = []
    for rank in RANKS:
        print(f"Generating rank {rank} reconstructions...")
        # Extract confs
        for mol_id in tqdm(all_molids):
            stmt = select([Molecule]).where(Molecule.id == int(mol_id))
            db_mol = db_session.execute(stmt).one()[0]

            # Filtering
            if db_mol.num_atoms <= max_num_atoms and \
                    db_mol.smallest_element_subset in smallest_element_subset and \
                    db_mol.morgan4_crc32 % 10 in morgan_digits:
                mol = Chem.Mol(db_mol.mol)
                out += get_svd(mol, rank, bins, mol_id)

    # Save results
    df = pd.DataFrame.from_dict(out)
    df.to_pickle(output_filename)


if __name__ == "__main__":
    generate_independence_analysis(MOL_DB_FILENAME, MAX_MOL, MAX_NUM_ATOMS, SMALLEST_ELEMENT_SUBSET, MORGAN_DIGITS,
                                   OUTPUT_FILENAME, BINS)



