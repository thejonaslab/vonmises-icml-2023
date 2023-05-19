""" Generate dataset analysis."""
import os

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Lipinski import RotatableBondSmarts
from sqlalchemy import select
from tqdm import tqdm

from vonmises.sqlalchemy_model import Molecule, create_session


WORKING_DIR = "results/dataset-analysis"


def generate_dataset_analysis(mol_db_filename, max_mol, max_num_atoms, smallest_element_subset, morgan_digits,
                              out_path) -> None:
    """
    Generate dataset analysis.

    :param mol_db_filename: Mol database filename.
    :param max_mol: Max number of molecules.
    :param max_num_atoms: Max number of atoms.
    :param smallest_element_subset: Smallest element subset.
    :param morgan_digits: Allowed morgan digits.
    :param out_path: Output path.
    """
    os.makedirs(WORKING_DIR, exist_ok=True)
    mol_db_filename = mol_db_filename
    assert os.path.exists(mol_db_filename)
    db_session = create_session(mol_db_filename)

    # Get molecule ids
    print("Getting ids...")
    all_molids = [a[0] for a in db_session.execute(select([Molecule.id])).all()]

    out = []
    # Extract confs
    count = 0
    for mol_id in tqdm(all_molids):
        stmt = select([Molecule]).where(Molecule.id == int(mol_id))
        db_mol = db_session.execute(stmt).one()[0]

        # Filtering
        if db_mol.num_atoms <= max_num_atoms and \
                db_mol.smallest_element_subset in smallest_element_subset and \
                db_mol.morgan4_crc32 % 10 in morgan_digits:
            count += 1
            mol = Chem.Mol(db_mol.mol)
            rotatable_bonds = mol.GetSubstructMatches(RotatableBondSmarts)
            out.append([db_mol.source_id, len(rotatable_bonds), mol.GetNumAtoms()])

            if count == max_mol:
                break

    df = pd.DataFrame(out)
    df.columns = ["ID", "Number of Rotatable Bonds", "Number of Atoms"]
    df.to_pickle(os.path.join(WORKING_DIR, out_path))


if __name__ == "__main__":
    generate_dataset_analysis("data/mol-data/nmrshiftdb-pt-conf-mols.db",
                              1000000,
                              64,
                              "HCONFSPCl",
                              {2, 3, 4, 5, 6, 7, 8, 9},
                              "nmrshiftdb_train.pkl")

    generate_dataset_analysis("data/mol-data/nmrshiftdb-pt-conf-mols.db",
                              1000000,
                              64,
                              "HCONFSPCl",
                              {0, 1},
                              "nmrshiftdb_test.pkl")

    generate_dataset_analysis("data/mol-data/GDB-17-stereo-pt-conf-mols.db",
                              1000000,
                              64,
                              "HCONFSPCl",
                              {2, 3, 4, 5, 6, 7, 8, 9},
                              "gdb_train.pkl")

    generate_dataset_analysis("data/mol-data/GDB-17-stereo-pt-conf-mols.db",
                              1000000,
                              64,
                              "HCONFSPCl",
                              {0, 1},
                              "gdb_test.pkl")
