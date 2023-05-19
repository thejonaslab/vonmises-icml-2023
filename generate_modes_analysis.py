""" Generate modes analysis. """
import os

import pandas as pd
from rdkit import Chem
from sqlalchemy import select
from tqdm import tqdm

from generate_distributions_pipeline import get_molid_random_subset
from vonmises import mol_utils
from vonmises.sqlalchemy_model import Molecule, create_session

MOL_DB_FILENAME = "data/mol-data/nmrshiftdb-pt-conf-mols.db"
MAX_MOL = 1000
MAX_NUM_ATOMS = 64
SMALLEST_ELEMENT_SUBSET = "HCONFSPCl"
MORGAN_DIGITS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
WORKING_DIR = "results/mode-analysis"


def generate_modes_analysis(mol_db_filename, max_mol, max_num_atoms, smallest_element_subset, morgan_digits) -> None:
    """
    Generate modes analysis.
    """
    os.makedirs(WORKING_DIR)
    mol_db_filename = mol_db_filename
    assert os.path.exists(mol_db_filename)
    db_session = create_session(mol_db_filename)

    # Get molecule ids
    print("Getting ids...")
    all_molids = [a[0] for a in db_session.execute(select([Molecule.id])).all()]
    all_molids = get_molid_random_subset(all_molids, max_mol)

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
            mol = Chem.Mol(db_mol.mol)
            df = mol_utils.compute_rotatable_bond_torsions(mol)
            try:
                df_modes = mol_utils.compute_num_torsion_modes(df)
                if not df_modes.empty:
                    out += list(df_modes['Mode Count'].to_numpy())
                    count += 1
            except Exception as e:
                print(e)
    print(f"Total number of mols: {count}")
    df = pd.DataFrame({'modes': out})
    df.to_pickle(os.path.join(WORKING_DIR, "mode_analysis.pkl"))


if __name__ == "__main__":
    generate_modes_analysis(MOL_DB_FILENAME, MAX_MOL, MAX_NUM_ATOMS, SMALLEST_ELEMENT_SUBSET, MORGAN_DIGITS)



