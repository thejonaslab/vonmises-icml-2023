""" Add RDKit binary molecules to database. """
import os
import pickle
import zlib

import click
from rdkit import Chem
from rdkit.Chem import rdchem, rdMolDescriptors
from tqdm import tqdm

from vonmises.sqlalchemy_model import *

DATA_DIR = "data/mol-data"


# noinspection PyUnresolvedReferences
def save_to_db(db_session: session.Session, rdmol: rdchem.Mol, source_id: int, session_commit: bool = True):
    """
    Save to db session.

    :param db_session: SQLAlchemy db session.
    :param rdmol: RDKit mol object.
    :param source_id: Unique molecule ID.
    :param session_commit: Whether or not to commit to db.
    """
    # Atom type sets
    atom_set_large = {'F', 'S', 'P', 'Cl'}

    # Extract hash of Morgan fingerprint (used for train/test split)
    mf = rdMolDescriptors.GetHashedMorganFingerprint(rdmol, 4)
    morgan4_crc32 = zlib.crc32(mf.ToBinary())

    num_atoms = rdmol.GetNumAtoms()

    smallest_element_subset = 'HCON'
    for rdatom in rdmol.GetAtoms():
        symbol = rdatom.GetSymbol()
        if symbol in atom_set_large:
            smallest_element_subset = 'HCONFSPCl'
            break

    mol = Molecule(source_id=source_id, mol=rdmol.ToBinary(), morgan4_crc32=morgan4_crc32, num_atoms=num_atoms,
                   smallest_element_subset=smallest_element_subset)

    db_session.add(mol)

    if session_commit:
        db_session.commit()


@click.command()
@click.argument('data_path', type=str)
@click.argument('out_name', type=str)
def mols_to_db(data_path: str, out_name: str) -> None:
    """
    Create metadata folder and file.

    :param data_path: Path to pickle file containing RDKit molecule binary files.
    :param out_name: Name for output file.
    """
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    print(f'Creating db session...')
    db_session = create_session(os.path.join(DATA_DIR, f'{out_name}.db'))

    mol_dict = pickle.load(open(data_path, "rb"))

    print(f'Processing RDKit mols...')
    for source_id, mol_binary in tqdm(mol_dict.items()):
        # noinspection PyUnresolvedReferences
        mol = Chem.Mol(mol_binary)
        save_to_db(db_session, mol, source_id)


if __name__ == '__main__':
    mols_to_db()
