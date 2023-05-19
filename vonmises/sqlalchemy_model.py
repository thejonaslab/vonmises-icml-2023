""" Data model. """
from typing import List
from typing_extensions import Literal

from sqlalchemy import Column, create_engine, Index, Integer, LargeBinary, MetaData, select, String, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import session, sessionmaker

Base = declarative_base()


class Molecule(Base):
    """
    Molecules and associated metadata.
    """
    __tablename__ = "molecules"
    id = Column(Integer, primary_key=True)

    # Source ID identifier for molecule
    source_id = Column(Integer)

    # RDKit molecule binary format for stability and unambiguous interpretation of bonds and atoms
    mol = Column(LargeBinary)

    # Hash of Morgan fingerprint
    morgan4_crc32 = Column(Integer, index=True, nullable=False)

    # Total number of atoms in the molecule
    num_atoms = Column(Integer, index=True, nullable=False)

    # Smallest set that atoms in molecule belong to (e.g. HCON, HCONFSPCL)
    smallest_element_subset = Column(String, index=True, nullable=False)

    # Including indices makes queries faster
    Index('everything', morgan4_crc32, num_atoms, smallest_element_subset)


def create_session(filename: str) -> session.Session:
    """
    Create a SQLAlchemy db session.

    :param filename: File name for db session.
    :return: db session.
    """
    engine = create_engine(f'sqlite:///{filename}', echo=False)
    Base.metadata.create_all(engine)

    Session = sessionmaker(bind=engine)
    session = Session()
    return session


def subset_query(db_path: str,
                 max_atoms: int = 32,
                 smallest_element_subset: Literal['HCON', 'HCONFSPCl'] = 'HCON',
                 phase: Literal["train", "test"] = "train",
                 max_num_mols: int = -1) -> List[int]:
    """
    SQLAlchemy query for molecule IDs satisfying max # atoms, atom type, and last digit of Morgan hash criteria.

    :param db_path: Path to database file.
    :param max_atoms: Maximum number of atoms allowed.
    :param smallest_element_subset: Smallest element subset.
    :param phase: Train or test phase.
    :param max_num_mols: Maximum number of molecules to include in query result.
    :return: List of Molecule ids satisfying the given filters.
    """
    if phase == "train":
        morgan_hash_digit = {2, 3, 4, 5, 6, 7, 8, 9}
    else:
        morgan_hash_digit = {0, 1}

    if smallest_element_subset == 'HCONFSPCl':
        atom_subsets = ['HCON', 'HCONFSPCl']
    else:
        atom_subsets = ['HCON']
    engine = create_engine(f"sqlite+pysqlite:///{db_path}?immutable=1", future=True)
    metadata_obj = MetaData()
    print(f'Reading from {db_path}...')
    molecules = Table("molecules", metadata_obj, autoload_with=engine)
    query = select([molecules.c.id,
                    molecules.c.num_atoms,
                    molecules.c.smallest_element_subset,
                    molecules.c.morgan4_crc32])
    if max_num_mols > 0:
        query = query.limit(max_num_mols)

    target_records = []
    with engine.connect() as conn:
        for row in conn.execute(query):
            if (row['num_atoms'] <= max_atoms) and \
               (row['smallest_element_subset'] in atom_subsets) and \
               (row['morgan4_crc32'] % 10 in morgan_hash_digit):
                target_records.append(row['id'])
            
    return target_records
