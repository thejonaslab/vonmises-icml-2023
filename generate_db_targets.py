"""
Generate prediction targets.
"""
import os
import pickle
from typing import Tuple

import numpy as np
from rdkit import Chem
from ruffus import *
from sqlalchemy import Column, create_engine, event, func, insert, Integer, LargeBinary, MetaData, select, Table
from sqlalchemy.engine import Engine
from tqdm import tqdm

from vonmises.sqlalchemy_model import create_session, Molecule
from vonmises import mol_utils


TARGET_DIR = "data/targets.db"

CHIRALITIES = {"kind": "chiralities"}
LENGTHS = {'kind': 'lengths'}
ANGLES = {'kind': 'angles'}
TORSIONS = {'kind': 'torsions'}

DEFAULT_TARGET_KINDS = {"chiralities": CHIRALITIES,
                        'lengths': LENGTHS,
                        'angles': ANGLES,
                        'torsions': TORSIONS}

TARGETS = {'nmrshiftdb-pt': {'data': 'data/mol-data/nmrshiftdb-pt-conf-mols.db',
                              'targets': DEFAULT_TARGET_KINDS},
           'GDB-17-stereo-pt': {'data': 'data/mol-data/GDB-17-stereo-pt-conf-mols.db',
                                'targets': DEFAULT_TARGET_KINDS}
           }


@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """
    Try and increase performance: https://blog.devart.com/increasing-sqlite-performance.html.

    :param dbapi_connection: Database connection.
    :param connection_record: Connection record.
    """
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA PAGE_SIZE=16384")
    cursor.close()


def target_params():
    """
    Generate target parameters.
    """
    for exp_name, ec in TARGETS.items():
        input_data = ec['data']
        assert os.path.exists(input_data)
        outfile = os.path.join(TARGET_DIR, f"{exp_name}.jobs")
        yield input_data, outfile, exp_name, ec


@mkdir(TARGET_DIR)
@files(target_params)
def create_target_jobs(infile, outfile, exp_name, exp_config):
    """
    Create the jobs for downstream Ruffus transform.

    :param infile: Input file.
    :param outfile: Output file.
    :param exp_name: Experiment name.
    :param exp_config: Experiment config.
    """
    for target_name, target_config in exp_config['targets'].items():
        tgt_outfile = os.path.join(TARGET_DIR, f"{exp_name}.{target_name}.job")

        with open(tgt_outfile, 'wb') as fid:
            pickle.dump({'infile': infile,
                         'outfile': outfile,
                         'exp_name': exp_name,
                         'exp_config': exp_config}, fid)

    with open(outfile, 'w') as fid:
        fid.write('done')


def create_chiralities_db(filename) -> Tuple[Engine, Table]:
    """
    Create chiralities database.

    :param filename: File name.
    :return: SQLite engine and table.
    """
    engine = create_engine(f"sqlite+pysqlite:///{filename}", future=True)
    metadata_obj = MetaData()
    length_table = Table(
        "chiralities",
        metadata_obj,
        Column('mol_id', Integer, primary_key=True),
        Column('chiralities_bin', LargeBinary),
        Column('atom_n', Integer),
        Column('atom_idx_bin', LargeBinary),
        )

    metadata_obj.create_all(engine)

    return engine, length_table


def create_lengths_db(filename) -> Tuple[Engine, Table]:
    """
    Create bond lengths database.

    :param filename: File name.
    :return: SQLite engine and table.
    """
    engine = create_engine(f"sqlite+pysqlite:///{filename}", future=True)
    metadata_obj = MetaData()
    length_table = Table(
        "lengths",
        metadata_obj,
        Column('mol_id', Integer, primary_key=True),  
        Column('bond_lengths_bin', LargeBinary),
        Column('bond_n', Integer), 
        Column('bond_idx_bin', LargeBinary),
        Column('atom_idx_bin', LargeBinary),
        )

    metadata_obj.create_all(engine)

    return engine, length_table


def create_angle_db(filename) -> Tuple[Engine, Table]:
    """
    Create bond angles database.

    :param filename: File name.
    :return: SQLite engine and table.
    """
    engine = create_engine(f"sqlite+pysqlite:///{filename}", future=True)
    metadata_obj = MetaData()
    angle_table = Table(
        "angles",
        metadata_obj,
        Column('mol_id', Integer, primary_key=True),  
        Column('angles_bin', LargeBinary),
        Column('angle_n', Integer), 
        Column('atom_idx_bin', LargeBinary),
        )

    metadata_obj.create_all(engine)

    return engine, angle_table


def create_torsion_db(filename) -> Tuple[Engine, Table]:
    """
    Create torsions database.

    :param filename: File name.
    :return: SQLite engine and table.
    """
    engine = create_engine(f"sqlite+pysqlite:///{filename}", future=True)
    metadata_obj = MetaData()
    torsion_table = Table(
        "torsions",
        metadata_obj,
        Column('mol_id', Integer, primary_key=True),  
        Column('angles_bin', LargeBinary),
        Column('bond_n', Integer),
        Column('bond_classes_bin', LargeBinary),
        Column('bond_indices_bin', LargeBinary),
        Column('atom_indices_bin', LargeBinary),
        Column('bond_indices_with_chirality_bin', LargeBinary)
        )

    metadata_obj.create_all(engine)

    return engine, torsion_table


@follows(create_target_jobs)
@transform(os.path.join(TARGET_DIR, "*.chiralities.job"), suffix(".job"), ".db")
def create_chiralities(infile, outfile_db):
    """
    Generate chirality targets.

    :param infile: Input file.
    :param outfile_db: Output database.
    """
    config = pickle.load(open(infile, 'rb'))
    assert os.path.exists(config['infile'])
    db_session = create_session(config['infile'])

    output_eng, length_table = create_chiralities_db(outfile_db)

    total_mol = db_session.scalar(select(func.count(Molecule.id)))

    conn = output_eng.connect()
    out_record_buffer = []
    for db_mol, in tqdm(db_session.execute(select([Molecule])).all(), total=total_mol):
        mol_id = db_mol.id
        mol = Chem.Mol(db_mol.mol)

        chiralities_dict = mol_utils.compute_chirality_probability_targets(mol)
        chiralities = chiralities_dict['chiralities'].astype(np.float16)
        atom_indices = chiralities_dict['atom_indices']

        out_record = {'mol_id': mol_id,
                      'chiralities_bin': chiralities.tobytes(),
                      'atom_n': len(atom_indices),
                      'atom_idx_bin': np.array(atom_indices).tobytes()}

        out_record_buffer.append(out_record)

        if len(out_record_buffer) > 1000:
            conn.execute(insert(length_table), out_record_buffer)
            conn.commit()
            out_record_buffer = []

    if len(out_record_buffer) > 0:
        conn.execute(insert(length_table), out_record_buffer)
        conn.commit()


@follows(create_target_jobs)
@transform(os.path.join(TARGET_DIR, "*.lengths.job"), suffix(".job"), ".db")
def create_lengths(infile, outfile_db):
    """
    Generate bond length targets.

    :param infile: Input file.
    :param outfile_db: Output database.
    """
    config = pickle.load(open(infile, 'rb'))
    assert os.path.exists(config['infile'])
    db_session = create_session(config['infile'])
    
    output_eng, length_table = create_lengths_db(outfile_db)

    total_mol = db_session.scalar(select(func.count(Molecule.id)))
    
    conn = output_eng.connect()
    out_record_buffer = []
    for db_mol, in tqdm(db_session.execute(select([Molecule])).all(), total=total_mol):
        mol_id = db_mol.id
        mol = Chem.Mol(db_mol.mol)
        
        lengths_dict = mol_utils.compute_lengths(mol)
        lengths = lengths_dict['lengths'].astype(np.float16)
        bond_indices = lengths_dict['bond_indices']
        
        out_record = {'mol_id': mol_id,
                      'bond_lengths_bin': lengths.tobytes(),
                      'bond_n': len(bond_indices),
                      'bond_idx_bin': np.array(bond_indices).tobytes()}

        out_record_buffer.append(out_record)

        if len(out_record_buffer) > 1000:
            conn.execute(insert(length_table), out_record_buffer)
            conn.commit()
            out_record_buffer = []
                      
    if len(out_record_buffer) > 0:
        conn.execute(insert(length_table), out_record_buffer)
        conn.commit()
        

@follows(create_target_jobs)
@transform(os.path.join(TARGET_DIR, "*.angles.job"), suffix(".job"), ".db")
def create_angles(infile, outfile_db):
    """
    Generate bond angle targets.

    :param infile: Input file.
    :param outfile_db: Output database.
    """
    config = pickle.load(open(infile, 'rb'))
    assert os.path.exists(config['infile'])
    db_session = create_session(config['infile'])
    
    output_eng, angles_table = create_angle_db(outfile_db)

    total_mol = db_session.scalar(select(func.count(Molecule.id)))

    conn = output_eng.connect()
    out_record_buffer = []
    for db_mol, in tqdm(db_session.execute(select([Molecule])).all(), total=total_mol):
        mol_id = db_mol.id
        mol = Chem.Mol(db_mol.mol)
        
        angles_dict = mol_utils.compute_angles(mol)
        angle_triplets = angles_dict['angle_triplets']
        angles_array = angles_dict['angles'].astype(np.float16)

        out_record = {'mol_id': mol_id,
                      'angles_bin': angles_array.tobytes(),
                      'angle_n': len(angle_triplets),
                      'atom_idx_bin': angle_triplets.tobytes()}

        out_record_buffer.append(out_record)

        if len(out_record_buffer) > 1000:
            conn.execute(insert(angles_table), out_record_buffer)
            conn.commit()
            out_record_buffer = []
                      
    if len(out_record_buffer) > 0:
        conn.execute(insert(angles_table), out_record_buffer)
        conn.commit()
        

@follows(create_target_jobs)
@transform(os.path.join(TARGET_DIR, "*.torsions.job"), suffix(".job"), ".db")
def create_torsions(infile, outfile_db):
    """
    Generate torsion targets.

    :param infile: Input file.
    :param outfile_db: Output database.
    """
    config = pickle.load(open(infile, 'rb'))
    assert os.path.exists(config['infile'])
    db_session = create_session(config['infile'])
    
    output_eng, torsions_table = create_torsion_db(outfile_db)

    total_mol = db_session.scalar(select(func.count(Molecule.id)))
    
    conn = output_eng.connect()
    out_record_buffer = []
    for db_mol, in tqdm(db_session.execute(select([Molecule])).all(), total=total_mol):
        mol_id = db_mol.id
        mol = Chem.Mol(db_mol.mol)
        
        r = mol_utils.compute_torsions(mol, restrict_to_rotatable=True)
        
        out_record = {'mol_id': mol_id,
                      'angles_bin': r['angles'].tobytes(),
                      'bond_n': len(r['angles']),
                      'bond_classes_bin': r['bond_classes'].tobytes(),
                      'bond_indices_bin': r['bond_indices'].tobytes(),
                      'atom_indices_bin': r['atom_index_tuples'].tobytes(),
                      'bond_indices_with_chirality_bin': r['bond_indices_with_chirality'].tobytes()}

        out_record_buffer.append(out_record)

        if len(out_record_buffer) > 1000:
            conn.execute(insert(torsions_table), out_record_buffer)
            conn.commit()
            out_record_buffer = []
                      
    if len(out_record_buffer) > 0:
        conn.execute(insert(torsions_table), out_record_buffer)
        conn.commit()


if __name__ == "__main__":
    pipeline_run([create_target_jobs,
                  create_chiralities,
                  create_lengths,
                  create_angles,
                  create_torsions],
                 multiprocess=8)
