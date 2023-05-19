""" Pipeline to generate distributions. """
import os
import pickle
import subprocess
import time

import numpy as np
from rdkit import Chem
from ruffus import *
from sqlalchemy import select
import sqlitedict
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from vonmises import confdists, etkdg, mol_utils, predictor
from vonmises.sqlalchemy_model import Molecule, create_session

WORKING_DIR = 'results/conformation.dists-nmrshiftdb'
MOL_DB_FILENAME = "data/mol-data/nmrshiftdb-pt-conf-mols.db"
NUM_MOL = 1000
MAX_NUM_ATOMS = 64
SMALLEST_ELEMENT_SUBSET = "HCONFSPCl"
MORGAN_DIGITS = {0, 1}
NUM_CONFS = 560
USE_CUDA = True

SAMPLES_EXPS = {
    'samples': {
        'mol_db_filename': MOL_DB_FILENAME,
        'num_mol': NUM_MOL,
        'max_num_atoms': MAX_NUM_ATOMS,
        'smallest_element_subset': SMALLEST_ELEMENT_SUBSET,
        'morgan_digits': MORGAN_DIGITS,
        'num_confs': NUM_CONFS
    },
}

ETKDG_EXPS = {
    'ETKDG-Clean': {
        'mol_db_filename': MOL_DB_FILENAME,
        'num_mol': NUM_MOL,
        'func': 'clean-1',
        'max_num_atoms': MAX_NUM_ATOMS,
        'smallest_element_subset': SMALLEST_ELEMENT_SUBSET,
        'morgan_digits': MORGAN_DIGITS,
        'num_confs': NUM_CONFS,
        'max_embed_attempts': 10,
        'max_workers': 64,
        'exception_for_num_failure': False
    },
}

VM_PRED_EXPS = {
    'VonMisesNet': {
        'mol_db_filename': MOL_DB_FILENAME,
        'model_config': {'chk': "models/nmrshiftdb_20220924_fix_chirality.nmrshiftdb_20220924.644553492873.best.chk",
                         'yaml': "models/nmrshiftdb_20220924_fix_chirality.nmrshiftdb_20220924.644553492873.yaml"},
        'num_mol': NUM_MOL,
        'use_cuda': USE_CUDA,
        'max_num_atoms': MAX_NUM_ATOMS,
        'smallest_element_subset': SMALLEST_ELEMENT_SUBSET,
        'morgan_digits': MORGAN_DIGITS,
        'num_confs': NUM_CONFS,
        'filtering': False
    },
    'VonMisesNet-Filtered': {
        'mol_db_filename': MOL_DB_FILENAME,
        'model_config': {
            'chk': "models/nmrshiftdb_20220924_fix_chirality.nmrshiftdb_20220924.644553492873.best.chk",
            'yaml': "models/nmrshiftdb_20220924_fix_chirality.nmrshiftdb_20220924.644553492873.yaml"},
        'num_mol': NUM_MOL,
        'use_cuda': USE_CUDA,
        'max_num_atoms': MAX_NUM_ATOMS,
        'smallest_element_subset': SMALLEST_ELEMENT_SUBSET,
        'morgan_digits': MORGAN_DIGITS,
        'num_confs': NUM_CONFS,
        'filtering': True,
        'max_attempts': 5
    },
}

# GEOMOL_EXPS = {
#     'GeoMol': {
#         'mol_db_filename': MOL_DB_FILENAME,
#         'model_dir': "comparisons/GeoMol/gdb-64a-HCONFSPCl-run-20221009",
#         'num_mol': NUM_MOL,
#         'max_num_atoms': MAX_NUM_ATOMS,
#         'smallest_element_subset': SMALLEST_ELEMENT_SUBSET,
#         'morgan_digits': MORGAN_DIGITS,
#         'num_confs': NUM_CONFS,
#         'dataset': 'drugs'
#     },
# }
#
# TORSIONAL_DIFFUSION_EXPS = {
#     'TorsionalDiffusion': {
#         'mol_db_filename': MOL_DB_FILENAME,
#         'model_dir': "./comparisons/torsional-diffusion/workdir/boltz_T300",
#         'original_model_dir': "comparisons/torsional-diffusion/workdir/drugs_seed_boltz",
#         'num_mol': NUM_MOL,
#         'max_num_atoms': MAX_NUM_ATOMS,
#         'smallest_element_subset': SMALLEST_ELEMENT_SUBSET,
#         'morgan_digits': MORGAN_DIGITS,
#         'num_confs': NUM_CONFS,
#         'temp': 293,
#         'model_steps': 20,
#         'use_cuda': USE_CUDA
#     },
# }


def get_molid_random_subset(molids, number, random_seed=0):
    """
    Get random subset of mol ids.
    """
    rng = np.random.default_rng(seed=random_seed)

    molids = np.sort(molids)
    molid_subset = rng.choice(molids,
                              size=number,
                              replace=False)
    return molid_subset


def sample_params():
    """
    Sample params.
    """
    for exp_name, ec in SAMPLES_EXPS.items():
        yield None, os.path.join(WORKING_DIR, exp_name + ".db"), exp_name, ec


@mkdir(WORKING_DIR)
@files(sample_params)
def create_samples_from_db(infile, outfile, exp_name, ec):
    """
    Create samples from db.
    """
    # Create db session.
    mol_db_filename = ec['mol_db_filename']
    assert os.path.exists(mol_db_filename)
    db_session = create_session(mol_db_filename)

    # Get molecule ids
    all_molids = [a[0] for a in db_session.execute(select([Molecule.id])).all()]
    if 'max_mol' in ec:
        all_molids = get_molid_random_subset(all_molids, ec['max_mol'])

    # Create out file
    db = sqlitedict.SqliteDict(outfile)

    # Extract confs
    count = 0
    for mol_pos, mol_id in enumerate(all_molids):
        stmt = select([Molecule]).where(Molecule.id == int(mol_id))
        db_mol = db_session.execute(stmt).one()[0]

        mol_id = db_mol.id
        mol = Chem.Mol(db_mol.mol)

        # Filtering
        if not (db_mol.num_atoms <= ec['max_num_atoms'] and
                db_mol.smallest_element_subset in ec['smallest_element_subset'] and
                db_mol.morgan4_crc32 % 10 in ec['morgan_digits']):
            continue

        cd = confdists.SampleDistribution(mol)
        db[mol_id] = cd

        if mol_pos % 1000 == 0:
            db.commit()

        count += 1
        if 'num_mol' in ec:
            if count == ec['num_mol']:
                break
    db.commit()


def etkdg_params():
    """
    ETKDG params.
    """
    for exp_name, ec in ETKDG_EXPS.items():
        yield None, os.path.join(WORKING_DIR, exp_name + ".db"), exp_name, ec


def etkdg_dist_gen(mol, ec):
    """
    ETKDG generation.
    """
    print('working on', Chem.MolToSmiles(mol))
    pickle.dump(mol, open(f'mol.{time.time()}.mol.pickle', 'wb'))

    etkdg_func = ec.get('func', 'default_etkdg')
    sample_n = ec['num_confs']

    start_time = time.time()
    if etkdg_func == 'default_etkdg':
        new_mol = etkdg.generate_etkdg_confs(mol, sample_n)
    elif etkdg_func == 'clean-1':
        try:
            new_mol = etkdg.generate_clean_etkdg_confs(mol, sample_n,
                                                       conform_to_existing_conf_idx=0,
                                                       max_embed_attempts=ec.get('max_embed_attempts', 10),
                                                       exception_for_num_failure=
                                                       ec.get('exception_for_num_failure', True))
        except Exception as e:
            print(f"error: {e} for mol {Chem.MolToSmiles(mol)}")
            return None

    else:
        raise NotImplementedError()

    end_time = time.time()

    mol = etkdg.copy_confs(new_mol, mol)

    cd = confdists.SampleDistribution(mol, {'runtime': end_time - start_time})

    return cd


@mkdir(WORKING_DIR)
@files(etkdg_params)
def create_etkdg_from_db(infile, outfile, exp_name, ec):
    """
    Create ETKDG db.
    """
    # Create db session
    mol_db_filename = ec['mol_db_filename']
    assert os.path.exists(mol_db_filename)
    db_session = create_session(mol_db_filename)

    # Get molecule ids
    all_molids = [a[0] for a in db_session.execute(select([Molecule.id])).all()]
    if 'max_mol' in ec:
        all_molids = get_molid_random_subset(all_molids, ec['max_mol'])

    # Load all molecules
    mols = []
    mol_ids = []
    count = 0
    for mol_pos, mol_id in enumerate(all_molids):
        stmt = select([Molecule]).where(Molecule.id == int(mol_id))
        db_mol = db_session.execute(stmt).one()[0]
        mol = Chem.Mol(db_mol.mol)

        if not (db_mol.num_atoms <= ec['max_num_atoms'] and
                db_mol.smallest_element_subset in ec['smallest_element_subset'] and
                db_mol.morgan4_crc32 % 10 in ec['morgan_digits']):
            continue

        mols.append(mol)
        mol_ids.append(mol_id)

        count += 1
        if 'num_mol' in ec:
            if count == ec['num_mol']:
                break

    # Create output file
    db = sqlitedict.SqliteDict(outfile)

    # Generate confs
    for mol_id, mol, cd in zip(mol_ids, mols, process_map(etkdg_dist_gen,
                                                          mols, [ec] * len(mols),
                                                          max_workers=ec.get('max_workers', 32))):
        if cd is not None:
            if cd.mol.GetNumConformers() > 0:
                print('setting the mol_id to', mol_id)
                db[int(mol_id)] = cd
                db.commit()
    db.commit()


def vm_pred_params():
    """
    VM pred params.
    """
    for exp_name, ec in VM_PRED_EXPS.items():
        yield None, os.path.join(WORKING_DIR, exp_name + ".db"), exp_name, ec


@mkdir(WORKING_DIR)
@files(vm_pred_params)
def create_vm_pred_db(infile, outfile, exp_name, ec):
    """
    Create VM prediction database.
    """
    # Create db session
    mol_db_filename = ec['mol_db_filename']
    assert os.path.exists(mol_db_filename)
    db_session = create_session(mol_db_filename)

    # Get molecule ids
    all_molids = [a[0] for a in db_session.execute(select([Molecule.id])).all()]
    if 'max_mol' in ec:
        all_molids = get_molid_random_subset(all_molids, ec['max_mol'])

    # Initialize predictor
    model_config = ec['model_config']
    pred = predictor.Predictor(model_config['chk'],
                               model_config['yaml'],
                               use_cuda=ec['use_cuda'])

    # Create out file
    out_db = sqlitedict.SqliteDict(outfile)

    # Load all molecules
    all_mols_info = []
    count = 0
    for mol_pos, mol_id in enumerate(all_molids):
        stmt = select([Molecule]).where(Molecule.id == int(mol_id))
        db_mol = db_session.execute(stmt).one()[0]

        if not (db_mol.num_atoms <= ec['max_num_atoms'] and
                db_mol.smallest_element_subset in ec['smallest_element_subset'] and
                db_mol.morgan4_crc32 % 10 in ec['morgan_digits']):
            continue

        mol_id = db_mol.id
        mol = Chem.Mol(db_mol.mol)

        # Start from an initial ETKDG-Clean conf
        try:
            mol = etkdg.generate_clean_etkdg_confs(mol, 1, conform_to_existing_conf_idx=0, seed=1234)
            all_mols_info.append({'mol_id': mol_id,
                                  'rdmol': mol})
        except Exception as e:
            print(e)

        count += 1
        if 'num_mol' in ec:
            if count == ec['num_mol']:
                break

    # Generate predictions
    all_mols = [d['rdmol'] for d in all_mols_info]
    preds, meta = pred.predict(all_mols)

    # Generate conformations
    for mol_pos, (in_mol, out_pred, out_meta) in tqdm(enumerate(zip(all_mols_info, preds, meta))):
        pred_time = out_meta["processing_time"]

        mol_id = in_mol['mol_id']
        start_time = time.time()

        new_mol = predictor.generate_confs(preds[mol_pos], ec['num_confs'], ec['filtering'], ec.get('max_attempts', 5))
        end_time = time.time()
        if new_mol.GetNumConformers() > 0:
            cd = confdists.SampleDistribution(new_mol, {'runtime': pred_time + (end_time - start_time)})
            out_db[int(mol_id)] = cd
        if mol_pos % 1000 == 0:
            out_db.commit()
    out_db.commit()


# def geomol_params():
#     """
#     GeoMol params.
#     """
#     for exp_name, ec in GEOMOL_EXPS.items():
#         yield None, os.path.join(WORKING_DIR, exp_name + ".db"), exp_name, ec
#
#
# @mkdir(WORKING_DIR)
# @files(geomol_params)
# def create_geomol_from_db(infile, outfile, exp_name, ec):
#     """
#     Create GeoMol db.
#     """
#     # Create db session
#     mol_db_filename = ec['mol_db_filename']
#     assert os.path.exists(mol_db_filename)
#     db_session = create_session(mol_db_filename)
#
#     # Get molecule ids
#     all_molids = [a[0] for a in db_session.execute(select([Molecule.id])).all()]
#     if 'max_mol' in ec:
#         all_molids = get_molid_random_subset(all_molids, ec['max_mol'])
#
#     # Create out file
#     db = sqlitedict.SqliteDict(outfile)
#
#     # Set CPU env variable
#     os.environ['GEOMOL_USE_CPU'] = "true"
#
#     # Set num confs env variable
#     os.environ['GEOMOL_NUM_CONFS'] = str(ec['num_confs'])
#
#     # Generate confs
#     count = 0
#     for mol_pos, mol_id in enumerate(all_molids):
#         print(f"{count} / {ec['num_mol']}")
#         stmt = select([Molecule]).where(Molecule.id == int(mol_id))
#         db_mol = db_session.execute(stmt).one()[0]
#
#         mol_id = db_mol.id
#         mol = Chem.Mol(db_mol.mol)
#         mol.RemoveAllConformers()
#
#         # Filtering
#         if not (db_mol.num_atoms <= ec['max_num_atoms'] and
#                 db_mol.smallest_element_subset in ec['smallest_element_subset'] and
#                 db_mol.morgan4_crc32 % 10 in ec['morgan_digits']):
#             continue
#
#         mol_renumbered, atom_mapping_array = mol_utils.reorder_mol_atoms_like_smiles(mol)
#
#         bin_out = set()
#         bin_out.add(mol_renumbered.ToBinary())
#         with open(f"geomol_{mol_id}.pkl", "wb") as f:
#             pickle.dump(bin_out, f)
#
#         output = subprocess.run(
#             f"python comparisons/GeoMol/generate_confs_rdkit.py --trained_model_dir {ec['model_dir']} "
#             f"--test_rdkit geomol_{mol_id}.pkl --dataset {ec['dataset']} --out geomol_{mol_id}_pred.pkl",
#             shell=True,
#             capture_output=True)
#         print('Subprocess: ')
#         print(f'args: {output.args}')
#         print(f'return code: {output.returncode}')
#         print(f'output: {output.stdout.decode("utf-8")}')
#         print(f'error: {output.stderr.decode("utf-8")}\n')
#
#         geomol_pred = pickle.load(open(f"geomol_{mol_id}_pred.pkl", "rb"))
#         for mol_pred in list(geomol_pred.values())[0]:
#             arr2 = [0] * (len(atom_mapping_array))
#             for i in range(0, len(atom_mapping_array)):
#                 arr2[atom_mapping_array[i]] = i
#             mol_pred = Chem.rdmolops.RenumberAtoms(mol_pred, arr2)
#             mol.AddConformer(mol_pred.GetConformers()[0])
#
#         timing = pickle.load(open(f"geomol_{mol_id}_pred.pkl_timing", "rb"))
#         cd = confdists.SampleDistribution(mol, timing)
#         db[mol_id] = cd
#
#         if os.path.exists(f"geomol_{mol_id}_pred.pkl"):
#             os.remove(f"geomol_{mol_id}_pred.pkl")
#         if os.path.exists(f"geomol_{mol_id}_pred.pkl_timing"):
#             os.remove(f"geomol_{mol_id}_pred.pkl_timing")
#         if os.path.exists(f"geomol_{mol_id}.pkl"):
#             os.remove(f"geomol_{mol_id}.pkl")
#
#         if mol_pos % 1000 == 0:
#             db.commit()
#
#         count += 1
#         if 'num_mol' in ec:
#             if count == ec['num_mol']:
#                 break
#     db.commit()
#
#
# def torsional_diffusion_params():
#     """
#     Torsional diffusion params.
#     """
#     for exp_name, ec in TORSIONAL_DIFFUSION_EXPS.items():
#         yield None, os.path.join(WORKING_DIR, exp_name + ".db"), exp_name, ec
#
#
# @mkdir(WORKING_DIR)
# @files(torsional_diffusion_params)
# def create_torsional_diffusion_from_db(infile, outfile, exp_name, ec):
#     """
#     Create torsional diffusion db.
#     """
#     # Create db session
#     mol_db_filename = ec['mol_db_filename']
#     assert os.path.exists(mol_db_filename)
#     db_session = create_session(mol_db_filename)
#
#     # Get molecule ids
#     all_molids = [a[0] for a in db_session.execute(select([Molecule.id])).all()]
#     if 'max_mol' in ec:
#         all_molids = get_molid_random_subset(all_molids, ec['max_mol'])
#
#     # Create out file
#     db = sqlitedict.SqliteDict(outfile)
#
#     # Set CUDA env variable
#     if ec['use_cuda']:
#         os.environ['TORSIONAL_DIFFUSION_USE_CPU'] = "false"
#     else:
#         os.environ['TORSIONAL_DIFFUSION_USE_CPU'] = "true"
#
#     # Generate conformations
#     count = 0
#     for mol_pos, mol_id in enumerate(all_molids):
#         print(f"{count} / {ec['num_mol']}")
#
#         # Load molecule
#         stmt = select([Molecule]).where(Molecule.id == int(mol_id))
#         db_mol = db_session.execute(stmt).one()[0]
#         mol_id = db_mol.id
#         mol = Chem.Mol(db_mol.mol)
#
#         # Filtering
#         if not (db_mol.num_atoms <= ec['max_num_atoms'] and
#                 db_mol.smallest_element_subset in ec['smallest_element_subset'] and
#                 db_mol.morgan4_crc32 % 10 in ec['morgan_digits']):
#             continue
#
#         count += 1
#         if 'num_mol' in ec:
#             if count == ec['num_mol']:
#                 break
#
#         # Generate initial conf using ETKDG-Clean
#         try:
#             mol = etkdg.generate_clean_etkdg_confs(mol, 1, conform_to_existing_conf_idx=0, seed=1234)
#         except Exception as e:
#             print(e)
#             continue
#
#         # Reorder atom indices
#         mol_renumbered, atom_mapping_array = mol_utils.reorder_mol_atoms_like_smiles(mol)
#
#         # Create input file
#         bin_out = set()
#         bin_out.add(mol_renumbered.ToBinary())
#         with open(f"torsional_diffusion_{mol_id}.pkl", "wb") as f:
#             pickle.dump(bin_out, f)
#
#         # Run torsional diffusion
#         output = subprocess.run(
#             f"python ./comparisons/torsional-diffusion/test_boltzmann_rdkit.py --model_dir {ec['model_dir']} "
#             f"--temp {ec['temp']} --model_steps {ec['model_steps']} --original_model_dir {ec['original_model_dir']} "
#             f"--out torsional_diffusion_{mol_id}_pred.pkl --n_samples {ec['num_confs']} "
#             f"--test_pkl torsional_diffusion_{mol_id}.pkl",
#             shell=True,
#             capture_output=True)
#         print('Subprocess: ')
#         print(f'args: {output.args}')
#         print(f'return code: {output.returncode}')
#         print(f'output: {output.stdout.decode("utf-8")}')
#         print(f'error: {output.stderr.decode("utf-8")}\n')
#
#         # Save predictions
#         try:
#             torsional_diffusion_pred = pickle.load(open(f"torsional_diffusion_{mol_id}_pred.pkl", "rb"))
#             torsional_diffusion_pred_mols = torsional_diffusion_pred["mol_pred"]
#             process_time = torsional_diffusion_pred["processing_time"]
#             for mol_pred in torsional_diffusion_pred_mols:
#                 mol_pred = Chem.Mol(mol_pred)
#
#                 # Undo atom index reordering
#                 arr2 = [0] * (len(atom_mapping_array))
#                 for i in range(0, len(atom_mapping_array)):
#                     arr2[atom_mapping_array[i]] = i
#                 mol_pred = Chem.rdmolops.RenumberAtoms(mol_pred, arr2)
#
#                 # Extract conformer
#                 mol.AddConformer(mol_pred.GetConformers()[0])
#
#             cd = confdists.SampleDistribution(mol, {'runtime': process_time})
#             db[mol_id] = cd
#         except Exception as e:
#             print(e)
#
#         if os.path.exists(f"torsional_diffusion_{mol_id}_pred.pkl"):
#             os.remove(f"torsional_diffusion_{mol_id}_pred.pkl")
#         if os.path.exists(f"torsional_diffusion_{mol_id}.pkl"):
#             os.remove(f"torsional_diffusion_{mol_id}.pkl")
#
#         if mol_pos % 1000 == 0:
#             db.commit()
#     db.commit()


if __name__ == "__main__":
    pipeline_run([create_samples_from_db, create_etkdg_from_db, create_vm_pred_db])
