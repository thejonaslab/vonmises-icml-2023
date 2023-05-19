""" Generate conformations for arbitrary SMILES strings. """
import pickle
import time

import click
import pandas as pd
from tqdm import tqdm

from vonmises.predictor import Predictor, generate_confs


@click.command()
@click.option('--csv_path', type=str)
@click.option('--out_path', type=str)
@click.option('--model_path', type=str)
@click.option('--model_config_path', type=str)
@click.option('--num_confs', type=int)
@click.option('--use_cuda', is_flag=True)
@click.option('--filter_confs', is_flag=True)
@click.option('--initial_geometry_seed', type=int, default=-1)
@click.option('--initial_geometry_max_attempts', type=int, default=10)
@click.option('--max_conf_generation_attempts', type=int, default=5)
def generate_confs_from_smiles(csv_path: str, out_path: str, model_path: str, model_config_path: str, num_confs: int,
                               use_cuda: bool = False, filter_confs: bool = False, initial_geometry_seed: int = -1,
                               initial_geometry_max_attempts: int = 10, max_conf_generation_attempts: int = 5):
    """
    Generate conformations for a set of SMILES strings that are in a CSV file, one string per line, with no header.
    A dictionary mapping from SMILES to RDKit molecule objects with the embedded conformations,
    as well as prediction metadata, is saved as a pickle file at ``out_path``.

    :param csv_path: Path to CSV file containing SMILES strings.
    :param out_path: Path to output file.
    :param model_path: Path to model parameters.
    :param model_config_path: Path to model config yaml file.
    :param num_confs: Number of conformations to generate per molecule.
    :param use_cuda: Whether to use CUDA.
    :param initial_geometry_seed: Seed for generating initial conformation.
    :param initial_geometry_max_attempts: Number of times to re-try a failed ETKDG embedding before giving up.
    :param filter_confs: Whether to discard conformations that violate minimum atomic distance thresholds.
    :param max_conf_generation_attempts: Maximum attempts to generate a conformation when filter_confs is True.
    """
    predictor = Predictor(model_path, model_config_path, use_cuda)
    smiles_strings = [smi[0] for smi in pd.read_csv(csv_path, header=None).values]
    print("Generating predictions...")
    preds, metas = predictor.predict(smiles_strings, smiles_input=True, initial_geometry_seed=initial_geometry_seed,
                                     initial_geometry_max_attempts=initial_geometry_max_attempts)
    out = {}
    print("Generating conformations...")
    for i in tqdm(range(len(preds))):
        start_time = time.time()
        smiles = smiles_strings[i]
        pred, meta = preds[i], metas[i]
        if pred is None:
            end_time = time.time()
            out[smiles] = {"mol": None,
                           "prediction_metadata": meta,
                           "conf_gen_metadata": {"processing_time": end_time - start_time,
                                                 "processing_error": "Input prediction dict was None."}}
        else:
            mol = generate_confs(pred, num_confs, filter_confs, max_conf_generation_attempts)
            end_time = time.time()
            out[smiles] = {"mol": mol,
                           "prediction_metadata": meta,
                           "conf_gen_metadata": {"processing_time": end_time - start_time,
                                                 "processing_error": None}}

    with open(out_path, 'wb') as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    generate_confs_from_smiles()
