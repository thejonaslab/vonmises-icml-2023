"""
Pipeline for analyzing expectations.
"""
import argparse
import copy
from glob import glob
import os
import pickle
from typing import Tuple

import ot
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Lipinski import RotatableBondSmarts
from scipy.stats import entropy
import sqlitedict
from tqdm.contrib.concurrent import process_map


parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str)
parser.add_argument("--working_dir", type=str)
parser.add_argument("--ignore_torsional_diffusion", action='store_true')
parser.add_argument("--max_dist_to_analyze", type=int, default=10)
parser.add_argument("--num_workers", type=int, default=8)
args = parser.parse_args()

INPUT_DIR = args.input_dir
WORKING_DIR = args.working_dir
BASELINE_NAME = "samples"

dist_to_analyze = []
if args.max_dist_to_analyze > 1:
    dist_to_analyze.append('average_distance_1-2.expect')
if args.max_dist_to_analyze > 2:
    dist_to_analyze.append('average_distance_1-3.expect')
if args.max_dist_to_analyze > 3:
    dist_to_analyze.append('average_distance_1-4.expect')
if args.max_dist_to_analyze > 4:
    dist_to_analyze.append('average_distance_1-5.expect')
if args.max_dist_to_analyze > 5:
    dist_to_analyze.append('average_distance_1-6.expect')
if args.max_dist_to_analyze > 6:
    dist_to_analyze.append('average_distance_1-7.expect')
if args.max_dist_to_analyze > 7:
    dist_to_analyze.append('average_distance_1-8.expect')
if args.max_dist_to_analyze > 8:
    dist_to_analyze.append('average_distance_1-9.expect')
if args.max_dist_to_analyze > 9:
    dist_to_analyze.append('average_distance_1-10.expect')

hist_to_analyze = ['hist_dihedral_32.expect']

dihedral_list_to_analyze = ['list_dihedral.expect']

list_to_analyze = []
if args.max_dist_to_analyze > 3:
    list_to_analyze.append('dist_list_1-4.expect')
if args.max_dist_to_analyze > 4:
    list_to_analyze.append('dist_list_1-5.expect')
if args.max_dist_to_analyze > 5:
    list_to_analyze.append('dist_list_1-6.expect')
if args.max_dist_to_analyze > 6:
    list_to_analyze.append('dist_list_1-7.expect')
if args.max_dist_to_analyze > 7:
    list_to_analyze.append('dist_list_1-8.expect')
if args.max_dist_to_analyze > 8:
    list_to_analyze.append('dist_list_1-9.expect')
if args.max_dist_to_analyze > 9:
    list_to_analyze.append('dist_list_1-10.expect')

distance_values = [2, 3, 4, 5, 6, 7, 8, 9, 10]


def generate_dataframe(metric_name: str):
    """
    Generate data frame for metric.
    """
    data_frames = []
    exp_names = []
    for f in glob(f"{INPUT_DIR}/*__{metric_name}"):
        if args.ignore_torsional_diffusion:
            if "TorsionalDiffusion" in f:
                continue
        data_frame = []
        exp_name = os.path.split(f)[1]
        exp_name = exp_name[:exp_name.index("__")]
        exp_names.append(exp_name)
        db = sqlitedict.SqliteDict(f, flag='r')
        for mol_id, v in db.items():
            mol = Chem.Mol(v['mol'])
            rotatable_bonds = mol.GetSubstructMatches(RotatableBondSmarts)
            num_rotatable_bonds = len(rotatable_bonds)
            for sample_key, hist_result in v["expectation_results"].items():
                data_frame.append([mol_id, sample_key, hist_result, num_rotatable_bonds])
        data_frame = pd.DataFrame(data_frame)
        if not data_frame.empty:
            data_frame.columns = ["mol_id", "sample_key", exp_name, "num_rot_bond"]
            data_frames.append(data_frame)

    if len(data_frames) > 0:
        df = data_frames[0]
        for i in range(1, len(data_frames)):
            df = pd.merge(df, data_frames[i], on=["mol_id", "sample_key", "num_rot_bond"])

        return df, exp_names

    else:
        return pd.DataFrame([]), None


def generate_performance_metrics(df):
    """

    :param df:
    :return:
    """
    df = df.dropna()
    df1 = pd.DataFrame(df.astype('float64').describe().loc['mean'])
    df1['name'] = df1.index
    df2 = pd.DataFrame(df.astype('float64').sem())
    df2.columns = ['sem']
    df2['name'] = df2.index
    df3 = pd.DataFrame(df.astype('float64').describe().loc['std'])
    df3['name'] = df3.index
    new_df = pd.merge(df1, df2, on='name')
    new_df = pd.merge(new_df, df3, on='name')
    new_df = new_df[["name", "mean", "sem", "std"]]
    new_df = new_df.sort_values(by="name")
    new_df = new_df.reset_index(drop=True)

    return new_df


def generate_sample_performance_dataframe(df):
    """

    :param df:
    :return:
    """
    df_sample = copy.deepcopy(df)
    for key in [BASELINE_NAME, "mol_id", "sample_key", "num_rot_bond"]:
        if key in df_sample.columns:
            df_sample = df_sample.drop([key], axis=1)
    df_sample = generate_performance_metrics(df_sample)

    return df_sample


def generate_molecule_performance_dataframe(df):
    """

    :param df:
    :return:
    """
    df_mol = copy.deepcopy(df)
    for key in [BASELINE_NAME, "sample_key", "num_rot_bond"]:
        if key in df_mol.columns:
            df_mol = df_mol.drop([key], axis=1)
    df_mol = df_mol.groupby("mol_id").mean()
    df_mol = generate_performance_metrics(df_mol)

    return df_mol


def generate_performance_by_mol_by_num_rot_bond(df):
    """

    :param df:
    :return:
    """
    return df.drop(["sample_key", "samples"], axis=1).groupby(["mol_id", "num_rot_bond"]).mean().groupby(
                "num_rot_bond").mean()


def generate_diffs_df(df, exp_names):
    """
    :return:
    """
    diffs_df = abs(df[exp_names].subtract(df[BASELINE_NAME], axis=0)).drop(
        BASELINE_NAME, axis=1)
    diffs_df["mol_id"] = df["mol_id"]
    diffs_df = diffs_df.reindex(sorted(diffs_df.columns), axis=1)
    diffs_df = diffs_df.dropna()

    return diffs_df


def compute_metric(expectation: str, metric: str) -> Tuple:
    """
    Compute EMD for a given expectation.

    :param expectation: Name of expectation.
    :param metric: Name of metric.
    :return: Model names, mean values, std values, splits.
    """
    df, exp_names = generate_dataframe(expectation)
    all_model_names = []
    all_mean_values = []
    all_std_values = []
    all_splits = []
    if not df.empty:
        if metric == "EMD":
            df.to_pickle(os.path.join(WORKING_DIR, f"emd_{expectation}_orig.pkl"))
            for i in range(df.shape[0]):
                list_baseline = df[BASELINE_NAME].iloc[i]
                for exp in [e for e in exp_names if e != BASELINE_NAME]:
                    list_pred = df[exp].iloc[i]
                    df.at[i, exp] = ot.emd2_1d(list_baseline, list_pred,
                                                 [1./len(list_baseline)]*len(list_baseline),
                                                 [1./len(list_pred)]*len(list_pred))
            df.to_pickle(os.path.join(WORKING_DIR, f"emd_{expectation}.pkl"))
        elif metric == "MAE":
            df = generate_diffs_df(df, exp_names)
            df.to_pickle(os.path.join(WORKING_DIR, f"dist_{expectation}.pkl"))
        elif metric == "Entropy":
            for i in range(df.shape[0]):
                hist_baseline = df[BASELINE_NAME].iloc[i]
                for exp in [e for e in exp_names if e != BASELINE_NAME]:
                    hist_pred = df[exp].iloc[i]
                    df.at[i, exp] = entropy(hist_pred, hist_baseline)
            df.to_pickle(os.path.join(WORKING_DIR, f"kl_{expectation}.pkl"))
        new_df = generate_molecule_performance_dataframe(df)
        all_model_names.append(new_df["name"].to_numpy())
        all_mean_values.append(new_df["mean"].to_numpy())
        all_std_values.append(new_df["sem"].to_numpy())
        for num in distance_values:
            if f"1-{num}" in expectation:
                all_splits.append(f"1-{num}")

    return all_model_names, all_mean_values, all_std_values, all_splits


def compute_mae(expectation) -> Tuple:
    """
    Compute MAE for a given expectation.

    :param expectation: Name of expectation.
    :return: Model names, mean values, std values, splits.
    """
    df, exp_names = generate_dataframe(expectation)
    all_model_names = []
    all_mean_values = []
    all_std_values = []
    all_splits = []
    if not df.empty:
        diffs_df = generate_diffs_df(df, exp_names)
        diffs_df.to_pickle(os.path.join(WORKING_DIR, f"dist_{expectation}.pkl"))
        new_df = generate_molecule_performance_dataframe(diffs_df)
        all_model_names.append(new_df["name"].to_numpy())
        all_mean_values.append(new_df["mean"].to_numpy())
        all_std_values.append(new_df["sem"].to_numpy())
        for num in distance_values:
            if f"1-{num}" in expectation:
                all_splits.append(f"1-{num}")

    return all_model_names, all_mean_values, all_std_values, all_splits


def run_analysis():
    """
    Run analysis on expectations.

    :return:
    """
    os.makedirs(WORKING_DIR, exist_ok=True)

    process_map_outs = []
    for res in process_map(compute_metric, hist_to_analyze, ["Entropy"] * len(hist_to_analyze)):
        process_map_outs.append(res)

    all_model_names = []
    all_mean_values = []
    all_std_values = []
    all_splits = []
    for res in process_map_outs:
        all_model_names += res[0]
        all_mean_values += res[1]
        all_std_values += res[2]
        all_splits += [""] * len(res[0])

    with open(os.path.join(WORKING_DIR, "kl_all_lists.pkl"), "wb") as b:
        pickle.dump([all_model_names, all_mean_values, all_std_values, all_splits], b, pickle.HIGHEST_PROTOCOL)

    process_map_outs = []
    for res in process_map(compute_metric, dihedral_list_to_analyze, ["EMD"] * len(dihedral_list_to_analyze)):
        process_map_outs.append(res)

    all_model_names = []
    all_mean_values = []
    all_std_values = []
    all_splits = []
    for res in process_map_outs:
        all_model_names += res[0]
        all_mean_values += res[1]
        all_std_values += res[2]
        all_splits += [""] * len(res[0])

    with open(os.path.join(WORKING_DIR, "emd_dihedral.pkl"), "wb") as b:
        pickle.dump([all_model_names, all_mean_values, all_std_values, all_splits], b, pickle.HIGHEST_PROTOCOL)

    process_map_outs = []
    for res in process_map(compute_metric, list_to_analyze, ["EMD"] * len(list_to_analyze)):
        process_map_outs.append(res)

    all_model_names = []
    all_mean_values = []
    all_std_values = []
    all_splits = []
    for res in process_map_outs:
        all_model_names += res[0]
        all_mean_values += res[1]
        all_std_values += res[2]
        all_splits += res[3]

    with open(os.path.join(WORKING_DIR, "mae_emd_per_molecule_all_lists.pkl"), "wb") as b:
        pickle.dump([all_model_names, all_mean_values, all_std_values, all_splits], b, pickle.HIGHEST_PROTOCOL)

    process_map_outs = []
    for res in process_map(compute_metric, dist_to_analyze, ["MAE"] * len(dist_to_analyze)):
        process_map_outs.append(res)

    all_model_names = []
    all_mean_values = []
    all_std_values = []
    all_splits = []
    for res in process_map_outs:
        all_model_names += res[0]
        all_mean_values += res[1]
        all_std_values += res[2]
        all_splits += res[3]

    with open(os.path.join(WORKING_DIR, "mae_expected_distance_per_molecule_all_lists.pkl"), "wb") as b:
        pickle.dump([all_model_names, all_mean_values, all_std_values, all_splits], b, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    run_analysis()
