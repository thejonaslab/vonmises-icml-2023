""" Generate timing analysis. """
from glob import glob
import os

import pandas as pd
from rdkit.Chem.Lipinski import RotatableBondSmarts
import sqlitedict

from generate_expectations_analysis_pipeline import generate_performance_metrics

DIRECTORIES = [["results/conformation.dists-nmrshiftdb-timing-cpu",
                "results/conformation.dists-nmrshiftdb-timing-cpu-analysis"],
               ["results/conformation.dists-nmrshiftdb-timing-gpu",
                "results/conformation.dists-nmrshiftdb-timing-gpu-analysis"]]


def generate_time_analysis():
    """
    Generate timing analysis.
    """
    for d in DIRECTORIES:
        input_dir = d[0]
        working_dir = d[1]

        input_distributions = {}
        for f in glob(f"{input_dir}/*.db"):
            f_db = os.path.splitext(os.path.basename(f))[0]
            input_distributions[f_db] = {'filename': f}

        os.makedirs(working_dir, exist_ok=True)
        data_frames = []
        exp_names = []
        for dist_config in input_distributions.values():
            data_frame = []
            infile = dist_config['filename']
            db = sqlitedict.SqliteDict(infile, flag='r')
            for mol_id, value in db.items():
                mol = value.mol
                rotatable_bonds = mol.GetSubstructMatches(RotatableBondSmarts)
                num_rotatable_bonds = len(rotatable_bonds)
                if len(value.meta) > 0:
                    data_frame.append([mol_id, num_rotatable_bonds, value.meta['runtime'] / mol.GetNumConformers()])
            if len(data_frame) > 0:
                data_frame = pd.DataFrame(data_frame)
                data_frame.columns = ["mol_id", "num_rot_bonds", os.path.split(infile)[-1]]
                data_frames.append(data_frame)
                exp_names.append(os.path.split(infile)[-1])

        if len(data_frames) > 0:
            df = data_frames[0]
            for i in range(1, len(data_frames)):
                df = pd.merge(df, data_frames[i], on=["mol_id", "num_rot_bonds"])

            new_df = generate_performance_metrics(df.drop(["mol_id", "num_rot_bonds"], axis=1))
            new_df.to_pickle(os.path.join(working_dir, "runtime_per_conformer.pkl"))

            new_df = df.groupby(["mol_id", "num_rot_bonds"]).mean().groupby("num_rot_bonds").mean()
            new_df.columns = [x[:-3] for x in new_df.columns]
            new_df.to_pickle(os.path.join(working_dir, "runtime_per_conformer_by_num_rot_bonds.pkl"))


if __name__ == "__main__":
    generate_time_analysis()
