"""
Pipeline for turning distributions into expectations, with
the recognition that computing some expectations may take
a while.
"""

import os
from glob import glob
import sqlitedict
from rdkit import Chem
import matplotlib.pyplot as plt
import numpy as np
from vonmises import mol_utils
from tqdm.contrib.concurrent import process_map
from rdkit.Chem import rdForceFieldHelpers
from rdkit.ForceField import rdForceField
from vonmises.mol_utils import compute_rotatable_bond_torsions, compute_chirality_all_confs
import seaborn as sns
import math

WORKING_DIR = "conformation.searching-for-chirality"
MAX_WORKERS = 20
PLOT_DIAG = False
PLOT_MMFF = False

input_distributions = {

}

for f in glob("conformation.dists-nmrshiftdb-fix-chirality-20220924-renamed/*.db"):
    f_db = os.path.splitext(os.path.basename(f))[0]

    input_distributions[f_db] = {'filename': f}


def generate_plot(mol_id, mol, infile):
    """
    Generate plot.
    """
    df = mol_utils.compute_rotatable_bond_torsions(mol)
    if not df.empty:
        g = mol_utils.plot_torsion_joint_histograms(df, plot_diag=PLOT_DIAG)
        g.figure.savefig(os.path.join(WORKING_DIR, f"{mol_id}.{os.path.split(infile)[-1]}.png"))
        plt.close()

    if PLOT_MMFF:
        mmff_p = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol)
        force_field = rdForceFieldHelpers.MMFFGetMoleculeForceField(mol, mmff_p)
        mmff_energies = []
        for conf in mol.GetConformers():
            pos = conf.GetPositions()
            pos = tuple(pos.flatten())
            energy = force_field.CalcEnergy(pos)
            mmff_energies.append(energy)

        g = sns.displot(mmff_energies)
        g.figure.savefig(os.path.join(WORKING_DIR, f"{mol_id}.{os.path.split(infile)[-1]}_mmff.png"))
        plt.close()


def generate_plots():
    """
    Generate all plots.
    """
    os.makedirs(WORKING_DIR, exist_ok=True)
    for dist_config in input_distributions.values():
        infile = dist_config['filename']
        if "Diffusion" in infile:
            db = sqlitedict.SqliteDict(infile, flag='r')
            mol_ids = []
            for mol_id, value in db.items():
                if value is not None:
                    mol_ids.append(mol_id)

    for dist_config in input_distributions.values():
        infile = dist_config['filename']
        db = sqlitedict.SqliteDict(infile, flag='r')
        mols = []
        for mol_id, value in db.items():
            if mol_id in mol_ids:
                if value is not None:
                    mols.append(value.mol)

        for _ in zip(mol_ids, process_map(generate_plot, mol_ids, mols, [infile] * len(mols), max_workers=MAX_WORKERS,
                                          chunksize=1)):
            continue

    samples_db = sqlitedict.SqliteDict("/data/swansonk1/vonmises-folder/conformation-vonmises-20220924/"
                                       "conformation.dists-nmrshiftdb-fix-chirality-20220924-renamed/samples.db",
                                       flag="r")
    mol = samples_db['861'].mol
    df_full = compute_rotatable_bond_torsions(mol)

    test = compute_chirality_all_confs(mol, 9)
    tmp = Chem.Mol(mol)
    tmp.RemoveAllConformers()
    for idx in range(mol.GetNumConformers()):
        if test[idx] > 0:
            tmp.AddConformer(mol.GetConformers()[idx])
    df_positive = compute_rotatable_bond_torsions(tmp)

    tmp = Chem.Mol(mol)
    tmp.RemoveAllConformers()
    for idx in range(mol.GetNumConformers()):
        if test[idx] < 0:
            tmp.AddConformer(mol.GetConformers()[idx])
    df_negative = compute_rotatable_bond_torsions(tmp)

    sns.histplot(df_positive['1-9 | C N'].to_numpy(), bins=list(np.arange(-math.pi - 1., math.pi + 1., 0.1)),
                 stat='density', element='step', label='R')
    sns.histplot(df_negative['1-9 | C N'].to_numpy(), bins=list(np.arange(-math.pi - 1., math.pi + 1., 0.1)),
                 stat='density', element='step', label='S')
    plt.legend()
    plt.ylabel("Density")
    plt.xlabel("Angle")
    plt.xlim((-math.pi, math.pi))
    plt.savefig(os.path.join(WORKING_DIR, "chirality_comparison.png"))
    plt.clf()
    plt.cla()
    plt.close()

    vm_db = sqlitedict.SqliteDict("/data/swansonk1/vonmises-folder/conformation-vonmises-20220924/"
                                  "conformation.dists-nmrshiftdb-fix-chirality-20220924-renamed/VonMisesNet.db",
                                  flag="r")
    mol = vm_db['861'].mol
    df_vm = compute_rotatable_bond_torsions(mol)

    torsional_db = sqlitedict.SqliteDict("/data/swansonk1/vonmises-folder/conformation-vonmises-20220924/"
                                         "conformation.dists-nmrshiftdb-fix-chirality-20220924-renamed/"
                                         "TorsionalDiffusion.db", flag="r")
    mol = torsional_db['861'].mol
    df_torsional = compute_rotatable_bond_torsions(mol)

    etkdg_db = sqlitedict.SqliteDict("/data/swansonk1/vonmises-folder/conformation-vonmises-20220924/"
                                         "conformation.dists-nmrshiftdb-fix-chirality-20220924-renamed/"
                                         "ETKDG-Clean.db", flag="r")
    mol = etkdg_db['861'].mol
    df_etkdg = compute_rotatable_bond_torsions(mol)

    geomol_db = sqlitedict.SqliteDict("/data/swansonk1/vonmises-folder/conformation-vonmises-20220924/"
                                         "conformation.dists-nmrshiftdb-fix-chirality-20220924-renamed/"
                                         "GeoMol.db", flag="r")
    mol = geomol_db['861'].mol
    df_geomol = compute_rotatable_bond_torsions(mol)

    sns.kdeplot(df_full['1-9 | C N'].to_numpy(), label='PT-HMC', bw_adjust=0.25)
    sns.kdeplot(df_vm['1-9 | C N'].to_numpy(), label='VonMisesNet', bw_adjust=0.25)
    sns.kdeplot(df_torsional['1-9 | C N'].to_numpy(), label='TorsionalDiffusion', bw_adjust=0.25)
    sns.kdeplot(df_etkdg['1-9 | C N'].to_numpy(), label='ETKDG-Clean', bw_adjust=0.25)
    sns.kdeplot(df_geomol['1-9 | C N'].to_numpy(), label='GeoMol', bw_adjust=0.25)
    plt.legend()
    plt.ylabel("Density")
    plt.xlabel("Angle")
    plt.xlim(-math.pi, math.pi)
    plt.savefig(os.path.join(WORKING_DIR, "chirality_comparison_VM_vs_diffusion.png"))


if __name__ == "__main__":
    generate_plots()
