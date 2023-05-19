""" Pipeline for computing expectations. """
import argparse
from glob import glob
import os
import warnings

from ruffus import *
import sqlitedict
from vonmises.confdists import *

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str)
parser.add_argument("--working_dir", type=str)
parser.add_argument("--restrict_to_vm_preds", action='store_true')
parser.add_argument("--restrict_to_non_ring", action='store_true')
parser.add_argument("--restrict_to_non_aromatic_ring", action='store_true')
parser.add_argument("--num_workers", type=int, default=8)
args = parser.parse_args()

INPUT_DIR = args.input_dir
WORKING_DIR = args.working_dir
if args.restrict_to_vm_preds:
    RESTRICT_TO_VM_PREDS = True
else:
    RESTRICT_TO_VM_PREDS = False
if args.restrict_to_non_ring:
    RESTRICT_TO_NON_RING = True
else:
    RESTRICT_TO_NON_RING = False
if args.restrict_to_non_aromatic_ring:
    RESTRICT_TO_NON_AROMATIC_RING = True
else:
    RESTRICT_TO_NON_AROMATIC_RING = False


def td(x):
    """
    Define path with WORKING_DIR.

    :param x: Local path.
    :return: Path joined with WORKING_DIR.
    """
    return os.path.join(WORKING_DIR, x)


input_distributions = {

}

for f in glob(f"{INPUT_DIR}/*.db"):
    f_db = os.path.splitext(os.path.basename(f))[0]
    print('possible infile', f_db)
    input_distributions[f_db] = {'filename': f}

possible_expectations = {
    'hist_dihedral_32': {'func': 'compute_dihedral_hist',
                         'args': {'bins': 32}},

    'list_dihedral': {'func': 'compute_dihedral_list'},

    'average_distance_1-2': {'func': 'compute_distance_avg_path_lim',
                             'args': {'min_atom_path_len': 2,
                                      'max_atom_path_len': 2,
                                      'restrict_to_vm_preds': RESTRICT_TO_VM_PREDS,
                                      'restrict_to_non_ring': RESTRICT_TO_NON_RING,
                                        'restrict_to_non_aromatic_ring': RESTRICT_TO_NON_AROMATIC_RING}},
    'average_distance_1-3': {'func': 'compute_distance_avg_path_lim',
                             'args': {'min_atom_path_len': 3,
                                      'max_atom_path_len': 3,
                                      'restrict_to_vm_preds': RESTRICT_TO_VM_PREDS,
                                      'restrict_to_non_ring': RESTRICT_TO_NON_RING,
                                        'restrict_to_non_aromatic_ring': RESTRICT_TO_NON_AROMATIC_RING}},
    'average_distance_1-4': {'func': 'compute_distance_avg_path_lim',
                             'args': {'min_atom_path_len': 4,
                                      'max_atom_path_len': 4,
                                      'restrict_to_vm_preds': RESTRICT_TO_VM_PREDS,
                                      'restrict_to_non_ring': RESTRICT_TO_NON_RING,
                                        'restrict_to_non_aromatic_ring': RESTRICT_TO_NON_AROMATIC_RING}},

    'average_distance_1-5': {'func': 'compute_distance_avg_path_lim',
                             'args': {'min_atom_path_len': 5,
                                      'max_atom_path_len': 5,
                                      'restrict_to_vm_preds': RESTRICT_TO_VM_PREDS,
                                      'restrict_to_non_ring': RESTRICT_TO_NON_RING,
                                        'restrict_to_non_aromatic_ring': RESTRICT_TO_NON_AROMATIC_RING}},

    'average_distance_1-6': {'func': 'compute_distance_avg_path_lim',
                             'args': {'min_atom_path_len': 6,
                                      'max_atom_path_len': 6,
                                      'restrict_to_vm_preds': RESTRICT_TO_VM_PREDS,
                                      'restrict_to_non_ring': RESTRICT_TO_NON_RING,
                                        'restrict_to_non_aromatic_ring': RESTRICT_TO_NON_AROMATIC_RING}},
    'average_distance_1-7': {'func': 'compute_distance_avg_path_lim',
                             'args': {'min_atom_path_len': 7,
                                      'max_atom_path_len': 7,
                                      'restrict_to_vm_preds': RESTRICT_TO_VM_PREDS,
                                      'restrict_to_non_ring': RESTRICT_TO_NON_RING,
                                        'restrict_to_non_aromatic_ring': RESTRICT_TO_NON_AROMATIC_RING}},
    'average_distance_1-8': {'func': 'compute_distance_avg_path_lim',
                             'args': {'min_atom_path_len': 8,
                                      'max_atom_path_len': 8,
                                      'restrict_to_vm_preds': RESTRICT_TO_VM_PREDS,
                                      'restrict_to_non_ring': RESTRICT_TO_NON_RING,
                                        'restrict_to_non_aromatic_ring': RESTRICT_TO_NON_AROMATIC_RING}},
    'average_distance_1-9': {'func': 'compute_distance_avg_path_lim',
                             'args': {'min_atom_path_len': 9,
                                      'max_atom_path_len': 9,
                                      'restrict_to_vm_preds': RESTRICT_TO_VM_PREDS,
                                      'restrict_to_non_ring': RESTRICT_TO_NON_RING,
                                        'restrict_to_non_aromatic_ring': RESTRICT_TO_NON_AROMATIC_RING}},
    'average_distance_1-10': {'func': 'compute_distance_avg_path_lim',
                              'args': {'min_atom_path_len': 10,
                                       'max_atom_path_len': 10,
                                       'restrict_to_vm_preds': RESTRICT_TO_VM_PREDS,
                                       'restrict_to_non_ring': RESTRICT_TO_NON_RING,
                                        'restrict_to_non_aromatic_ring': RESTRICT_TO_NON_AROMATIC_RING}},
    'dist_list_1-4': {'func': 'compute_distance_lists',
                      'args': {'min_atom_path_len': 4,
                               'max_atom_path_len': 4,
                               'restrict_to_vm_preds': RESTRICT_TO_VM_PREDS,
                               'restrict_to_non_ring': RESTRICT_TO_NON_RING,
                                        'restrict_to_non_aromatic_ring': RESTRICT_TO_NON_AROMATIC_RING}
                      },
    'dist_list_1-5': {'func': 'compute_distance_lists',
                      'args': {'min_atom_path_len': 5,
                               'max_atom_path_len': 5,
                               'restrict_to_vm_preds': RESTRICT_TO_VM_PREDS,
                               'restrict_to_non_ring': RESTRICT_TO_NON_RING,
                                        'restrict_to_non_aromatic_ring': RESTRICT_TO_NON_AROMATIC_RING}
                      },
    'dist_list_1-6': {'func': 'compute_distance_lists',
                      'args': {'min_atom_path_len': 6,
                               'max_atom_path_len': 6,
                               'restrict_to_vm_preds': RESTRICT_TO_VM_PREDS,
                               'restrict_to_non_ring': RESTRICT_TO_NON_RING,
                                        'restrict_to_non_aromatic_ring': RESTRICT_TO_NON_AROMATIC_RING}
                      },
    'dist_list_1-7': {'func': 'compute_distance_lists',
                      'args': {'min_atom_path_len': 7,
                               'max_atom_path_len': 7,
                               'restrict_to_vm_preds': RESTRICT_TO_VM_PREDS,
                               'restrict_to_non_ring': RESTRICT_TO_NON_RING,
                                        'restrict_to_non_aromatic_ring': RESTRICT_TO_NON_AROMATIC_RING}
                      },
    'dist_list_1-8': {'func': 'compute_distance_lists',
                      'args': {'min_atom_path_len': 8,
                               'max_atom_path_len': 8,
                               'restrict_to_vm_preds': RESTRICT_TO_VM_PREDS,
                               'restrict_to_non_ring': RESTRICT_TO_NON_RING,
                                        'restrict_to_non_aromatic_ring': RESTRICT_TO_NON_AROMATIC_RING}
                      },
    'dist_list_1-9': {'func': 'compute_distance_lists',
                      'args': {'min_atom_path_len': 9,
                               'max_atom_path_len': 9,
                               'restrict_to_vm_preds': RESTRICT_TO_VM_PREDS,
                               'restrict_to_non_ring': RESTRICT_TO_NON_RING,
                                        'restrict_to_non_aromatic_ring': RESTRICT_TO_NON_AROMATIC_RING}
                      },
    'dist_list_1-10': {'func': 'compute_distance_lists',
                       'args': {'min_atom_path_len': 10,
                                'max_atom_path_len': 10,
                                'restrict_to_vm_preds': RESTRICT_TO_VM_PREDS,
                                'restrict_to_non_ring': RESTRICT_TO_NON_RING,
                                        'restrict_to_non_aromatic_ring': RESTRICT_TO_NON_AROMATIC_RING}
                       },
}

to_compute = [
    ('*', 'hist_dihedral_32'),
    ('*', 'list_dihedral'),
    ('*', 'average_distance_1-2'),
    ('*', 'average_distance_1-3'),
    ('*', 'average_distance_1-4'),
    ('*', 'average_distance_1-5'),
    ('*', 'average_distance_1-6'),
    ('*', 'average_distance_1-7'),
    ('*', 'average_distance_1-8'),
    ('*', 'average_distance_1-9'),
    ('*', 'average_distance_1-10'),
    ('*', 'dist_list_1-4'),
    ('*', 'dist_list_1-5'),
    ('*', 'dist_list_1-6'),
    ('*', 'dist_list_1-7'),
    ('*', 'dist_list_1-8'),
    ('*', 'dist_list_1-9'),
    ('*', 'dist_list_1-10')
]


def params():
    """
    Pipeline parameters.
    """
    for distribution_name, expectation_name in to_compute:
        if distribution_name == '*':
            distribution_names = list(input_distributions.keys())
        else:
            distribution_names = [distribution_name]
        for dn in distribution_names:
            dist_config = input_distributions[dn]

            outfile = td(f"{dn}__{expectation_name}.expect")

            infile = dist_config['filename']
            yield infile, outfile, possible_expectations[expectation_name]


@mkdir(WORKING_DIR)
@files(params)
def compute_expectation(infile, outfile, expectation_config):
    """
    Compute expectations.

    :param infile: Input file name.
    :param outfile: Output file name.
    :param expectation_config: Expectation config.
    """
    print("computing", outfile)
    dist_dict = sqlitedict.SqliteDict(infile, flag='r')

    out_dict = sqlitedict.SqliteDict(outfile)

    expectation_func = eval(expectation_config['func'])
    args = expectation_config.get('args', {})
    for k, v in dist_dict.items():
        try:
            out_dict[k] = {"expectation_results": expectation_func(v, **args), "mol": v.mol}
        except Exception as e:
            warnings.warn(f"Could not complete lookup {e}")

    out_dict.commit()


if __name__ == "__main__":
    pipeline_run([compute_expectation], multiprocess=args.num_workers)
