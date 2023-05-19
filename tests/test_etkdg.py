"""
Tests for cleaned-up ETKDG conformer generation. 
We know that etkdg does not do a good job of preserving
atom identities in situations with symmetry. By adding
various chiral constraints we can fix that. Here we test to make sure
that rigid molecules have no intra-atom distance multimodality

"""

import pytest
from rdkit import Chem
import rdkit.Chem.AllChem
from vonmises import etkdg
from vonmises.mol_utils import get_all_conf_pos, get_conf_dist
import numpy as np

def assert_unimodal_distances(mol_with_confs, mean_median_delta_diff = 0.1):

    etkdg_pos = get_all_conf_pos(mol_with_confs)
    etkdg_dists = get_conf_dist(etkdg_pos)

    for i in range(mol_with_confs.GetNumAtoms()):
        for j in range(i + 1, mol_with_confs.GetNumAtoms()):
            d = etkdg_dists[:, i, j]
            mean_median_delta =  np.abs(np.mean(d)- np.median(d))
            #print(mean_median_delta)
            if mean_median_delta > mean_median_delta_diff:
                raise Exception(f"atoms {i} and {j} have a distance distribution where mean-median delta={mean_median_delta}")




TARGET_SMILES = {'cyclopropane' : 'C1CC1',
                 'chloroethane' : 'F/C=C',
                 'n1' : "C=N"
                 #'dichloroethane' : 'F/C=C/F'
}




def test_clean_etkdg():

    
    for name, smiles in TARGET_SMILES.items():
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        

        mol_with_confs = etkdg.generate_etkdg_confs(mol, 4000)

        with pytest.raises(Exception):
        
            assert_unimodal_distances(mol_with_confs)
        
    
        mol_with_confs = etkdg.generate_clean_etkdg_confs(mol, 4000)
        assert_unimodal_distances(mol_with_confs)    


EDGE_CASES = {'formaldehyde' : "CO", # double-bonded O
              'other': "[H]C(=O)[C@]1(C(=O)c2nc(C#N)nn2[H])N([H])C([H])=N[C@@]1([H])C([H])([H])[H]", # C=N and C=O

              'n1' : "C=N"
              } 

def test_edge_cases():
    """

    """

    for name, smiles in EDGE_CASES.items():
        
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))


        mol_with_single_conf = etkdg.generate_etkdg_confs(mol, 1)


        mol_many_confs = etkdg.generate_clean_etkdg_confs(mol_with_single_conf, 1000,
                                                          conform_to_existing_conf_idx=0)

        # just make sure there's no segfault

        

    
def test_preserve_existing_double_chi():
    """

    """
    for smiles_name in ['chloroethane', 'n1']:
        smiles = TARGET_SMILES[smiles_name]

        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))

        mol_with_single_conf = etkdg.generate_etkdg_confs(mol, 1)

        etkdg_pos = get_all_conf_pos(mol_with_single_conf)

        etkdg_dists_orig = get_conf_dist(etkdg_pos)

        mol_many_confs = etkdg.generate_clean_etkdg_confs(mol_with_single_conf, 1000,
                                                          conform_to_existing_conf_idx=0)

        etkdg_pos = get_all_conf_pos(mol_many_confs)
        etkdg_dists = get_conf_dist(etkdg_pos)

        for i in range(mol.GetNumAtoms()):
            for j in range(i +1, mol.GetNumAtoms()):

                delta = np.mean(etkdg_dists[:, i, j]) - etkdg_dists_orig[0,i, j]

                assert np.abs(delta) < 0.1


def test_preserve_existing_tetra_chi():
    """

    """
    smiles = TARGET_SMILES['cyclopropane']

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    
    mol_with_single_conf = etkdg.generate_etkdg_confs(mol, 1)
    
    etkdg_pos = get_all_conf_pos(mol_with_single_conf)
    
    etkdg_dists_orig = get_conf_dist(etkdg_pos)
    
    mol_many_confs = etkdg.generate_clean_etkdg_confs(mol_with_single_conf, 1000,
                                                      conform_to_existing_conf_idx=0)
    
    etkdg_pos = get_all_conf_pos(mol_many_confs)
    etkdg_dists = get_conf_dist(etkdg_pos)

    for i in range(mol.GetNumAtoms()):
        for j in range(i +1, mol.GetNumAtoms()):
            
            delta = np.mean(etkdg_dists[:, i, j]) - etkdg_dists_orig[0,i, j]
            
            assert np.abs(delta) < 0.16
            


PREVIOUS_BUGS = {'jake_1' :  "[H]c1c([H])c([H])c(C(N=C=NC(C([H])([H])[H])(C([H])([H])[H])C([H])([H])[H])(c2c([H])c([H])c([H])c([H])c2[H])c2c([H])c([H])c([H])c([H])c2[H])c([H])c1[H]",
                 'jake_1_bin' : b'\xef\xbe\xad\xde\x00\x00\x00\x00\r\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x002\x00\x00\x004\x00\x00\x00\x80\x01\x06@(\x00\x00\x00\x03\x04\x06\x00 \x00\x00\x00\x04\x06\x00(\x00\x00\x00\x02\x04\x06\x00 \x00\x00\x00\x04\x06\x00 \x00\x00\x00\x04\x06@(\x00\x00\x00\x03\x04\x06@(\x00\x00\x00\x03\x04\x06\x00 \x00\x00\x00\x04\x06\x00 \x00\x00\x00\x04\x06@(\x00\x00\x00\x03\x04\x06@(\x00\x00\x00\x03\x04\x06@(\x00\x00\x00\x03\x04\x06@(\x00\x00\x00\x03\x04\x06@(\x00\x00\x00\x03\x04\x06@(\x00\x00\x00\x03\x04\x06@(\x00\x00\x00\x03\x04\x06@(\x00\x00\x00\x03\x04\x06@(\x00\x00\x00\x03\x04\x06@(\x00\x00\x00\x03\x04\x06@(\x00\x00\x00\x03\x04\x06@(\x00\x00\x00\x03\x04\x06@(\x00\x00\x00\x03\x04\x06@(\x00\x00\x00\x03\x04\x06@(\x00\x00\x00\x03\x04\x07\x00(\x00\x00\x00\x03\x03\x07\x00(\x00\x00\x00\x03\x03\x01\x00(\x00\x00\x00\x01\x01\x01\x00(\x00\x00\x00\x01\x01\x01\x00(\x00\x00\x00\x01\x01\x01\x00(\x00\x00\x00\x01\x01\x01\x00(\x00\x00\x00\x01\x01\x01\x00(\x00\x00\x00\x01\x01\x01\x00(\x00\x00\x00\x01\x01\x01\x00(\x00\x00\x00\x01\x01\x01\x00(\x00\x00\x00\x01\x01\x01\x00(\x00\x00\x00\x01\x01\x01\x00(\x00\x00\x00\x01\x01\x01\x00(\x00\x00\x00\x01\x01\x01\x00(\x00\x00\x00\x01\x01\x01\x00(\x00\x00\x00\x01\x01\x01\x00(\x00\x00\x00\x01\x01\x01\x00(\x00\x00\x00\x01\x01\x01\x00(\x00\x00\x00\x01\x01\x01\x00(\x00\x00\x00\x01\x01\x01\x00(\x00\x00\x00\x01\x01\x01\x00(\x00\x00\x00\x01\x01\x01\x00(\x00\x00\x00\x01\x01\x01\x00(\x00\x00\x00\x01\x01\x01\x00(\x00\x00\x00\x01\x01\x01\x00(\x00\x00\x00\x01\x01\x0b\x00\x01\x00\x01\x18\x00\x18\x02(\x02\x02\x19(\x02\x19\x03\x00\x03\x04\x00\x01\x05\x00\x01\x06\x00\x03\x07\x00\x03\x08\x00\x05\x0ch\x0c\x0b\x05h\x0c\t\nh\x0c\t\rh\x0c\n\x0bh\x0c\x0c\rh\x0c\x0e\x00h\x0c\x12\x00h\x0c\x0e\x0fh\x0c\x0f\x10h\x0c\x10\x11h\x0c\x11\x12h\x0c\x06\x17h\x0c\x06\x13h\x0c\x13\x14h\x0c\x14\x15h\x0c\x15\x16h\x0c\x16\x17h\x0c\x0e\x1a\x00\x12\x1b\x00\x0f\x1c\x00\x11\x1d\x00\x0c\x1e\x00\x0b\x1f\x00\x17 \x00\x13!\x00\x10"\x00\r#\x00\n$\x00\x16%\x00\x14&\x00\t\'\x00\x15(\x00\x04)\x00\x04*\x00\x04+\x00\x07,\x00\x07-\x00\x07.\x00\x08/\x00\x080\x00\x081\x00\x14\x03\x06\t\r\x0c\x05\x0b\n\x06\x0e\x0f\x10\x11\x12\x00\x06\x13\x14\x15\x16\x17\x06\x17b\x02\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x002\\ \x01>\x03\t\xca?)\\\xcf>B>X\xbf\x1c|\xe1>/nc>\t\xf9\xa0\xbe\x00o\xd1\xbf\xc3\xf5\x98?\x88\xf4\x9b>\x1d8\x17\xc0\x0e-V@\xf7u\x80>.\x90p\xbfD\xfay@B\xcf\xee\xbf\xce\xaa\xef>\xfeC\xaa?\xdfO\xc5\xbf\x0bF\x15?\xcb\xa1\x8d\xbf\xdch\xe0?;p.\xc0\xd3\xde@@\xb4Yu\xbe#\xdbU\xc0\xb0r\x8c@#Jo\xc0\xd8\xf0\x04?d;W@\x0c\x02\x81\xc0B>\xe8\xbd\xd0\xd5\n@{\x14F\xc0\x96!\x0e\xbe\xf3\x8e\x93?\xf9\x0f\xc9\xbf\x116\x8c?C\xad!@\x9aw \xc0`v\x8f?\x99\xbbb@T\xe3\xbd?\r\xe0\xad?\x94\xf6\x86>\x9e\xef\x17@H\xbf\x19@v\xe0\xdc>\xda\x1b\xf4?\xa9\xa4j@2w=?$\xb9\x0c?j\xbcx@sha?\xc4\xb1\xae\xbe\xe7\xfb5@\xb4Y5?\x19\xe2\x1c\xc0\xa4p\xcd?]\xdc\xa6\xbf\x1d\xc9E\xc0\xf91\xde?]m!\xc0\rq4\xc0yXX?\x89\xd2b\xc0\xd6V\xf4\xbf\xc6m4\xbe*:V\xc0\xf3\x1f\xa2\xbfm\xe7\x9b\xbe\xecQ\x08\xc0\xfc\x18\xf3\xbd/\xddT\xbf8g\x84>\\\x8f\x02\xbf\x116\x1c\xc08g\x08@\x18\x95\xec?\x9b\xe6\xbd>\xecQ\xb8<\xbct\xb3\xbf\x0f\x0bA@\x12\x83P?}\xd0[@<\xbd\x0e@\xdd\xb5\xa4>mV=>\xfd\xf6\x9b@\xf9\xa0\x8f?6\xcd\x1b\xbf\xc0[\xc8?\x93\xa9*@\xc7\xbaT\xc0\xf7\xe4!\xbfw-a>\xa4p\r\xbf\xa1\xd6\x8c\xbfq\x1b\xfd\xbf;p*\xc0\xf2A\x13@W[\x01\xbf\xea\x95&@\x16j\x8f@\xd74_?9\xd6\x11\xc0\x99\xbb\xce?\xa85\x8f@\x92\xcb\x9f\xc0\xb4Y\x15\xbf\x7f\xd9\x01@\xaeG\xd9\xbf\x9bU_\xbf\xf0\xa7\x84\xc0\xb1Ps\xc0\xc7)"@9E+\xc0V\x0e\x8f\xc0u\x02\n?\xc8\x07\x85@ioT\xc0\x98nr?;\xdf\x8f\xc0\xcc\xeeI\xbf*\xa9#\xbf\xa3#\x81@\xe5\xd0B?Y\x86h\xbfd\xcc\x9b@_\x07>?A\xf1\x83\xbeh\x91M@\x81\x04\t@\xab\xcf\x01\xc0mV\x11@\xc5\xfe\x16@R\xb8*\xc0\x99\xbbz@\x98L\xe5?TRo\xc0)\\\'@\xcd;N\xbe\x9e^\x8b\xc0Zd\x7f@\xe3\xa5\xbb>:#R\xc0\x13a\xa9@\xca\xc3\xa2\xbf\xdfOE\xc0\xd5\t\x94@\x16'
                 }


import pickle

def test_clean_runs():
    """
    Just test if we can successfully run on these molecules
    """

    for name, smiles in PREVIOUS_BUGS.items():
        if isinstance(smiles, bytes):
            mol = Chem.Mol(smiles)
        else:
            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        
        mol_with_confs = etkdg.generate_clean_etkdg_confs(mol, 100)

        if len(mol.GetConformers() ) > 0:
            mol_with_conf = etkdg.generate_clean_etkdg_confs(mol, num=1, conform_to_existing_conf_idx=0)
