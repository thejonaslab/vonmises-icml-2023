""" Class to generate Von Mises predictions from mol objects. """
import math
import random
import time
from typing import Dict, List, Tuple, Union
import yaml

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdchem, rdMolTransforms
from rdkit.Chem.Lipinski import RotatableBondSmarts
import torch
from torch.distributions.von_mises import VonMises
from tqdm import tqdm

from vonmises import etkdg
from vonmises.featurize_mols import FeaturizeMol
from vonmises.graph_builders import BuildVonMisesGraph
from vonmises.nets import VonMisesNet
from vonmises.mol_utils import compute_chirality, compute_chirality_probability_targets, compute_lengths, \
    compute_angles, compute_torsions, flip_chirality, filter_conformation


def set_torsion_angle_from_predictions(mol: rdchem.Mol, conformation_index: int, num_von_mises: int,
                                       loc_preds: torch.Tensor, conc_preds: torch.Tensor, weight_preds: torch.Tensor,
                                       torsion_angle_atom_quartet: Tuple[int]):
    """
    Set a torsion angle by sampling from predicted von Mises mixture distribution.

    :param mol: RDKit mol object.
    :param conformation_index: Conformation index in mol.
    :param num_von_mises: Number of von Mises distributions in the mixture.
    :param loc_preds: von Mises mean predictions.
    :param conc_preds: von Mises concentration predictions.
    :param weight_preds: von Mises weight predictions.
    :param torsion_angle_atom_quartet: Quartet of atom indices defining the torsion angle.
    """
    von_mises_idx = np.random.choice(range(num_von_mises), p=torch.exp(weight_preds).cpu().numpy())
    angle = VonMises(torch.tensor([loc_preds[von_mises_idx]]),
                     torch.tensor([conc_preds[von_mises_idx]])).sample().item()
    rdMolTransforms.SetDihedralRad(mol.GetConformers()[conformation_index],
                                   int(torsion_angle_atom_quartet[0]),
                                   int(torsion_angle_atom_quartet[1]),
                                   int(torsion_angle_atom_quartet[2]),
                                   int(torsion_angle_atom_quartet[3]),
                                   angle)


def generate_confs(predictions: Dict, num_confs: int, filter_confs: bool = False, max_attempts: int = 5) -> \
        rdchem.Mol:
    """
    Generate conformations from VonMisesNet predictions.

    :param predictions: VonMisesNet predictions dictionary.
    :param num_confs: Number of conformations to generate per molecule.
    :param filter_confs: Whether or not to perform filtering based on inter-atomic distances.
    :param max_attempts: Maximum number of attempts to generate a single conformation.
    :return: RDKit mol object with generated conformations embedded.
    """
    # Transform von Mises distribution mean predictions into [-pi, pi]
    loc_preds = predictions["loc_preds"] % (2.0 * math.pi) - math.pi
    chiral_loc_preds_pos = predictions["chiral_loc_preds_pos"] % (2.0 * math.pi) - math.pi
    chiral_loc_preds_neg = predictions["chiral_loc_preds_neg"] % (2.0 * math.pi) - math.pi

    # Create output and temporary mol objects
    mol = Chem.Mol(predictions["mol"])
    mol_tmp = Chem.Mol(mol)
    mol_tmp.RemoveAllConformers()
    mol_tmp.AddConformer(mol.GetConformers()[0])
    mol_tmp_tmp = Chem.Mol(mol_tmp)
    mol.RemoveAllConformers()

    mol_tmp = etkdg.generate_clean_etkdg_confs(mol_tmp, 1, conform_to_existing_conf_idx=0,
                                               exception_for_num_failure=False, max_embed_attempts=10)
    if mol_tmp.GetNumConformers() == 0:
        print("Failed to generate confs!")
        mol_tmp = mol_tmp_tmp

    # Compute rotatable bonds
    rotatable_bonds = mol_tmp.GetSubstructMatches(RotatableBondSmarts)
    rotatable_bond_idxs = [mol_tmp.GetBondBetweenAtoms(x[0], x[1]).GetIdx() for x in rotatable_bonds]

    assert mol_tmp.GetNumConformers() == 1, "Input mol must have at least 1 conformation!"

    # Get number of von Mises distributions per rotatable bond
    num_von_mises = loc_preds.shape[1]

    # Set bond lengths for all non-ring bonds
    for idx, bond in enumerate(mol_tmp.GetBonds()):
        if not bond.IsInRing():
            Chem.rdMolTransforms.SetBondLength(mol_tmp.GetConformers()[0],
                                               bond.GetBeginAtom().GetIdx(),
                                               bond.GetEndAtom().GetIdx(),
                                               predictions["bond_preds"][idx].item())

    # Set bond angles for all bond angles for which at least one of the bonds is rotatable
    for idx, angle in enumerate(predictions["angle_preds"]):
        bond_1 = mol_tmp.GetBondBetweenAtoms(int(predictions["angle_triplets"][idx][0]),
                                             int(predictions["angle_triplets"][idx][1]))
        bond_2 = mol_tmp.GetBondBetweenAtoms(int(predictions["angle_triplets"][idx][1]),
                                             int(predictions["angle_triplets"][idx][2]))
        if bond_1.GetIdx() in rotatable_bond_idxs or bond_2.GetIdx() in rotatable_bond_idxs:
            Chem.rdMolTransforms.SetAngleRad(mol_tmp.GetConformers()[0],
                                             int(predictions["angle_triplets"][idx][2]),
                                             int(predictions["angle_triplets"][idx][1]),
                                             int(predictions["angle_triplets"][idx][0]),
                                             angle.item())

    # Generate specified number of conformations
    num_generated = 0
    while num_generated < num_confs:
        attempt_idx = 0
        while attempt_idx < max_attempts:
            attempt_idx += 1

            # Set chiralities for chirality inversion atoms based on predicted probabilities
            new_chiralities = []
            for i in range(len(predictions["chirality_atom_idxs"])):
                chirality_prob = predictions["chiral_preds"][i]
                new_chirality = random.choices([-1, 1], weights=[1.0 - chirality_prob, chirality_prob])[0]
                new_chiralities.append(new_chirality)
                current_chirality = compute_chirality(mol_tmp,
                                                      mol_tmp.GetConformers()[0],
                                                      int(predictions["chirality_atom_idxs"][i]))
                if int(current_chirality) != int(new_chirality):
                    flip_chirality(mol_tmp, 0, int(predictions["chirality_atom_idxs"][i]))

            # Set torsion angles for rotatable bonds without chirality inversion atoms via predicted von Mises mixtures
            for i, quartet in enumerate(predictions["torsion_no_chirality_quartets"]):
                set_torsion_angle_from_predictions(mol_tmp, 0, num_von_mises, loc_preds[i],
                                                   predictions["conc_preds"][i],
                                                   predictions["weight_preds"][i], quartet)

            # Set torsion angles for rotatable bonds with chirality inversion atoms via predicted von Mises mixtures
            for i, quartet in enumerate(predictions["torsion_chirality_quartets"]):
                if quartet[1] in predictions["chirality_atom_idxs"]:
                    chirality = new_chiralities[np.where(predictions["chirality_atom_idxs"] == quartet[1])[0][0]]
                else:
                    chirality = new_chiralities[np.where(predictions["chirality_atom_idxs"] == quartet[2])[0][0]]

                # Positive (R chirality) predictions if chirality is positive
                if chirality == 1:
                    set_torsion_angle_from_predictions(mol_tmp, 0, num_von_mises, chiral_loc_preds_pos[i],
                                                       predictions["chiral_conc_preds_pos"][i],
                                                       predictions["chiral_weight_preds_pos"][i], quartet)

                # Negative (S chirality) predictions if chirality is negative
                else:
                    set_torsion_angle_from_predictions(mol_tmp, 0, num_von_mises, chiral_loc_preds_neg[i],
                                                       predictions["chiral_conc_preds_neg"][i],
                                                       predictions["chiral_weight_preds_neg"][i], quartet)

            mol_tmp.GetConformers()[0].SetId(num_generated)

            # Optional filtering step
            filter_conf = False
            if filter_confs:
                filter_conf = filter_conformation(mol_tmp, num_generated)

            if not filter_conf:
                mol.AddConformer(mol_tmp.GetConformers()[0])
                break
        num_generated += 1

    return mol


class Predictor(object):
    """ Class to generate Von Mises predictions from mol object. """
    def __init__(self, model_path: str, model_config_path: str, use_cuda: bool = False):
        """
        :param model_path: Path to model parameters file.
        :param model_config_path: Path to model yaml config file.
        :param use_cuda: Whether to use CUDA.
        """
        super(Predictor).__init__()
        self.model_path = model_path
        self.model_config_path = model_config_path
        self.use_cuda = use_cuda
        self.experiment_config = yaml.load(open(self.model_config_path, 'r'), Loader=yaml.FullLoader)
        self.cuda = torch.cuda.is_available() and use_cuda
        if self.cuda:
            self.checkpoint_data = torch.load(self.model_path)
        else:
            self.checkpoint_data = torch.load(self.model_path, map_location=torch.device('cpu'))

        self.MAX_N = self.experiment_config['dataset_params']['graph_build_config'].get('max_nodes', 128)
        self.featurize_mol = FeaturizeMol(self.experiment_config["dataset_params"]["atom_feat_config"],
                                          self.experiment_config["dataset_params"]["bond_feat_config"],
                                          self.experiment_config["dataset_params"]["preprocess_config"])
        self.experiment_config['dataset_params']['graph_build_config']['include_target'] = False
        self.build_graph = BuildVonMisesGraph(**self.experiment_config["dataset_params"]["graph_build_config"])
        self.model = None

    # noinspection PyUnresolvedReferences
    def predict(self, mols: List[Union[rdchem.Mol, str]] = None, smiles_input: bool = False,
                generate_initial_geometry: bool = False, initial_geometry_seed: int = -1,
                initial_geometry_max_attempts: int = 0, use_etkdg_clean: bool = False,
                conform_to_existing_conf_idx: int = -1) -> \
            Tuple[List[Dict], List[Dict]]:
        """
        Generate predictions with von Mises model. If generate_initial_geometry is false, each input mol is required
        to have at least one embedded conformation. The first conformation is used as the starting point. VonMisesNet
        expects usage of explicit H atoms.

        :param mols: List of RDKit molecule objects or list of SMILES strings.
        :param smiles_input: Whether mols is a list of SMILES strings.
        :param generate_initial_geometry: Whether to generate a new initial geometry via ETKDG.
        :param initial_geometry_seed: Seed for generating initial conformation.
        :param initial_geometry_max_attempts: Number of times to re-try a failed ETKDG embedding before giving up.
        :param use_etkdg_clean: Whether or not to use ETKDG-Clean.
        :param conform_to_existing_conf_idx: For ETKDG-Clean; if >= 0, mirror chirality of conformation at this index.
        :return: List of prediction dictionaries, list of metadata.
        """
        predictions = []
        metadata = []

        for mol in mols:
            if smiles_input:
                assert type(mol) == str, f"SMILES input expected, got {mol}"
            else:
                assert type(mol) == rdchem.Mol, f"rdchem.Mol input expected, got {mol}"

        if smiles_input:
            print("Setting generate_initial_geometry to True for SMILES inputs")
            generate_initial_geometry = True
            print("Setting use_etkdg_clean to False for SMILES inputs")
            use_etkdg_clean = False

        for mol in tqdm(mols):
            start_time = time.time()
            if smiles_input:
                mol = Chem.AddHs(Chem.MolFromSmiles(mol))
            if generate_initial_geometry:
                if use_etkdg_clean:
                    mol = etkdg.generate_clean_etkdg_confs(mol, 1, seed=initial_geometry_seed,
                                                           max_embed_attempts=initial_geometry_max_attempts,
                                                           conform_to_existing_conf_idx=conform_to_existing_conf_idx,
                                                           exception_for_num_failure=False)
                else:
                    mol = etkdg.generate_etkdg_confs(mol, 1, seed=initial_geometry_seed,
                                                     max_embed_attempts=initial_geometry_max_attempts)

            if mol.GetNumConformers() == 0:
                predictions.append(None)
                end_time = time.time()
                meta = {"processing_time": end_time - start_time,
                        "processing_error": "Mol object had 0 conformations."}
                metadata.append(meta)
                continue

            torsion_targets = compute_torsions(mol, restrict_to_rotatable=True, restrict_to_first_conf=True)
            chirality_targets = compute_chirality_probability_targets(mol)
            angle_targets = compute_angles(mol)
            length_targets = compute_lengths(mol)

            # Featurize Molecule
            features = self.featurize_mol(mol, angle_targets['angle_triplets'])

            # Compatible now with rot-only
            graph = self.build_graph(mol, torsion_targets, angle_targets, length_targets, chirality_targets, features)

            num_vertex_features = graph['x'].shape[1]

            if self.model is None:
                self.model = VonMisesNet(self.MAX_N, num_vertex_features, **self.experiment_config['net_params'])

                if self.cuda:
                    self.model = self.model.cuda()

                self.model.load_state_dict(self.checkpoint_data['net_state_dict'])

            graph['x'] = torch.tensor(graph['x'])
            graph['edge_index'] = torch.tensor(graph['edge_index'])
            graph['torsion_mask'] = torch.tensor(graph['torsion_mask'])
            graph['chirality_torsion_mask'] = torch.tensor(graph['chirality_torsion_mask'])
            graph['angle_mask'] = torch.tensor(graph['angle_mask'])
            graph['len_mask'] = torch.tensor(graph['len_mask'])
            graph['chiral_mask'] = torch.tensor(graph['chiral_mask'])
            if self.cuda:
                graph['x'] = graph['x'].cuda()
                graph['edge_index'] = graph['edge_index'].cuda()
                graph['torsion_mask'] = graph['torsion_mask'].cuda()
                graph['chirality_torsion_mask'] = graph['chirality_torsion_mask'].cuda()
                graph['angle_mask'] = graph['angle_mask'].cuda()
                graph['len_mask'] = graph['len_mask'].cuda()
                graph['chiral_mask'] = graph['chiral_mask'].cuda()

            # Create batch dimension
            for k, v in graph.items():
                if isinstance(v, torch.Tensor):
                    graph[k] = v.unsqueeze(0)
                    
            # Generate predictions
            with torch.no_grad():
                self.model.eval()
                loc_preds, conc_preds, weight_preds, angle_preds, len_preds, chiral_preds, chiral_loc_preds_pos, \
                chiral_loc_preds_neg, chiral_conc_preds_pos, chiral_conc_preds_neg, chiral_weight_preds_pos, \
                chiral_weight_preds_neg = self.model(graph)

                loc_preds = loc_preds[0]
                conc_preds = conc_preds[0]
                weight_preds = weight_preds[0]
                angle_preds = angle_preds[0]
                len_preds = len_preds[0]
                chiral_preds = chiral_preds[0]
                chiral_loc_preds_pos = chiral_loc_preds_pos[0]
                chiral_loc_preds_neg = chiral_loc_preds_neg[0]
                chiral_conc_preds_pos = chiral_conc_preds_pos[0]
                chiral_conc_preds_neg = chiral_conc_preds_neg[0]
                chiral_weight_preds_pos = chiral_weight_preds_pos[0]
                chiral_weight_preds_neg = chiral_weight_preds_neg[0]

                angle_mask = graph['angle_mask'][0]
                torsion_mask = graph['torsion_mask'][0]
                length_mask = graph['len_mask'][0]
                chiral_mask = graph['chiral_mask'][0]
                chirality_torsion_mask = graph['chirality_torsion_mask'][0]

                loc_preds = loc_preds[torsion_mask > 0]
                conc_preds = conc_preds[torsion_mask > 0]
                weight_preds = weight_preds[torsion_mask > 0]

                angle_preds = angle_preds[angle_mask > 0]
                len_preds = len_preds[length_mask > 0]

                chiral_preds = chiral_preds[chiral_mask > 0]
                chiral_loc_preds_pos = chiral_loc_preds_pos[chirality_torsion_mask > 0]
                chiral_loc_preds_neg = chiral_loc_preds_neg[chirality_torsion_mask > 0]
                chiral_conc_preds_pos = chiral_conc_preds_pos[chirality_torsion_mask > 0]
                chiral_conc_preds_neg = chiral_conc_preds_neg[chirality_torsion_mask > 0]
                chiral_weight_preds_pos = chiral_weight_preds_pos[chirality_torsion_mask > 0]
                chiral_weight_preds_neg = chiral_weight_preds_neg[chirality_torsion_mask > 0]

            torsion_no_chirality_quartets = torsion_targets['atom_index_tuples'][
                np.where(torsion_targets['bond_indices_with_chirality'] == 0)[0]].astype(np.int32)
            torsion_chirality_quartets = torsion_targets['atom_index_tuples'][
                np.where(torsion_targets['bond_indices_with_chirality'] == 1)[0]].astype(np.int32)

            # noinspection PyUnresolvedReferences
            prediction = {"mol": mol.ToBinary(),
                          "bond_preds": len_preds.cpu(),
                          "angle_preds": angle_preds.cpu(),
                          "angle_triplets": angle_targets['angle_triplets'],
                          "loc_preds": loc_preds.cpu(),
                          "conc_preds": conc_preds.cpu(),
                          "weight_preds": weight_preds.cpu(),
                          "torsion_no_chirality_quartets": torsion_no_chirality_quartets,
                          "chiral_preds": chiral_preds.cpu(),
                          "chirality_atom_idxs": chirality_targets["atom_indices"],
                          "chiral_loc_preds_pos": chiral_loc_preds_pos.cpu(),
                          "chiral_loc_preds_neg": chiral_loc_preds_neg.cpu(),
                          "chiral_conc_preds_pos": chiral_conc_preds_pos.cpu(),
                          "chiral_conc_preds_neg": chiral_conc_preds_neg.cpu(),
                          "chiral_weight_preds_pos": chiral_weight_preds_pos.cpu(),
                          "chiral_weight_preds_neg": chiral_weight_preds_neg.cpu(),
                          "torsion_chirality_quartets": torsion_chirality_quartets}

            predictions.append(prediction)

            end_time = time.time()

            meta = {"processing_time": end_time - start_time,
                    "processing_error": None}
            metadata.append(meta)

        return predictions, metadata
