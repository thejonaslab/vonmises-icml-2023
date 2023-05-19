""" Classes for building molecular graph objects, masks, and adjacency matrices. """
import math
from typing import Dict, List

import numpy as np
from rdkit.Chem import rdchem

from vonmises.mol_utils import compute_bond_idxs


class BuildVonMisesGraph:
    """
    Class for building a graph object to pass into prediction models.
    """

    def __init__(self, max_nodes: int = 512,
                 max_torsions: int = 8192,
                 include_target: bool = True):
        """
        :param max_nodes: Vertex padding parameter.
        :param max_torsions: Torsion padding parameter.
        :param include_target: Whether to include the targets in the graph.
        """
        self.MAX_nodes = max_nodes
        self.MAX_torsions = max_torsions
        self.include_target = include_target

        self.calculate_mask = CalculateMask()
        self.calculate_adjacency = CalculateAdjacencyMatrix(max_nodes)

    def __call__(self, mol: rdchem.Mol, torsion_targets: Dict, angle_targets: Dict, length_targets: Dict,
                 chirality_targets: Dict, features: Dict) -> Dict:
        """
        Build graph as a dictionary from molecule and features.

        :param mol: RDKit mol object.
        :param torsion_targets: Dictionary of torsion targets, where angles have shape [num_torsion_bonds x num_confs].
        :param angle_targets: Dictionary of angles and triplets of atom indices defining these angles.
        :param length_targets: Dictionary of bond length targets.
        :param features: Dictionary of features.
        :param chirality_targets: Dictionary of chirality inversion atom indices and chiralities per conformation.
        :return: Molecular graph.
        """
        graph = {}

        torsion_angles = torsion_targets['angles']
        torsion_bond_indices_with_chirality = torsion_targets['bond_indices_with_chirality']
        torsion_atom_index_tuples = torsion_targets['atom_index_tuples']
        torsion_target_sorted_bond_idxs = torsion_targets['bond_indices']
        bond_angles, angle_triplets = angle_targets['angles'], angle_targets['angle_triplets']
        bond_lengths = length_targets['lengths']
        num_angles = bond_angles.shape[0]
        num_bonds = bond_lengths.shape[0]
        atom_features = features['atom_features']
        bond_features = features['bond_features']
        angle_features = features['angle_features']
        assert num_bonds == features['num_bonds']

        if self.include_target:
            # Extract rotatable bond torsions that do not have chirality inversion endpoint atoms
            torsion_target_no_chirality = torsion_angles[torsion_bond_indices_with_chirality == 0, :].T
            torsions_no_chirality = np.ones(self.MAX_torsions, dtype=np.float32) * (-100)
            torsions_no_chirality[:torsion_target_no_chirality.size] = torsion_target_no_chirality.\
                astype(np.float32).flatten()
            graph['y'] = torsions_no_chirality + math.pi
            graph['y_shape'] = np.array([[torsion_target_no_chirality.shape[0], torsion_target_no_chirality.shape[1]]],
                                        dtype=np.int32)

            # Extract rotatable bond torsions that have chirality inversion endpoint atoms
            torsion_target_chirality = torsion_angles[torsion_bond_indices_with_chirality == 1, :].T
            torsions_chirality = np.ones(self.MAX_torsions, dtype=np.float32) * (-100)
            torsions_chirality[:torsion_target_chirality.size] = torsion_target_chirality.\
                astype(np.float32).flatten()
            graph['y_chirality_torsions'] = torsions_chirality + math.pi
            graph['y_chirality_torsions_shape'] = np.array([[torsion_target_chirality.shape[0],
                                                             torsion_target_chirality.shape[1]]], dtype=np.int32)

            # Extract the columns in torsion_target that correspond to rotatable bonds connected to chirality inversion
            # atoms
            graph['y_bond_indices_with_chirality'] = torsion_bond_indices_with_chirality
            graph['y_bond_indices_with_chirality'] = np.append(graph['y_bond_indices_with_chirality'], np.array(
                [0 for _ in range(self.MAX_nodes - graph['y_bond_indices_with_chirality'].shape[0])], dtype=np.float32))
            graph['y_bond_indices_with_chirality_shape'] = np.array([[torsion_bond_indices_with_chirality.shape[0]]])

            # Extract the chiralities per conformation for each chirality inversion atom
            chirality_result = []
            chirality_atom_index_tuples = torsion_atom_index_tuples[np.where(
                torsion_bond_indices_with_chirality == 1)[0]]
            for i in range(chirality_atom_index_tuples.shape[0]):
                begin_atom_idx = chirality_atom_index_tuples[i][1]
                end_atom_idx = chirality_atom_index_tuples[i][2]
                if begin_atom_idx in chirality_targets["atom_indices"]:
                    chirality_result_idx = np.where(chirality_targets["atom_indices"] == begin_atom_idx)
                    chirality_result.append(chirality_targets["chiralities"][chirality_result_idx])
                else:
                    chirality_result_idx = np.where(chirality_targets["atom_indices"] == end_atom_idx)
                    chirality_result.append(chirality_targets["chiralities"][chirality_result_idx])
            chirality_result = np.array(chirality_result)
            chirality_result_df = np.ones(self.MAX_torsions, dtype=np.float32) * (-100)
            chirality_result_df[:chirality_result.size] = np.array(chirality_result, dtype=np.float32).flatten()
            graph['y_chirality_result'] = chirality_result_df

            # Compute probability of R chirality for each chirality inversion atom
            graph['y_chirality_prob'] = np.array(
                [len([x for x in chirality_targets["chiralities"][i] if x > 0]) /
                 chirality_targets["chiralities"].shape[1] for i in range(chirality_targets["chiralities"].shape[0])],
                dtype=np.float32)
            graph['y_chirality_prob'] = np.append(graph['y_chirality_prob'],
                                                  np.array([0 for _ in range(self.MAX_nodes -
                                                                             graph['y_chirality_prob'].shape[0])],
                                                           dtype=np.float32))
            graph['y_chirality_prob_shape'] = np.array([[chirality_targets["atom_indices"].shape[0]]])

        # Extract bond angle targets
        if self.include_target:
            graph['y_angles'] = np.mean(bond_angles.astype(np.float32), axis=1)
            graph['y_angles'] = np.append(graph['y_angles'],
                                          np.array([0 for _ in range(self.MAX_nodes -
                                                                     graph['y_angles'].shape[0])],
                                                   dtype=np.float32))
            graph['y_angles_shape'] = np.array([[graph['y_angles'].shape[0]]])
        else:
            graph['y_angles_shape'] = np.array([[num_angles]])

        # Extract bond length targets
        if self.include_target:
            graph['y_lens'] = np.mean(bond_lengths.astype(np.float32), axis=1)
            graph['y_lens'] = np.append(graph['y_lens'],
                                        np.array([0 for _ in range(self.MAX_nodes - graph['y_lens'].shape[0])],
                                                 dtype=np.float32))
            graph['y_lens_shape'] = np.array([[graph['y_lens'].shape[0]]])
        else:
            graph['y_lens_shape'] = np.array([[num_bonds]])

        graph['edge_index'] = self.calculate_adjacency(features['bond_idx'], features['num_atoms'],
                                                       features['num_bonds'], num_angles, angle_triplets)

        # Incorporate the vertex (bond, atom, angle) features
        num_bond_features = len(bond_features[0])
        num_atom_features = len(atom_features[0])
        num_angle_features = len(angle_features[0])

        bond_features = [bond_feat + [-10] * (num_atom_features + num_angle_features) for
                         bond_feat in bond_features]
        atom_features = [[-10] * num_bond_features + atom_feat + [-10] * num_angle_features
                         for atom_feat in atom_features]

        angle_feats = [[-10] * (num_bond_features + num_atom_features) + angle_feat for
                       angle_feat in angle_features]

        vertex_features_nopad = np.array(bond_features + atom_features + angle_feats, dtype=np.float32)

        num_vertices = vertex_features_nopad.shape[0]

        mask_nopad = self.calculate_mask(mol, num_vertices, features['rotatable_bonds'], False,
                                         torsion_bond_indices_with_chirality, torsion_target_sorted_bond_idxs)
        mask_nopad_chirality = self.calculate_mask(mol, num_vertices, features['rotatable_bonds'], True,
                                                   torsion_bond_indices_with_chirality, torsion_target_sorted_bond_idxs)

        vertex_features = np.zeros((self.MAX_nodes, num_atom_features + num_bond_features + num_angle_features),
                                   dtype=np.float32)
        vertex_features[:num_vertices, :] = vertex_features_nopad
        graph['x'] = vertex_features

        # Mask for rotatable bond nodes without chirality inversion endpoint atoms
        mask = np.zeros(self.MAX_nodes)
        mask[:num_vertices] = mask_nopad
        graph['torsion_mask'] = np.array(mask, dtype=np.float32)

        # Mask for rotatable bond nodes with chirality inversion endpoint atoms
        mask = np.zeros(self.MAX_nodes)
        mask[:num_vertices] = mask_nopad_chirality
        graph['chirality_torsion_mask'] = np.array(mask, dtype=np.float32)

        # Mask for bond angle nodes
        mask_angles = [0] * self.MAX_nodes
        mask_angles[num_vertices - num_angles:num_vertices] = [1] * num_angles
        graph['angle_mask'] = np.array(mask_angles, dtype=np.float32)

        # Mask for bonds
        mask_len = [0] * self.MAX_nodes
        mask_len[:num_bonds] = [1] * num_bonds
        graph['len_mask'] = np.array(mask_len, dtype=np.float32)

        # Mask for chirality inversion atom nodes
        mask_chiral = [0] * self.MAX_nodes
        mask_chiral[num_bonds:num_vertices - num_angles] = \
            [1 if x in chirality_targets["atom_indices"] else 0 for x in range(mol.GetNumAtoms())]
        graph['chiral_mask'] = np.array(mask_chiral, dtype=np.float32)

        return graph


class CalculateMask:
    """
    Generates the mask for a molecule.
    """

    def __init__(self):
        pass

    # noinspection PyUnresolvedReferences
    def __call__(self, mol: rdchem.Mol, num_vertices: int, rotatable_bonds: List, chirality: bool,
                 bond_indices_with_chirality: np.ndarray,
                 torsion_target_sorted_bond_idxs) -> List:
        """
        Compute rotatable bond torsion masks.

        :param mol: RDKit mol object.
        :param num_vertices: Number of vertices in the graph (not necessarily number of atoms).
        :param rotatable_bonds: List of rotatable bonds.
        :param chirality: Whether or not to create a mask for chirality torsions.
        :param bond_indices_with_chirality: Bonds with chirality.
        :param torsion_target_sorted_bond_idxs: The original bond indices from RDKit.
        :return: Mask.
        """
        mask = [0] * num_vertices
        bond_idxs = compute_bond_idxs(mol, rotatable_bonds)
        for i in bond_idxs:
            index_in_sorted_bond_idxs = np.where(torsion_target_sorted_bond_idxs == i)[0][0]
            if chirality:
                if bond_indices_with_chirality[index_in_sorted_bond_idxs] == 1:
                    mask[i] = 1
            else:
                if bond_indices_with_chirality[index_in_sorted_bond_idxs] == 0:
                    mask[i] = 1
        return mask


class CalculateAdjacencyMatrix:
    """
    A class to calculate the adjacency matrix of a graph.
    """

    def __init__(self, MAX_nodes: int):
        """
        :param MAX_nodes: Parameter for padding.
        """
        self.MAX_nodes = MAX_nodes

    def __call__(self, bonds, num_atoms: int, num_bonds: int, num_angles: int, angle_triplets: List) -> np.ndarray:
        """
        Find the adjacency matrix given the graph.

        :param bonds: Iterable of rdkit Bond objects to connect nodes.
        :param num_atoms: Number of atoms in the graph.
        :param num_bonds: Number of bonds in the graph.
        :param num_angles: Number of angle nodes in the graph.
        :param angle_triplets: angle_triplet info from targets.
        """
        # Compute edge connectivity in COO format corresponding to a complete graph on
        # num_bonds + num_atoms + num_angles nodes
        edge_index_nopad = np.zeros([num_bonds + num_atoms + num_angles, num_bonds + num_atoms + num_angles],
                                    dtype=np.float32)
        for bond in bonds:
            bond_index = bond.GetIdx()
            begin_atom_index = bond.GetBeginAtom().GetIdx()
            end_atom_index = bond.GetEndAtom().GetIdx()
            edge_index_nopad[bond_index, num_bonds + begin_atom_index] = 1
            edge_index_nopad[num_bonds + begin_atom_index, bond_index] = 1
            edge_index_nopad[num_bonds + end_atom_index, bond_index] = 1
            edge_index_nopad[bond_index, num_bonds + end_atom_index] = 1

        for i, (a, b, c) in enumerate(angle_triplets):
            edge_index_nopad[num_bonds + a, num_bonds + num_atoms + i] = 1
            edge_index_nopad[num_bonds + c, num_bonds + num_atoms + i] = 1
            edge_index_nopad[num_bonds + num_atoms + i, num_bonds + a] = 1
            edge_index_nopad[num_bonds + num_atoms + i, num_bonds + c] = 1

        edge_index = np.zeros((self.MAX_nodes, self.MAX_nodes), dtype=np.float32)
        edge_index[:edge_index_nopad.shape[0], :edge_index_nopad.shape[1]] = edge_index_nopad
        return edge_index
