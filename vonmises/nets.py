""" VonMisesNet definition. """
from typing import Tuple

import torch
import torch.nn.functional as F

NORMALIZATIONS = {
    "layer": torch.nn.LayerNorm,
    "batch": torch.nn.BatchNorm1d,
    "instance": torch.nn.InstanceNorm1d
}


def get_normalization_module(norm_type: str, hidden_size: int, num_vertices: int = None):
    """
    Get normalization module.

    :param norm_type: Normalization type.
    :param hidden_size: Hidden size.
    :param num_vertices: Number of vertices.
    :return: Normalization module.
    """
    norm = NORMALIZATIONS.get(norm_type, None)
    if norm is not None:
        if norm_type == "batch":
            norm = norm(num_vertices)
        else:
            norm = norm(hidden_size)
    return norm


class VonMisesNet(torch.nn.Module):
    """
     VonMisesNet definition.
     """

    def __init__(self, num_vertices, num_vertex_features, hidden_size: int = 256, num_layers: int = 32,
                 final_linear_size: int = 1024, final_output_size: int = 1, reduce: str = 'mean', min_conc: float = 1.0,
                 max_conc: float = 20.0, init_norm: str = "", linear_first: bool = True, extra_norm: str = "",
                 extra_layers: int = 0, end_norm: str = "", conc_norm: str = ""):
        """
        :param num_vertices: Number of vertices (padded) per graph in the expected inputs.
        :param num_vertex_features: The number of features per node (vertex) in the expected inputs.
        :param hidden_size: Size of hidden layers.
        :param num_layers: Number of times to repeat the message passing steps.
        :param final_linear_size: Size of final linear layers.
        :param final_output_size: Output size (for each of the five outputs).
        :param reduce: How to combine messages at each vertex during message passing.
        :param min_conc: Minimum allowed concentration per von Mises distribution.
        :param max_conc: Maximum allowed concentration per von Mises distribution.
        :param init_norm: What type of normalization method to use before message passing, if any.
        :param linear_first: Whether to have linear layers before (True) or after (False) each message passing step.
        :param extra_norm: What type of normalization method to use between each extra linear layer, if any.
        :param extra_layers: How many extra linear layers to run after each message passing step.
        :param end_norm: What type of normalization method to use after all message passing steps, if any.
        :param conc_norm: What type of normalization method to use on the concentration predictions, if any.
        """
        super(VonMisesNet, self).__init__()
        self.reduce = reduce
        self.min_conc = min_conc
        self.max_conc = max_conc
        self.num_vertices = num_vertices
        self.num_vertex_features = num_vertex_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.final_linear_size = final_linear_size
        self.final_output_size = final_output_size
        self.vertex_featurize = torch.nn.Linear(self.num_vertex_features, self.hidden_size)
        self.stack = GCNStack(self.hidden_size, linear_first, self.num_layers,
                              num_vertices=self.num_vertices, extra_norm=extra_norm, extra_layers=extra_layers,
                              end_norm=end_norm, reduce=self.reduce)

        self.init_norm = get_normalization_module(init_norm, self.hidden_size, num_vertices=self.num_vertices)
        self.conc_norm = get_normalization_module(conc_norm, self.hidden_size, num_vertices=self.num_vertices)
        self.conc_chiral_norm_pos = get_normalization_module(conc_norm, self.hidden_size,
                                                             num_vertices=self.num_vertices)
        self.conc_chiral_norm_neg = get_normalization_module(conc_norm, self.hidden_size,
                                                             num_vertices=self.num_vertices)

        self.loc_layer = OutputLayer(self.hidden_size, self.final_linear_size, self.final_output_size)
        self.conc_layer = OutputLayer(self.hidden_size, self.final_linear_size, self.final_output_size, self.conc_norm)
        self.weight_layer = OutputLayer(self.hidden_size, self.final_linear_size, self.final_output_size)
        self.angle_layer = OutputLayer(self.hidden_size, self.final_linear_size, 1)
        self.len_layer = OutputLayer(self.hidden_size, self.final_linear_size, 1)
        self.chiral_layer = OutputLayer(self.hidden_size, self.final_linear_size, 1)
        self.loc_chiral_layer_pos = OutputLayer(self.hidden_size, self.final_linear_size, self.final_output_size)
        self.conc_chiral_layer_pos = OutputLayer(self.hidden_size, self.final_linear_size, self.final_output_size,
                                                 self.conc_chiral_norm_pos)
        self.weight_chiral_layer_pos = OutputLayer(self.hidden_size, self.final_linear_size, self.final_output_size)
        self.loc_chiral_layer_neg = OutputLayer(self.hidden_size, self.final_linear_size, self.final_output_size)
        self.conc_chiral_layer_neg = OutputLayer(self.hidden_size, self.final_linear_size, self.final_output_size,
                                                 self.conc_chiral_norm_neg)
        self.weight_chiral_layer_neg = OutputLayer(self.hidden_size, self.final_linear_size, self.final_output_size)
        self.weight_softmax = torch.nn.LogSoftmax(dim=2)

    def forward(self, batch) -> Tuple:
        """
        Forward pass.

        :param batch: Data batch.
        :return: Predictions.
        """
        v_i_in = self.vertex_featurize(batch['x'])
        G = batch['edge_index']

        if self.init_norm is not None:
            v_i_in = self.init_norm(v_i_in)

        v_i_in = self.stack(v_i_in, G=G)

        angle_preds = self.angle_layer(v_i_in)
        angle_preds = angle_preds * angle_preds

        len_preds = self.len_layer(v_i_in)
        len_preds = len_preds * len_preds

        chiral_preds = self.chiral_layer(v_i_in)
        chiral_preds = chiral_preds * chiral_preds

        loc_preds = self.loc_layer(v_i_in)

        conc_preds = self.conc_layer(v_i_in)
        conc_preds = self.max_conc * torch.sigmoid(conc_preds) + self.min_conc

        weight_preds = self.weight_layer(v_i_in)
        weight_preds = self.weight_softmax(weight_preds)

        chiral_loc_preds_pos = self.loc_chiral_layer_pos(v_i_in)

        chiral_conc_preds_pos = self.conc_chiral_layer_pos(v_i_in)
        chiral_conc_preds_pos = self.max_conc * torch.sigmoid(chiral_conc_preds_pos) + self.min_conc

        chiral_weight_preds_pos = self.weight_chiral_layer_pos(v_i_in)
        chiral_weight_preds_pos = self.weight_softmax(chiral_weight_preds_pos)

        chiral_loc_preds_neg = self.loc_chiral_layer_neg(v_i_in)

        chiral_conc_preds_neg = self.conc_chiral_layer_neg(v_i_in)
        chiral_conc_preds_neg = self.max_conc * torch.sigmoid(chiral_conc_preds_neg) + self.min_conc

        chiral_weight_preds_neg = self.weight_chiral_layer_neg(v_i_in)
        chiral_weight_preds_neg = self.weight_softmax(chiral_weight_preds_neg)

        return loc_preds, conc_preds, weight_preds, angle_preds, len_preds, chiral_preds, chiral_loc_preds_pos, \
               chiral_loc_preds_neg, chiral_conc_preds_pos, chiral_conc_preds_neg, chiral_weight_preds_pos, \
               chiral_weight_preds_neg


class GCNStack(torch.nn.Module):
    """
    A stack of GCN layers with options for how to loop over each layer.
    """

    def __init__(self,
                 hidden_size: int,
                 order: bool,
                 num_layers: int,
                 reduce: str = 'mean',
                 num_vertices: int = None,
                 extra_norm: str = "",
                 extra_layers: int = 0,
                 end_norm: str = "",
                 use_residual: bool = True):
        """
        :param hidden_size: Feature size of expected input tensors.
        :param num_vertices: Number of vertices per graph of expected inputs.
        :param extra_norm: Normalizations before each extra layer (if non-empty, length should equal extra_layers).
        :param extra_layers: Number of extra linear layers to add.
        :param end_norm: Normalization to apply after all extra layers.
        :param order: Whether to do the linear layer before the message passing (True) or after (False).
        :param reduce: How to merge messages passed to each vertex (see reduce in pytorch_scatter).
        :param use_residual: Whether to add the residual at each layer.
        """
        super(GCNStack, self).__init__()
        self.hidden_size = hidden_size
        self.num_vertices = num_vertices
        self.num_layers = num_layers
        self.use_residual = use_residual
        self.reduce = reduce
        self.layers = torch.nn.ModuleList([GCNLayer(self.hidden_size, reduce=self.reduce,
                                                    num_vertices=self.num_vertices, extra_norm=extra_norm,
                                                    extra_layers=extra_layers, end_norm=end_norm, order=order)
                                           for _ in range(self.num_layers)])

    def forward(self, v_i_in, G=None):
        """
        Forward pass.

        :param v_i_in: Features tensors for nodes in graph.
        :param G: Adjacency matrix of graph.
        """
        for l in self.layers:
            v_i_c = l(v_i_in, G=G)
            if self.use_residual:
                v_i_in = v_i_in + v_i_c
            else:
                v_i_in = v_i_c
        return v_i_in


class GCNLayer(torch.nn.Module):
    """
    A single GCN layer with message passing.
    """

    def __init__(self, hidden_size,
                 reduce: str = "mean",
                 num_vertices: int = None,
                 extra_norm: str = "",
                 extra_layers: int = 0,
                 end_norm: str = "",
                 order: bool = True):
        """
        :param hidden_size: Feature size of expected input tensors.
        :param num_vertices: Number of vertices per graph of expected inputs.
        :param extra_norm: Normalizations before each extra layer (if non-empty, length should equal extra_layers).
        :param extra_layers: Number of extra linear layers to add.
        :param end_norm: Normalization to apply after all extra layers.
        :param order: Whether to do the linear layer before the message passing (True) or after (False).
        """
        super(GCNLayer, self).__init__()
        self.hidden_size = hidden_size
        self.reduce = reduce
        self.num_vertices = num_vertices
        self.L_v = torch.nn.Linear(self.hidden_size, self.hidden_size)  # Linear layers for vertices
        self.N_extra = torch.nn.ModuleList([get_normalization_module(extra_norm, self.hidden_size,
                                                                     self.num_vertices) for _ in range(extra_layers)])
        self.L_extra = torch.nn.ModuleList([torch.nn.Linear(self.hidden_size, self.hidden_size)
                                            for _ in range(extra_layers)])
        self.N_end = get_normalization_module(end_norm, self.hidden_size, self.num_vertices)
        self.order = order

    def forward(self, v_i_in, G=None):
        """
        Forward pass.

        :param v_i_in: Features tensors for nodes in graph.
        :param G: Adjacency matrix of graph.
        """
        if self.order:
            v_i_prime = self.L_v(v_i_in)
            v_i_e = self.message_passing(v_i_prime, G=G)
        else:
            v_i_prime = self.message_passing(v_i_in, G=G)
            v_i_e = self.L_v(v_i_prime)
        for i, l in enumerate(self.L_extra):
            if not self.N_extra[i] is None:
                v_i_e = self.N_extra[i](v_i_e)
            v_i_e = F.relu(l(v_i_e))

        v_i_e = F.relu(v_i_e)            
        if self.N_end is not None:
            v_i_e = self.N_end(v_i_e)
        return v_i_e

    def message_passing(self, v_i_in, G=None):
        """
        Completes message passing.

        :param v_i_in: Features tensors for nodes in graph.
        :param G: Adjacency matrix of graph.
        """
        if not self.reduce == "nodes_only":
            n = torch.sum(G, 2)
            e = torch.eye(self.num_vertices)
            if G.is_cuda:
                n = n.cuda()
                e = e.cuda()
            G = G + torch.einsum('ij,jk->ijk', [n, e])
        out = torch.einsum('ijk,ikl->ijl', [G, v_i_in])
        if self.reduce == 'mean':
            diag = torch.diagonal(G, dim1=-2, dim2=-1)
            add_ones = torch.zeros_like(diag)
            add_ones[diag == 0] = 1
            diag = diag + add_ones  # Avoid divide by zeros
            out = out / diag.unsqueeze(2).repeat(1, 1, self.hidden_size)
        return out


class OutputLayer(torch.nn.Module):
    """
    A straightforward MLP for creating prediction outputs.
    """

    def __init__(self, hidden_size: int,
                 final_linear_size: int,
                 final_output_size: int,
                 end_norm=None):
        """
        :param hidden_size: Feature size of expected input tensors.
        :param final_linear_size: Output size of final linear layer.
        :param final_output_size: Output size of MLP.
        :param end_norm: Normalization function to apply to final output, if any.
        """
        super(OutputLayer, self).__init__()
        self.hidden_size = hidden_size
        self.final_linear_size = final_linear_size
        self.final_output_size = final_output_size
        self.end_norm = end_norm
        self.init_layer_norm = torch.nn.LayerNorm(self.hidden_size)
        self.final_linear_layer = torch.nn.Linear(self.hidden_size, self.final_linear_size)
        self.layer_norm = torch.nn.LayerNorm(self.final_linear_size)
        self.output_layer = torch.nn.Linear(self.final_linear_size, self.final_output_size)

    def forward(self, v_i_in: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        :param v_i_in: input feature tensor.
        :return: Predictions.
        """
        preds = self.init_layer_norm(v_i_in)
        preds = self.final_linear_layer(preds)
        preds = F.relu(preds)
        preds = self.output_layer(preds)
        if self.end_norm is not None:
            preds = self.end_norm(preds)
        return preds
