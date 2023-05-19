""" Neural network util functions. """
import numpy as np
from typing import Optional

import torch
import torch.nn as nn


def param_count(model: nn.Module) -> int:
    """
    Determine number of trainable parameters.

    :param model: An nn.Module.
    :return: The number of trainable parameters.
    """
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def _eval_poly(y, coef):
    """
    Modified Bessel function helper. Borrowed from PyTorch source code.

    :param y: Input.
    :param coef: Coefficient.
    :return: Bessel function helper output.
    """
    coef = list(coef)
    result = coef.pop()
    while coef:
        result = coef.pop() + y * result
    return result


_I0_COEF_SMALL = [1.0, 3.5156229, 3.0899424, 1.2067492, 0.2659732, 0.360768e-1, 0.45813e-2]
_I0_COEF_LARGE = [0.39894228, 0.1328592e-1, 0.225319e-2, -0.157565e-2, 0.916281e-2,
                  -0.2057706e-1, 0.2635537e-1, -0.1647633e-1, 0.392377e-2]
_I1_COEF_SMALL = [0.5, 0.87890594, 0.51498869, 0.15084934, 0.2658733e-1, 0.301532e-2, 0.32411e-3]
_I1_COEF_LARGE = [0.39894228, -0.3988024e-1, -0.362018e-2, 0.163801e-2, -0.1031555e-1,
                  0.2282967e-1, -0.2895312e-1, 0.1787654e-1, -0.420059e-2]

_COEF_SMALL = [_I0_COEF_SMALL, _I1_COEF_SMALL]
_COEF_LARGE = [_I0_COEF_LARGE, _I1_COEF_LARGE]


def _log_modified_bessel_fn(x, order=0) -> torch.Tensor:
    """
    Compute ``log(I_order(x))`` for ``x > 0``, where `order` is either 0 or 1. Borrowed from PyTorch source code.

    :param x: Input.
    :param order: Order of Bessel function.
    :return: Value of log modified Bessel function.
    """
    assert order == 0 or order == 1

    # compute small solution
    y = (x / 3.75)
    y = y * y
    small = _eval_poly(y, _COEF_SMALL[order])
    if order == 1:
        small = x.abs() * small
    small = small.log()

    # compute large solution
    y = 3.75 / x
    large = x - 0.5 * x.log() + _eval_poly(y, _COEF_LARGE[order]).log()

    result = torch.where(x < 3.75, small, large)
    return result


def von_mises_logprob_nogeo(loc: torch.Tensor, concentration: torch.Tensor,
                            log_weights: torch.Tensor,
                            data: torch.Tensor) -> torch.Tensor:
    """
    Compute the von Mises data likelihood without using pytorch-scatter.

    :param loc: von Mises loc parameters [num_torsions x num_von_mises].
    :param concentration: von Mises concentration parameters [num_torsions x num_von_mises].
    :param log_weights: log of von Mises weights [num_torsions x num_von_mises].
    :param data: Data to evaluate log probability [num_confs x num_bonds].
    :return: Likelihood of bond in each data point [num_confs x num_bonds].
    """
    num_von_mises = loc.shape[1]
    num_confs = data.shape[0]
    num_torsions = loc.shape[0]

    assert num_torsions == data.shape[1]

    exp_term = torch.cos(data.reshape(num_confs, 1, num_torsions).repeat(1, num_von_mises, 1) -
                         torch.transpose(loc, 0, 1).repeat(num_confs, 1, 1))
    exp_term = exp_term * torch.transpose(concentration, 0, 1).repeat(num_confs, 1, 1)
    log_prob = exp_term

    log_prob = log_prob - np.log(2 * np.pi)

    normalizer = torch.transpose(_log_modified_bessel_fn(concentration), 0, 1).repeat(num_confs, 1, 1)

    log_prob = log_prob - normalizer
    log_prob = log_prob + torch.transpose(log_weights, 0, 1).repeat(num_confs, 1, 1)

    log_prob = torch.logsumexp(log_prob, 1)

    return log_prob


def von_mises_loss_stable(loc: torch.Tensor, concentration: torch.Tensor,
                          log_weights: torch.Tensor,
                          data: torch.Tensor) -> torch.Tensor:
    """
    Compute the von Mises mixture loss on data.

    :param loc: von Mises loc parameters [num_torsions x num_von_mises].
    :param concentration: von Mises concentration parameters [num_torsions x num_von_mises].
    :param log_weights: log of von Mises weights [num_torsions x num_von_mises].
    :param data: Data to evaluate log probability [num_confs x num_bonds].
    :return: Negative log likelihood of data.
    """
    prob = von_mises_logprob_nogeo(loc, concentration, log_weights, data)
    avg_log_prob = prob.mean()  # Average loss per bond

    return -1.0 * avg_log_prob


def dense_loss(batch, targets, loc_preds, conc_preds, weight_preds) -> Optional[torch.Tensor]:
    """
    Dense von Mises loss.

    :param batch: Dense graph batch.
    :param targets: Targets.
    :param loc_preds: von Mises loc predictions.
    :param conc_preds: von Mises concentration predictions.
    :param weight_preds: von Mises weight predictions.
    :return: Loss.
    """
    loss = None
    for i in range(targets.shape[0]):
        mask = torch.where(batch['torsion_mask'][i] == 1)
        C, N = batch['y_shape'][i][0]
        tgt = torch.reshape(targets[i][:C * N], (C, N))

        if tgt.shape[1] != 0:
            if loss is None:
                loss = von_mises_loss_stable(loc_preds[i][mask], conc_preds[i][mask], weight_preds[i][mask], tgt)
            else:
                loss += von_mises_loss_stable(loc_preds[i][mask], conc_preds[i][mask], weight_preds[i][mask], tgt)
    return loss


def dense_loss_chirality(batch, targets, loc_preds_pos, loc_preds_neg, conc_preds_pos, conc_preds_neg,
                         weight_preds_pos, weight_preds_neg) -> Optional[torch.Tensor]:
    """
    Dense von Mises loss for rotatable bonds connected to a chirality inversion atom.

    :param batch: Dense graph batch.
    :param targets: Targets.
    :param loc_preds_pos: von Mises loc predictions, positive chirality.
    :param loc_preds_neg: von Mises loc predictions, negative chirality.
    :param conc_preds_pos: von Mises concentration predictions, positive chirality.
    :param conc_preds_neg: von Mises concentration predictions, negative chirality.
    :param weight_preds_pos: von Mises weight predictions, positive chirality.
    :param weight_preds_neg: von Mises weight predictions, negative chirality.
    :return: Loss.
    """
    loss = None
    for i in range(targets.shape[0]):
        mask = torch.where(batch['chirality_torsion_mask'][i] == 1)[0]
        C, N = batch['y_chirality_torsions_shape'][i][0]
        tgt = torch.reshape(targets[i][:C * N], (C, N))

        chirality_results = torch.reshape(batch["y_chirality_result"][i][:C * N], (N, C))

        if tgt.shape[1] != 0:
            tgt_pos = tgt[torch.where(chirality_results[0, :] > 0)]
            tgt_neg = tgt[torch.where(chirality_results[0, :] < 0)]

            if tgt_pos.shape[0] != 0:
                if loss is None:
                    loss = von_mises_loss_stable(loc_preds_pos[i][mask], conc_preds_pos[i][mask],
                                                 weight_preds_pos[i][mask], tgt_pos)
                else:
                    loss += von_mises_loss_stable(loc_preds_pos[i][mask], conc_preds_pos[i][mask],
                                                  weight_preds_pos[i][mask], tgt_pos)
            if tgt_neg.shape[0] != 0:
                if loss is None:
                    loss = von_mises_loss_stable(loc_preds_neg[i][mask], conc_preds_neg[i][mask],
                                                 weight_preds_neg[i][mask], tgt_neg)
                else:
                    loss += von_mises_loss_stable(loc_preds_neg[i][mask], conc_preds_neg[i][mask],
                                                  weight_preds_neg[i][mask], tgt_neg)

    return loss
