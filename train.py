""" Train VonMisesNet. """
import os
import random

random.seed(0)
import time
from typing import Dict, Tuple

import click
import numpy as np

np.random.seed(0)
from tensorboardX import SummaryWriter
import torch

torch.manual_seed(0)
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import yaml

from vonmises import dbdataset
from vonmises.nets import VonMisesNet
from vonmises.net_utils import dense_loss, dense_loss_chirality, param_count
from vonmises.sqlalchemy_model import *

CHECKPOINT_DIR = "checkpoints"
LATEST_CHECKPOINT_EVERY = 2
SAVE_CHECKPOINT_EVERY = 20
TB_LOG_DIR = "tblogs"


def save_checkpoint(epoch_i, model, optimizer, best_test_loss, checkpoint_filename):
    """
    Save model checkpoint.

    :param epoch_i: Epoch number.
    :param model: PyTorch neural network.
    :param optimizer: PyTorch optimizer.
    :param best_test_loss: Best test loss recorded so far.
    :param checkpoint_filename: Checkpoint filename.
    """
    print(f'Saving checkpoint for epoch {epoch_i:8d}')
    torch.save({'epoch_i': epoch_i,
                'net_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_test_loss': best_test_loss},
               checkpoint_filename)


def train_model(experiment_config: Dict, experiment_output_prefix: str, load_from_checkpoint: bool = False,
                detect_anomaly: bool = False):
    """
    Create the model using the parameters in the experiment_config dictionary and run training.

    :param experiment_config: Dictionary containing experiment parameters.
    :param experiment_output_prefix: Prefix for saving metadata.
    :param load_from_checkpoint: Whether or not to load model from a checkpoint.
    :param detect_anomaly: Whether or not to use torch.autograd.set_detect_anomaly.
    """
    checkpoint_filename = os.path.join(CHECKPOINT_DIR, f"{experiment_output_prefix}.chk")
    best_model_filename = os.path.join(CHECKPOINT_DIR, f"{experiment_output_prefix}.best.chk")

    batch_size = experiment_config['batch_size']
    MAX_N = experiment_config['dataset_params']['graph_build_config'].get('max_nodes', 128)

    max_atoms = experiment_config['max_atoms']
    max_num_mols = experiment_config.get('max_num_mols', -1)
    smallest_element_subset = experiment_config['smallest_element_subset']

    # Train-test split using hashes of morgan fingerprints, selecting for molecules with desired # atoms and types
    print(f'Computing train/test split...')
    dataset_config = experiment_config['dataset']
    train_data_ids = subset_query(dataset_config['mol_db'], max_atoms,
                                  smallest_element_subset, "train",
                                  max_num_mols)

    test_data_ids = subset_query(dataset_config['mol_db'], max_atoms,
                                 smallest_element_subset, "test",
                                 max_num_mols)

    print(f'Loading targets from db...')
    train_data = dbdataset.create_dataset(dataset_config['mol_db'],
                                          dataset_config['target_file'],
                                          train_data_ids,
                                          **experiment_config['dataset_params'])

    test_data = dbdataset.create_dataset(dataset_config['mol_db'],
                                         dataset_config['target_file'],
                                         test_data_ids,
                                         **experiment_config['dataset_params'])

    num_vertex_features = train_data[0]['x'].shape[1]

    train_data_length, test_data_length = len(train_data), len(test_data)
    print(f'Train size = {train_data_length:,} | Test size = {test_data_length:,}')

    persistent_workers = experiment_config['persistent_workers']
    num_workers = experiment_config['num_workers']

    train_sampler = dbdataset.SubsetSampler(experiment_config.get('train_epoch_size', len(train_data)),
                                            len(train_data),
                                            shuffle=experiment_config['shuffle'],
                                            )

    test_sampler = dbdataset.SubsetSampler(experiment_config.get('test_epoch_size', len(test_data)),
                                           len(test_data)
                                           )

    train_loader = DataLoader(train_data, batch_size, sampler=train_sampler,
                              persistent_workers=persistent_workers, num_workers=num_workers,
                              drop_last=True)
    test_loader = DataLoader(test_data, batch_size, sampler=test_sampler,
                             persistent_workers=persistent_workers, num_workers=num_workers,
                             drop_last=True)

    model = VonMisesNet(MAX_N, num_vertex_features, **experiment_config['net_params'])
    loss_fn = dense_loss

    tb_writer = SummaryWriter(os.path.join(TB_LOG_DIR, experiment_output_prefix))

    cuda = torch.cuda.is_available()

    print('Number of parameters = {:,}'.format(param_count(model)))

    if cuda:
        model = model.cuda()

    optimizer = Adam(model.parameters(), **experiment_config['opt_params'])

    start_epoch = 0
    best_test_loss = float('inf')

    # Load from checkpoint if it exists
    if load_from_checkpoint:
        checkpoint_data = torch.load(checkpoint_filename)
        model.load_state_dict(checkpoint_data['net_state_dict'])
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        start_epoch = checkpoint_data['epoch_i'] + 1
        best_test_loss = checkpoint_data['best_test_loss']
        print("Starting at epoch", start_epoch)

    epoch_n = experiment_config['epoch_n']
    clip_grad = experiment_config['clip_grad']
    clip_cutoff = experiment_config['clip_cutoff']

    with torch.autograd.set_detect_anomaly(detect_anomaly):
        for epoch_i in trange(start_epoch, epoch_n):
            print('Starting epoch', epoch_i)
            train_loss, n_iter, net_time, t1, t2, torsion_loss, bond_loss, angle_loss, chiral_loss, \
            chiral_torsion_loss = run_epoch(model, train_loader, optimizer, False, cuda, loss_fn, clip_grad,
                                            clip_cutoff)
            print(f'Epoch {epoch_i}, train loss: {train_loss}')
            tb_writer.add_scalar(f"train/loss", train_loss, epoch_i)
            tb_writer.add_scalar(f"train/torsion_loss", torsion_loss, epoch_i)
            tb_writer.add_scalar(f"train/bond_loss", bond_loss, epoch_i)
            tb_writer.add_scalar(f"train/angle_loss", angle_loss, epoch_i)
            tb_writer.add_scalar(f"train/chiral_loss", chiral_loss, epoch_i)
            tb_writer.add_scalar(f"train/chiral_torsion_loss", chiral_torsion_loss, epoch_i)
            tb_writer.add_scalar("points_per_sec", n_iter / (t2 - t1), epoch_i)
            tb_writer.add_scalar("train efficiency", net_time / (t2 - t1), epoch_i)

            test_loss, _, _, _, _, torsion_loss, bond_loss, angle_loss, chiral_loss, chiral_torsion_loss = run_epoch(
                model, test_loader, optimizer, True, cuda, loss_fn, clip_grad, clip_cutoff)
            print(f'Epoch {epoch_i}, test loss: {test_loss}')
            tb_writer.add_scalar(f"test/loss", test_loss, epoch_i)
            tb_writer.add_scalar(f"test/torsion_loss", torsion_loss, epoch_i)
            tb_writer.add_scalar(f"test/bond_loss", bond_loss, epoch_i)
            tb_writer.add_scalar(f"test/angle_loss", angle_loss, epoch_i)
            tb_writer.add_scalar(f"test/chiral_loss", chiral_loss, epoch_i)
            tb_writer.add_scalar(f"test/chiral_torsion_loss", chiral_torsion_loss, epoch_i)

            if test_loss < best_test_loss:
                save_checkpoint(epoch_i, model, optimizer, None, best_model_filename)
                best_test_loss = test_loss

            if epoch_i % LATEST_CHECKPOINT_EVERY == 0:
                save_checkpoint(epoch_i, model, optimizer, best_test_loss, checkpoint_filename)

            if epoch_i % SAVE_CHECKPOINT_EVERY == 0:
                save_checkpoint(epoch_i, model, optimizer, best_test_loss,
                                os.path.join(CHECKPOINT_DIR, f"{experiment_output_prefix}.epoch_{epoch_i}.chk"))


def run_epoch(model: torch.nn.Module, data: DataLoader, optimizer: Adam, predict_only: bool, cuda: bool,
              loss_fn, clip_grad: bool, clip_cutoff: int) -> \
        Tuple[float, int, float, float, float, float, float, float, float, float]:
    """
    Train for one epoch.

    :param model: PyTorch neural network.
    :param data: PyTorch DataLoader.
    :param optimizer: Adam optimizer.
    :param predict_only: Whether to just predict (eval mode) or train.
    :param cuda: Whether or not cuda is available.
    :param loss_fn: loss function to use for torsion predictions.
    :param clip_grad: Whether or not to do gradient clipping.
    :param clip_cutoff: Threshold for gradient clipping.
    :return: Losses and metadata.
    """
    if predict_only:
        model.eval()
    else:
        model.train()

    loss_sum, n_iter, net_time, torsion_loss_sum, len_loss_sum, angle_loss_sum, chiral_loss_sum, \
    chiral_torsion_loss_sum = 0, 0, 0, 0, 0, 0, 0, 0
    t1 = time.time()
    for batch in tqdm(data, total=len(data)):
        t1_net = time.time()

        # Move batch to cuda
        targets = batch['y']
        targets_chirality = batch['y_chirality_torsions']
        if cuda:
            batch['x'] = batch['x'].cuda()
            batch['edge_index'] = batch['edge_index'].cuda()
            targets = targets.cuda()
            targets_chirality = targets_chirality.cuda()

        # Zero gradients
        model.zero_grad()

        # Generate predictions
        loc_preds, conc_preds, weight_preds, angle_preds, len_preds, chiral_preds, chiral_loc_preds_pos, \
        chiral_loc_preds_neg, chiral_conc_preds_pos, chiral_conc_preds_neg, chiral_weight_preds_pos, \
        chiral_weight_preds_neg = model(batch)

        # Compute loss for rotatable bond torsions without chirality inversion atoms
        loss = loss_fn(batch, targets, loc_preds, conc_preds, weight_preds)
        torsion_loss_sum += loss.item()

        # Compute loss for rotatable bond torsions with chirality inversion atoms
        loss_chirality_torsions = dense_loss_chirality(batch, targets_chirality, chiral_loc_preds_pos,
                                                       chiral_loc_preds_neg, chiral_conc_preds_pos,
                                                       chiral_conc_preds_neg, chiral_weight_preds_pos,
                                                       chiral_weight_preds_neg)
        if loss_chirality_torsions is not None:
            loss += loss_chirality_torsions
            chiral_torsion_loss_sum += loss_chirality_torsions.item()

        # Compute angle loss
        criterion = torch.nn.MSELoss()
        m = torch.where(batch['angle_mask'] == 1)
        tes = [batch['y_angles'][i][:len(torch.where(m[0] == i)[0])].cuda().unsqueeze(1) for i in
               range(batch['y_angles'].shape[0])]
        y_a = torch.cat(tes)
        angle_loss = criterion(angle_preds[m], y_a)
        angle_loss *= 32
        angle_loss_sum += angle_loss.item()
        loss += angle_loss

        # Compute bond length loss
        criterion = torch.nn.MSELoss()
        m = torch.where(batch['len_mask'] == 1)
        tes = [batch['y_lens'][i][:len(torch.where(m[0] == i)[0])].cuda().unsqueeze(1) for i in
               range(batch['y_lens'].shape[0])]
        y_l = torch.cat(tes)
        len_loss = criterion(len_preds[m], y_l)
        len_loss *= 32
        len_loss_sum += len_loss.item()
        loss += len_loss

        # Compute chirality probability loss
        criterion = torch.nn.MSELoss()
        chiral_loss = None
        m = torch.where(batch['chiral_mask'] == 1)
        if chiral_preds[m].shape[0] != 0:
            tes = [batch['y_chirality_prob'][i][:len(torch.where(m[0] == i)[0])].cuda().unsqueeze(1) for i in
                   range(batch['y_chirality_prob'].shape[0])]
            y_c = torch.cat(tes)
            chiral_loss = criterion(chiral_preds[m], y_c)

        if chiral_loss is not None:
            chiral_loss *= 32
            chiral_loss_sum += chiral_loss.item()
            loss += chiral_loss

        loss_sum += loss.item()
        n_iter += targets.shape[0]

        if not predict_only:
            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_cutoff)
            optimizer.step()

        t2_net = time.time()
        net_time += t2_net - t1_net

    avg_loss = loss_sum / n_iter
    torsion_avg_loss = torsion_loss_sum / n_iter
    len_avg_loss = len_loss_sum / n_iter
    angle_avg_loss = angle_loss_sum / n_iter
    chiral_avg_loss = chiral_loss_sum / n_iter
    chiral_torsion_avg_loss = chiral_torsion_loss_sum / n_iter

    t2 = time.time()

    return avg_loss, n_iter, net_time, t1, t2, torsion_avg_loss, len_avg_loss, angle_avg_loss, chiral_avg_loss, \
           chiral_torsion_avg_loss


@click.command()
@click.argument('exp_config_filename', type=str)
@click.argument('exp_extra_name', type=str, default="")
@click.option('-r', '--run-name', type=str, default="")
@click.option('--anomaly', is_flag=True)
def run_training(exp_config_filename: str, exp_extra_name: str, run_name: str = "", anomaly: bool = False):
    """
    Run training for an experiment.

    :param exp_config_filename: yaml file for configuration.
    :param exp_extra_name: Convenient name for the experiment.
    :param run_name: Automatic numeric identifier for the experiment. Set this in order to continue training from an
    existing checkpoint.
    :param anomaly: Whether or not to use torch.autograd.set_detect_anomaly.
    """
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    if not os.path.exists(TB_LOG_DIR):
        os.makedirs(TB_LOG_DIR)

    # Extract the experiment config name
    exp_config_name = os.path.splitext(os.path.basename(exp_config_filename))[0]

    # Generate a default numeric identifier if it is not provided
    if run_name == "":
        run_name = "{:08d}{:04d}".format(int(time.time()) % 100000000, os.getpid() % 10000)

    # Create the total experiment name
    if exp_extra_name != "":
        exp_output_prefix = exp_extra_name + "." + exp_config_name + "." + run_name
    else:
        exp_output_prefix = exp_config_name + "." + run_name

    # Save experiment config file in checkpoint directory
    saved_yaml_filename = os.path.join(CHECKPOINT_DIR, f"{exp_output_prefix}.yaml")

    # If a checkpoint exists for this experiment name, load it. Otherwise, use the provided one.
    print("Looking for", saved_yaml_filename)
    if os.path.exists(saved_yaml_filename):
        load_from_checkpoint = True
        exp_config = yaml.load(open(saved_yaml_filename, 'r'), Loader=yaml.FullLoader)
    else:
        load_from_checkpoint = False
        exp_config = yaml.load(open(exp_config_filename, 'r'), Loader=yaml.FullLoader)

        # Save an exact copy of the yaml
        with open(saved_yaml_filename, 'w') as fid:
            fid.write(open(exp_config_filename, 'r').read())

    print("load_from_checkpoint=", load_from_checkpoint)
    print("exp_output_prefix:", exp_output_prefix)
    train_model(exp_config, exp_output_prefix, load_from_checkpoint, anomaly)


if __name__ == '__main__':
    run_training()
