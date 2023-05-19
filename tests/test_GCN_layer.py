""" Test GCN layer. """
import torch
import numpy as np

from vonmises.nets import GCNLayer


def test_basic(set_value):
    """Test a single GCN layer"""
    # noinspection PyArgumentEqualDefault
    l_dense = GCNLayer(1, num_vertices=3, reduce='mean', order=True)
    set_value(l_dense)

    G = np.array([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]], dtype=np.float32)

    v_i_in = torch.tensor([[1.], [2.], [3.]])

    actual_out_dense = l_dense(torch.unsqueeze(v_i_in, 0), G=torch.tensor([G]))
    expected_out = torch.tensor([[5.5], [6.], [6.5]])

    assert torch.allclose(actual_out_dense, torch.unsqueeze(expected_out, 0))


def test_extra_linears(set_value):
    """Test a single GCN layer followed by 2 extra linear layers"""
    # noinspection PyArgumentEqualDefault
    l_dense = GCNLayer(1, num_vertices=3, reduce='sum', extra_layers=2, order=True)
    set_value(l_dense)

    G = np.array([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]], dtype=np.float32)

    v_i_in = torch.tensor([[1.], [2.], [3.]])

    actual_dense = l_dense(torch.unsqueeze(v_i_in, 0), G=torch.tensor([G]))
    expected_extra = torch.tensor([[13.], [14.], [15.]])

    assert torch.allclose(actual_dense, torch.unsqueeze(expected_extra, 0))


def test_all_extras(set_inc):
    """Test a single dense GCN layer followed by 2 extra linear layers with instance norms and a final batch norm"""
    # noinspection PyArgumentEqualDefault
    l_dense = GCNLayer(2, num_vertices=3, reduce='sum', extra_layers=2, extra_norm='layer', end_norm='layer',
                       order=True)
    set_inc(l_dense)

    G_dense = torch.tensor([[[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]], [[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]]])
    v_i_in_dense = torch.tensor([[[1., 2.], [2., 3.], [3., 4.]], [[4., 5.], [5., 6.], [6., 7.]]])

    actual_dense = l_dense(v_i_in_dense, G=G_dense)
    expected_dense = torch.reshape(torch.tensor([[0., 2.], [0., 2.], [0., 2.], [0., 2.], [0., 2.], [0., 2.]]),
                                   (2, 3, 2))

    assert torch.allclose(actual_dense, expected_dense, atol=1e-05)


def test_batch_differences(set_inc):
    """Using all extra as above, demonstrate differences in batch norm between dense and sparse models"""
    # noinspection PyArgumentEqualDefault
    l_dense = GCNLayer(2, num_vertices=3, reduce='sum', extra_layers=2, extra_norm='layer', end_norm='batch',
                       order=True)
    set_inc(l_dense)

    G_dense = torch.tensor([[[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]], [[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]]])
    v_i_in_dense = torch.tensor([[[1., 2.], [2., 3.], [3., 4.]], [[4., 5.], [5., 6.], [6., 7.]]])

    actual_dense = l_dense(v_i_in_dense, G=G_dense)
    expected_dense = torch.tensor([[[0., 0.], [0., 2.], [0., 4.]], [[0., 0.], [0., 2.], [0., 4.]]])

    assert torch.allclose(actual_dense, expected_dense, atol=1e-04)


# noinspection PyArgumentList
def test_reduce_mean_with_zero(set_value):
    """Testing the proper handling of mean when a node has zero connections"""
    # noinspection PyArgumentEqualDefault
    l_dense = GCNLayer(1, num_vertices=3, reduce='sum', extra_layers=2, order=True)
    set_value(l_dense)

    G = np.array([[0., 0., 1.], [0., 0., 0.], [1., 0., 0.]], dtype=np.float32)

    v_i_in = torch.tensor([[1.], [2.], [3.]])

    actual_dense = l_dense(torch.unsqueeze(v_i_in, 0), G=torch.tensor([G]))
    expected_extra = torch.tensor([[8.], [2.], [8.]])

    assert torch.allclose(actual_dense, torch.unsqueeze(expected_extra, 0))
