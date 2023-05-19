import torch
from vonmises.nets import OutputLayer

def test_size_one(set_value):
    """
    Test an output layer with only one input, all param weights at one
    """
    l = OutputLayer(1, 1, 1)
    set_value(l)

    actual_2 = l(torch.tensor([2.])).data
    actual_3 = l(torch.tensor([3.])).data
    actual_batch_2 = l(torch.tensor([[3.], [2.]])).data

    print(actual_2, actual_3, actual_batch_2)

    expected_1_batch = torch.tensor([3.])
    expected_2_batch = torch.tensor([[3.], [3.]])

    assert torch.allclose(actual_2, expected_1_batch)
    assert torch.allclose(actual_3, expected_1_batch)
    assert torch.allclose(actual_batch_2, expected_2_batch)

def test_size_two(set_value):
    """
    Test an output layer with two inputs, all param weights one
    """
    l = OutputLayer(2, 2, 2)
    set_value(l)

    expected_1_batch = torch.tensor([[7.], [7.]])
    expected_2_batch = torch.tensor([[7.,7.],[7.,7.]])

    actual_22 = l(torch.tensor([2., 2.])).data
    actual_23 = l(torch.tensor([2., 3.])).data
    actual_2_batch = l(torch.tensor([[2.,2.],[2.,3.]])).data

    assert torch.allclose(actual_22, expected_1_batch)
    assert torch.allclose(actual_23, expected_1_batch)
    assert torch.allclose(actual_2_batch, expected_2_batch)

def test_increasing(set_inc):
    """
    Test an output layer with two inputs, param weights set increasing
    """
    l = OutputLayer(2, 2, 2)
    set_inc(l)

    expected_0 = torch.tensor([[4.0, 15.0],[7.0, 26.0]])
    actual_0 = l(torch.tensor([[2.,2.], [2.,3.]])).data

    assert torch.allclose(actual_0, expected_0)
    
    
    set_inc(l, 1)

    expected_1 = torch.tensor([[33.0, 72.0],[46.0, 101.0]])
    actual_1 = l(torch.tensor([[2.,2.], [2.,3.]])).data

    assert torch.allclose(actual_1, expected_1)

def test_batch(set_inc):
    b = torch.nn.BatchNorm1d(2)
    l = OutputLayer(2, 2, 2, end_norm=b)
    set_inc(l, 1)

    expected = torch.tensor([[0., 0.], [2.,4.]])
    actual = l(torch.tensor([[2.,2.],[2.,3.]])).data

    assert torch.allclose(actual, expected, atol=2e-03)
