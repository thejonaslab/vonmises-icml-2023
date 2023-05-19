""" Test utilities. """
import pytest
import torch
import torch.nn as nn


def fixed_value(p, val=1) -> None:
    """
    Set fixed value.

    :param p: Net parameters.
    :param val: Value.
    """
    for param in p.parameters():
        param.data = nn.parameter.Parameter(val*torch.ones_like(param))


def incremental(p, init_val=0) -> None:
    """
    Set incremental values.

    :param p: Net parameters.
    :param init_val: Initial value.
    """
    for param in p.parameters():
        z = torch.tensor([float(i+init_val) for i in range(param.nelement())])
        param.data = nn.parameter.Parameter(torch.reshape(z, param.shape))


@pytest.fixture
def set_value():
    """
    Set value.
    """
    return fixed_value


@pytest.fixture 
def set_inc():
    """
    Set incremental.
    """
    return incremental
