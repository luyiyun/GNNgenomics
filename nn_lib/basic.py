#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import nn
from torch.nn import Sequential as Seq, Linear as Lin


def act_layer(act_layer, inplace=False, neg_slope=0.2, n_prelu=1):
    """
    convenient function to get act
    """
    act = act_layer.lower()
    if act == "relu":
        layer = nn.ReLU(inplace)
    elif act == "leakyrelu":
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == "prelu":
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError("activation layer [%s] is not found" % act)
    return layer


def norm_layer(norm_type, nc):
    """
    convenient function to get norm layer
    """
    norm = norm_type.lower()
    if norm == "batch":
        layer = nn.BatchNorm1d(nc)
    elif norm == "instance":
        layer = nn.InstanceNorm1d(nc)
    else:
        raise NotImplementedError(
            "normalization layer [%s] is not found" % norm)
    return layer


class MultiSeq(Seq):
    """
    a sequential class can receive several inputs
    """
    def __init__(self, *args):
        super(MultiSeq, self).__init__(*args)

    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class MLP(Seq):
    """
    linear --> act --> norm MLP
    """
    def __init__(self, channels, act="relu", norm=None, bias=True):
        m = []
        for i in range(1, len(channels)):
            m.append(Lin(channels[i - 1], channels[i], bias))
            if norm is not None:
                m.append(norm_layer(norm, channels[-1]))
            if act is not None:
                m.append(act_layer(act))
        self.m = m
        super(MLP, self).__init__(*self.m)
