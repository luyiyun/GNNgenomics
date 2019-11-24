#!/usr/bin/env python
# -*- coding: utf-8 -*-

import inspect
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch_geometric as tg

from .basic import act_layer, norm_layer


def remove_duplicated_edges(edge_index1, edge_index2):
    """
    remove duplicated edges in edge_index2 for edge_index1
    """
    # get indexes for two indexes
    try:
        max1 = edge_index1.max() + 1
        index1 = edge_index1[1] * max1 + edge_index1[0]
        max2 = edge_index2.max()+1
        index2 = edge_index2[1] * max2 + edge_index2[0]
        mask = np.isin(index2.cpu().numpy(), index1.cpu().numpy())
        edge_index2_nodup = edge_index2[:, ~mask].to(edge_index1)
    except RuntimeError:
        import ipdb; ipdb.set_trace()
    return torch.cat([edge_index1, edge_index2_nodup], dim=1)


def add_knn_graph(x, edge_index, batch=None, edge_weight=None, k=None):
    if k is None or k == 0:
        return edge_index

    knn_edge_index = tg.nn.knn_graph(x, k, batch, loop=False)
    new_edge_index = remove_duplicated_edges(edge_index, knn_edge_index)
    if edge_weight is None:
        return new_edge_index

    knn_edge_weight = torch.full(
        (new_edge_index.size(1) - edge_index.size(1),),
        edge_weight.min()).to(edge_weight)
    new_edge_weight = torch.cat([edge_weight, knn_edge_weight])
    return new_edge_index, new_edge_weight


class StaticGNN(nn.Module):
    def __init__(
        self, in_channels, out_channels, conv="gcn",
        bias=True, act="relu", norm=None,
        **kwargs
    ):
        super(StaticGNN, self).__init__()
        if conv.lower() == "gcn":
            self.gconv = tg.nn.GCNConv(
                in_channels, out_channels, bias=bias, **kwargs)
        elif conv.lower() == "cheb":
            self.gconv = tg.nn.ChebConv(
                in_channels, out_channels, bias=bias, **kwargs)
        elif conv.lower() == "gat":
            if 'head' not in kwargs.keys():
                raise ValueError("use gat conv must give head args.")
            head = kwargs["head"]
            self.gconv = tg.nn.GATConv(
                in_channels, out_channels//head, bias=bias, **kwargs)
        elif conv.lower() == "sage":
            self.gconv = tg.nn.SAGEConv(
                in_channels, out_channels, bias=bias, **kwargs)
        else:
            raise NotImplementedError("conv {%s} is not implemented" % conv)
        # get conv forward args
        self.conv_forward_args = inspect.getfullargspec(self.gconv.forward)[0]
        self.conv_name = conv
        # unlinear part
        m = []
        if norm is not None:
            m.append(norm_layer(norm, out_channels))
        if act is not None:
            m.append(act_layer(act))
        self.unlinear = nn.Sequential(*m)

    def forward(self, x, edge_index, batch=None, edge_weight=None):
        forward_args = {}
        if "batch" in self.conv_forward_args:
            forward_args['batch'] = batch
        if "edge_weight" in self.conv_forward_args:
            forward_args['edge_weight'] = edge_weight
        elif edge_weight is not None:
            warnings.warn("%s doesn't have edge_weight args, but give it."
                          % self.conv)
        out = self.unlinear(
            self.gconv(x=x, edge_index=edge_index, **forward_args))
        return out


class DynamicGNN(StaticGNN):
    def __init__(
        self, in_channels, out_channels, conv="gcn",
        bias=True, act="relu", norm=None, neighbors=10,
        **kwargs
    ):
        super(DynamicGNN, self).__init__(
            in_channels, out_channels, conv="gcn",
            bias=True, act="relu", norm=None, **kwargs
        )
        self.neighbors = neighbors

    def forward(self, x, edge_index, batch=None, edge_weight=None):
        if self.neighbors > 0:
            edge_index = add_knn_graph(
                x, edge_index, batch, edge_weight, self.neighbors)
            if edge_weight is not None:
                edge_index, edge_weight = edge_index
        out = super(DynamicGNN, self).forward(
            x, edge_index, batch, edge_weight)
        return out


class ResDynamicBlock(nn.Module):
    def __init__(
        self, channels, neighbors=10, conv="gcn", norm=None,
        bias=True, act="relu", res_scale=1, **kwargs
    ):
        super(ResDynamicBlock, self).__init__()
        if isinstance(channels, int):
            channels = [channels, channels]
        elif isinstance(channels, (list, tuple)):
            channels = list(channels)
            channels.append(channels[0])
        else:
            raise ValueError("channles must be int , list or tuple.")
        self.ms = nn.ModuleList()
        for i in range(1, len(channels)):
            layer = DynamicGNN(
                channels[i-1], channels[i], conv, bias, act, norm,
                neighbors, **kwargs)
            self.ms.append(layer)
        self.res_scale = res_scale

    def forward(self, x, edge_index, batch=None, edge_weight=None):
        x_clone = x.clone()
        for m in self.ms:
            x = m(x, edge_index, batch, edge_weight)
        return x + x_clone * self.res_scale
