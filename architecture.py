#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch

from nn_lib import StaticGNN, ResDynamicBlock, MLP


class DeepDynGNN(nn.Module):
    def __init__(self, in_dim, opts):
        super(DeepDynGNN, self).__init__()
        # read some args from opts
        block_hiddens = opts.block_hiddens
        if isinstance(block_hiddens, int):
            block_hiddens = [block_hiddens]
        act = opts.act
        norm = opts.norm
        bias = opts.bias
        conv = opts.conv.lower()
        heads = opts.heads
        n_blocks = opts.n_blocks
        neighbors = opts.neighbors
        # for special conv, another args will be needed
        kwargs = {}
        if conv == "gat":
            kwargs["head"] = heads
        # first layer, static GNN
        self.head = StaticGNN(
            in_dim, block_hiddens[0], conv=conv, bias=bias, act=act,
            norm=norm, **kwargs)
        # ResBlocks
        res_scale = 1 if opts.block.lower() == "res" else 0
        self.backbone_list = nn.ModuleList()
        for _ in range(n_blocks):
            self.backbone_list.append(
                ResDynamicBlock(block_hiddens, neighbors, conv=conv,
                                norm=norm, bias=bias, act=act,
                                res_scale=res_scale, **kwargs)
            )
        # prediction
        fusion_dims = block_hiddens[0] * (n_blocks + 1)
        units = [fusion_dims] + opts.prediction_hidden_units + [opts.out_dim]
        MLPs = []
        for i in range(len(units) - 1):
            MLPs.append(MLP([units[i], units[i+1]], act, norm, bias))
            if i < (len(units) - 2):
                MLPs.append(nn.Dropout(opts.dropout))
        self.prediction = nn.Sequential(*MLPs)

    def forward(self, dat):
        x, edge_index, batch, edge_weight = dat.x, dat.edge_index, \
            dat.batch, dat.edge_weight
        feats = [self.head(x, edge_index, batch, edge_weight)]
        for block in self.backbone_list:
            feats.append(block(feats[-1], edge_index, batch, edge_weight))
        feats = torch.cat(feats, 1)
        feats, _ = to_dense_batch(feats, batch)  # B x N x F
        feats = torch.mean(feats, 1)  # B X F
        out = self.prediction(feats)  # B x 1
        return out

