#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from sksurv.metrics import concordance_index_censored as c_index


class Loss:
    """
    summary Loss, please ensure the tensors used have been detached.
    """
    def __init__(self):
        self.init()

    def init(self):
        self.loss_all = 0.
        self.count = 0

    def add(self, batch_loss, batch_size):
        if torch.isnan(batch_loss).item():
            return
        self.loss_all += batch_loss * batch_size
        self.count += batch_size

    def value(self):
        return (self.loss_all / self.count).detach().cpu().item()


class C_index:
    def __init__(self):
        self.init()

    def init(self):
        self.preds, self.targets = [], []

    def add(self, pred, target):
        self.preds.append(pred)
        self.targets.append(target)

    def value(self):
        preds = torch.cat(self.preds, dim=0).detach().cpu().numpy()
        targets = torch.cat(self.targets, dim=0).detach().cpu().numpy()
        return c_index(targets[:, 1].astype("bool"), targets[:, 0],
                       preds.squeeze())[0]

    def __call__(self, preds, targets):
        preds = preds.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        return c_index(targets[:, 1].astype("bool"), targets[:, 0],
                       preds.squeeze())[0]


scores_dict = {
    "c_index": C_index
}
