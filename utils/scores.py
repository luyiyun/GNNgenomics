#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from sksurv.metrics import concordance_index_censored as c_index
from sklearn.metrics import roc_auc_score


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


class CPUMetric:
    def __init__(self, cpu_func):
        self.cpu_func = cpu_func

    def init(self):
        self.preds, self.targets = [], []

    def add(self, pred, target):
        self.preds.append(pred)
        self.targets.append(target)

    def value(self):
        preds = torch.cat(self.preds, dim=0).detach().cpu().numpy()
        targets = torch.cat(self.targets, dim=0).detach().cpu().numpy()
        return self.cpu_func(targets, preds)

    def __call__(self, preds, targets):
        preds = preds.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        return self.cpu_func(targets, preds)


class Cindex(CPUMetric):
    def __init__(self):
        def c_index_func(targets, preds):
            return c_index(targets[:, 1].astype("bool"),
                           targets[:, 0],
                           -preds.squeeze())[0]
        super(Cindex, self).__init__(c_index_func)


class AUC(CPUMetric):
    def __init__(self):
        def auc_func(targets, preds):
            targets, preds = targets.squeeze(), preds.squeeze()
            return roc_auc_score(targets, preds)
        super(AUC, self).__init__(auc_func)


scores_dict = {
    "c_index": Cindex,
    "auc": AUC,
}
