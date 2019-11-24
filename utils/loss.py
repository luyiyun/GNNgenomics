#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class CoxLoss(nn.Module):

    def __init__(self, reduction="mean"):
        super(CoxLoss, self).__init__()
        if reduction == "mean":
            self.reduction_fn = torch.mean
        elif reduction == "sum":
            self.reduction_fn = torch.sum
        else:
            raise NotImplementedError("%s is not implemented." % reduction)

    def forward(self, pred, target):
        pred, time, event = pred.squeeze(-1), target[:, 0], target[:, 1]
        time = time.float()
        time_index = time.argsort(descending=True)
        try:
            pred, event = pred[time_index], event[time_index]
        except IndexError:
            import ipdb; ipdb.set_trace()
        pred_exp_cumsum = pred.exp().cumsum(0)
        ls = (pred / pred_exp_cumsum.log())[event == 1]
        return -self.reduction_fn(ls)


class SVMLoss(nn.Module):
    def __init__(self, r=1, reduction="mean"):
        super(SVMLoss, self).__init__()
        self.r = r
        if reduction == "mean":
            self.reduction_fn = torch.mean
        elif reduction == "sum":
            self.reduction_fn = torch.sum
        else:
            raise NotImplementedError("%s is not implemented." % reduction)

    def forward(self, pred, target):
        pred, time, event = pred.squeeze(-1), target[:, 0], target[:, 1]
        time = time.float()
        regression_loss = self._regression(pred, time, event)
        ranking_loss = self._ranking(pred, time, event)
        return self.r * ranking_loss + (1 - self.r) * regression_loss

    def _regression(self, pred, time, event):
        diff = pred - time
        diff[event == 0] = diff[event == 0].clamp(0.)
        return self.reduction_fn(diff ** 2)

    def _ranking(self, pred, time, event):
        n = pred.size(0)
        index = torch.arange(n)
        index1, index2 = index.repeat(n), index.repeat_interleave(n)
        mask1 = time[index1] > time[index2]
        mask2 = (event[index2] == 1)
        mask = mask1 & mask2
        diff = 1. - pred[index1[mask]] + pred[index2[mask]]
        return self.reduction_fn(diff.clamp(0.).pow(2))


losses_dict = {
    "cox_loss": CoxLoss,
    "svm_loss": SVMLoss
}
