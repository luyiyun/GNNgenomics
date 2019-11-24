#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from copy import deepcopy

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from utils.scores import Loss
from utils.visual import VisObj


def train_one_batch(model, criterion, optimizer, batch):
    """ outputs have been detached to prepare for next. """
    optimizer.zero_grad()
    batch_out = model(batch)
    batch_loss = criterion(batch_out, batch.y)
    batch_loss.backward()
    optimizer.step()
    return model, batch_loss.detach(), batch_out.squeeze(-1).detach()


def train_one_epoch(
    model, criterion, optimizer, train_dataloader, scores, device="cuda:0"
):
    """ Loss should be in scores if loss is necessary. """
    loss_sum = Loss()
    for s in scores:  # init score
        s.init()
    for batch in tqdm(train_dataloader, "train: "):
        batch = batch.to(torch.device(device))
        _, batch_loss, batch_out = train_one_batch(
            model, criterion, optimizer, batch)
        loss_sum.add(batch_loss, batch.num_graphs)
        for score in scores:
            score.add(batch_out, batch.y)
    return (model, loss_sum.value(),
            {s.__class__.__name__.lower(): s.value() for s in scores})


def eval_one_epoch(
    model, criterion, eval_dataloader, scores, device="cuda:0"
):
    epoch_pred = []
    epoch_target = []
    loss_sum = Loss()
    for s in scores:
        s.init()
    for batch in tqdm(eval_dataloader, "eval: "):
        batch = batch.to(torch.device(device))
        batch_out = model(batch)
        batch_loss = criterion(batch_out, batch.y)
        epoch_pred.append(batch_out.detach())
        epoch_target.append(batch.y)
        loss_sum.add(batch_loss, batch.num_graphs)

    epoch_pred = torch.cat(epoch_pred, dim=0)
    epoch_target = torch.cat(epoch_target, dim=0)
    values_scores = {s.__class__.__name__.lower():
                     s(epoch_pred, epoch_target)
                     for s in scores}
    epoch_loss = loss_sum.value()
    return epoch_loss, values_scores, epoch_pred, epoch_target


def train_val_test(model, criterion, optimizer, dataloaders, scores, opts):
    # store results
    history = {}
    for k in dataloaders.keys():
        history[k] = {"loss": []}
        for s in scores:
            s_name = s.__class__.__name__.lower()
            history[k][s_name] = []
    # the position
    model = model.to(torch.device(opts.device))
    # visdom
    if opts.use_visdom:
        vis = VisObj(port=opts.visdom_port, env=opts.env_name)
    # early_stop_dict
    if opts.early_stop:
        assert "val" in dataloaders.keys()
        early_stop_dict = {
            "best_score": -np.Inf, "best_epoch": -1, "tolerance": 0,
            "best_model": deepcopy(model.state_dict())
        }

    for e in tqdm(range(opts.epoch), "epoch: "):
        # train phase
        history_part = history["train"]
        model.train()
        with torch.enable_grad():
            _, epoch_loss, epoch_scores = train_one_epoch(
                model, criterion, optimizer,
                dataloaders["train"], scores, opts.device)
            history_part["loss"].append(epoch_loss)
            for s_name, s_value in epoch_scores.items():
                history_part[s_name].append(s_value)

        # eval phase
        if "val" in dataloaders.keys():
            model.eval()
            history_part = history["val"]
            with torch.no_grad():
                epoch_loss, epoch_scores, _, _ = eval_one_epoch(
                    model, criterion, dataloaders["val"], scores, opts.device)
                history_part["loss"].append(epoch_loss)
                for s_name, s_value in epoch_scores.items():
                    history_part[s_name].append(s_value)

        # visdom
        if opts.use_visdom:
            vis.line(e, history, ["train", "val"],
                     metrics=list(history_part.keys()))
        # lr schedule

        # early stop
        if opts.early_stop:
            now_score = history["val"][opts.early_stop_score][-1]
            if now_score > early_stop_dict["best_score"]:
                early_stop_dict['best_score'] = now_score
                early_stop_dict["best_epoch"] = e
                early_stop_dict["tolerance"] = 0
                early_stop_dict["best_model"] = deepcopy(model.state_dict())
            else:
                early_stop_dict["tolerance"] += 1
            if early_stop_dict["tolerance"] > opts.early_stop_tolerance:
                model.load_state_dict(early_stop_dict["best_model"])
                early_stop_dict.pop("best_model")
                print("\nEarly Stop!")
                for k, v in early_stop_dict.items():
                    print("%s: %s" % (k, v))
                history.update(early_stop_dict)
                break

    # test phase
    if "test" in dataloaders.keys():
        model.eval()
        history_part = history["test"]
        with torch.no_grad():
            epoch_loss, epoch_scores, _, _ = eval_one_epoch(
                model, criterion, dataloaders["val"], scores, opts.device)
            history_part["loss"].append(epoch_loss)
            for s_name, s_value in epoch_scores.items():
                history_part[s_name].append(s_value)

    return model, history


def save_generally(obj, filename):
    if isinstance(obj, nn.Module):
        torch.save(obj, filename)
    if isinstance(obj, (dict, list, tuple)):
        with open(filename, "w") as f:
            json.dump(obj, f)
    if isinstance(obj, torch.Tensor):
        obj = obj.detach().cpu().numpy()
        np.savetxt(filename, obj)
    if isinstance(obj, np.ndarray):
        np.savetxt(filename, obj)
