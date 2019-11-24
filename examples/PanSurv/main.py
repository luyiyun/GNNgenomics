#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from os.path import dirname

import torch
import torch_geometric.data as pyg_data
import torch.optim as optim

ROOT_DIR = dirname(dirname(dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)
from utils.datasets import GenomicsData
from utils.loss import losses_dict
from utils.scores import scores_dict
from architecture import DeepDynGNN
from train import train_val_test, save_generally, eval_one_epoch
from opts import PanSurvConfig
from processes import processes_dict


def main():
    # load config
    config = PanSurvConfig()
    opts = config.initialize()
    config.save(os.path.join(opts.to, "config.json"))
    print(config)

    # prepare dataset
    pre_transform = processes_dict[opts.processed_name]
    datasets = {}
    if opts.task == "train":
        datasets["train"] = GenomicsData(
            opts.root_name, opts.source_files, pre_transform, "train",
            opts.processed_name, val_prop=opts.split[1],
            test_prop=opts.split[2], random_seed=opts.random_seed
        )
    else:
        assert opts.split[1] == 0.0
        assert opts.split[2] == 0.0
        datasets["eval"] = GenomicsData(
            opts.root_name, opts.source_files, pre_transform, "train",
            opts.processed_name, val_prop=opts.split[1],
            test_prop=opts.split[2], random_seed=opts.random_seed
        )
    if opts.split[2] > 0.0:
        datasets["test"] = GenomicsData(
            opts.root_name, opts.source_files, pre_transform, "test",
            opts.processed_name, val_prop=opts.split[1],
            test_prop=opts.split[2], random_seed=opts.random_seed
        )
    if opts.split[1] > 0.0:
        datasets["val"] = GenomicsData(
            opts.root_name, opts.source_files, pre_transform, "val",
            opts.processed_name, val_prop=opts.split[1],
            test_prop=opts.split[2], random_seed=opts.random_seed
        )
    in_dim = datasets["train"].features_dim
    dataloaders = {k: pyg_data.DataLoader(dat, batch_size=opts.batch_size,
                                          shuffle=(k == "train"))
                   for k, dat in datasets.items()}

    # networks
    if opts.load_model is not None:
        model = torch.load(opts.load_model)
    else:
        model = DeepDynGNN(in_dim, opts)

    # criterion
    criterion = losses_dict[opts.criterion.lower()]()

    # scores
    scores = [scores_dict[s]() for s in opts.scores]

    if opts.task == "train":
        if opts.optimizer.lower() == "adam":
            optimizer = optim.Adam(model.parameters(), opts.learning_rate)
        else:
            raise NotImplementedError("%s is not implemented." %
                                      opts.optimizer)

        # train model
        model, hist = train_val_test(
            model, criterion, optimizer, dataloaders, scores, opts)

        # save model
        save_generally(model, os.path.join(opts.to, "model.pth"))
        save_generally(hist, os.path.join(opts.to, "hist.json"))

    elif opts.task == "eval":
        assert opts.load_model is not None
        # predict model
        eval_loss, eval_scores, pred, target = eval_one_epoch(
            model, criterion, dataloaders["eval"], scores, opts.device)
        print("eval loss is : %.4f" % eval_loss)
        for k, v in eval_scores.items():
            print("eval %s is : %.4f" % (k, v))
        # save pred
        eval_scores.update({"loss": eval_loss})
        save_generally(eval_scores, os.path.join(opts.to, "eval_res.json"))
        save_generally(pred, os.path.join(opts.to, "pred.txt"))
        save_generally(target, os.path.join(opts.to, "target.txt"))

    else:
        raise NotImplementedError("task must be one of train, pred and eval.")


if __name__ == '__main__':
    main()
