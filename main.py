#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
如果需要添加一个新的数据集，则需要添加在processes中定义一个函数，并将其放入
dict processes_dict中，key是processes_name，这个函数需要返回3个ndarray，依次是
cli、seq、graph，其中：
    cli是表示预测的标签，可以是一列，也可以是多列；
    seq表示使用的基因，X，必须是2维矩阵；
    graph表示使用的graph，2维，必须是Nx2或Nx3的，其中N表示边的数量，多出的第3列
        表示边的权重；

如果希望添加新的loss和score，则需要在utils.loss和utils.scores中添加相应的对象，
    并将其添加到losses_dict或scores_dict中
"""

import os

import torch
import torch_geometric.data as pyg_data
import torch.optim as optim

from utils import GenomicsData, losses_dict, scores_dict
from processes import processes_dict
from architecture import DeepDynGNN
from train import train_val_test, save_generally, eval_one_epoch
from config import Config
from tradition import traditional_surv_analysis


def main():
    # load config
    config = Config()
    opts = config.initialize()
    config.save(os.path.join(opts.to, "config.json"))
    print(config)

    # prepare dataset
    pre_transform = processes_dict[opts.processed_name]
    datasets = {}
    if opts.task in ["train", "tradition"]:
        datasets["train"] = GenomicsData(
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
    else:
        datasets["eval"] = GenomicsData(
            opts.root_name, opts.source_files, pre_transform, "train",
            opts.processed_name, val_prop=opts.split[1],
            test_prop=opts.split[2], random_seed=opts.random_seed
        )
    in_dim = datasets["train"].features_dim
    num_nodes = datasets["train"].num_nodes
    dataloaders = {k: pyg_data.DataLoader(dat, batch_size=opts.batch_size,
                                          shuffle=(k == "train"))
                   for k, dat in datasets.items()}

    # tranditional evaulation:
    if opts.task == "tradition":
        train_scores, test_scores = traditional_surv_analysis(datasets, opts)
        save_generally([train_scores, test_scores],
                       os.path.join(opts.to, "tradition.json"))
        print("")
        print("train:")
        print(train_scores)
        print("test:")
        print(test_scores)
        return

    # networks
    if opts.load_model is not None:
        model = torch.load(opts.load_model)
    else:
        model = DeepDynGNN(in_dim, num_nodes, opts)

    # criterion
    kwargs = {}
    if opts.criterion.lower() == "svm_loss":
        kwargs["r"] = opts.svm_loss_r
    criterion = losses_dict[opts.criterion.lower()](**kwargs)

    # scores
    scores = {s: scores_dict[s]() for s in opts.scores}

    if opts.task == "train":
        if opts.optimizer.lower() == "adam":
            optimizer = optim.Adam(model.parameters(), opts.learning_rate)
        elif opts.optimizer.lower() == "adamw":
            optimizer = optim.AdamW(model.parameters(), opts.learning_rate)
        elif opts.optimizer.lower() == "adammax":
            optimizer = optim.Adamax(model.parameters(), opts.leanring_rate)
        elif opts.optimizer.lower() == "rms":
            optimizer = optim.RMSprop(model.parameters(), opts.learning_rate)
        elif opts.optimizer.lower() == "momentum":
            optimizer = optim.SGD(model.parameters(), opts.learning_rate,
                                  momentum=0.9)
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


if __name__ == '__main__':
    main()
