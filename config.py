#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse


class Config:

    def __init__(self):
        parser = argparse.ArgumentParser()

        # base
        parser.add_argument("--task", default="train",
                            help="train(default) or test or pred")
        parser.add_argument("--device", default="cuda:0",
                            help="cuda:0(default) or cpu")
        parser.add_argument("-rs", "--random_seed", type=int, default=2019,
                            help="random seed, default is 2019")

        # datasets args
        parser.add_argument("-sd", "--seq_data",
                            default="/home/dl/NewDisk/TCGA/Pan/HiSeqV2",
                            help="the genomics seq data path, default pan rnaseq"
                            )
        parser.add_argument("-cd", "--cli_data",
                            default="/home/dl/NewDisk/TCGA/Pan/PANCAN_clinicalMatrix",
                            help="the cli data path, default pan cli")
        parser.add_argument("-gd", "--graph_data",
                            default="/home/dl/NewDisk/Graph/STRING.csv",
                            help="the graph data path, default string")
        parser.add_argument("-rn", "--root_name", default="pan_surv",
                            help="root name, default pan_surv")
        parser.add_argument("--processed_name", default="surv_tcgaPan_ppi",
                            help="processed name, default surv_tcgaPan_ppi")
        parser.add_argument("-s", "--split", default=[0.8, 0.1, 0.2],
                            type=int, nargs=3,
                            help=("train val test prop, "
                                  "default are 0.8, 0.1, 0.2"))
        parser.add_argument("-bs", "--batch_size", default=8, type=int,
                            help="batch size, default 8")

        # network architecture
        parser.add_argument("-bh", "--block_hiddens", default=[10], nargs="+",
                            help="res block channels, default [10]")
        parser.add_argument("--act", default="relu")
        parser.add_argument("--norm", default="batch")
        parser.add_argument("--bias", default=True, type=bool)
        parser.add_argument("--conv", default="gcn")
        parser.add_argument("--heads", default=1, type=int)
        parser.add_argument("--n_blocks", default=10, type=int)
        parser.add_argument("--neighbors", default=10, type=int)
        parser.add_argument("--block", default="res",
                            help="the type of blocks, default res")
        parser.add_argument("--dropout", default=0.2, type=float,
                            help="dropout rate, default 0.2")
        parser.add_argument("--out_dim", default=1, type=int,
                            help="output dims, default 1")

        # training args
        parser.add_argument("-e", "--epoch", type=int, default=100,
                            help="train epoch")
        parser.add_argument("--optimizer", default="Adam",
                            help="optimizer, default Adam")
        parser.add_argument("-lr", "--learning_rate", type=float,
                            default=0.001, help="learning rate, default=0.001")

        # pred or eval args
        parser.add_argument("-lm", "--load_model", default=None,
                            help="path of loaded model, default None")

        # results args
        parser.add_argument("-t", "--to", default="default_save",
                            help="save to path, default default_save")
        parser.add_argument("--scores", default=["c_index"], nargs="+",
                            help="evaluation scores, default c_index")
        parser.add_argument("-c", "--criterion", default="cox_loss",
                            help="loss function, default cox_loss")
        # early stop args
        parser.add_argument("--early_stop", action="store_true",
                            help="use early stop?")
        parser.add_argument("--early_stop_score", default="c_index",
                            help="early stop using this, default c_index")
        parser.add_argument("--early_stop_tolerance", default=10, type=int,
                            help="the tolerance of early stop, default 10 epoch")

        # visual args
        parser.add_argument("--use_visdom", action="store_true",
                            help="use visdom(just for train)?")
        parser.add_argument("-vp", "--visdom_port", default=8097, type=int,
                            help="the port of visdom, default 8097")
        parser.add_argument("-en", "--env_name", default=None,
                            help=("the name of enviroment of visdom, "
                                  "default --to"))

        self.args = parser.parse_args()

    def initialize(self):
        # make dir
        if not os.path.exists("./RESULTS"):
            os.mkdir("./RESULTS")
        if not os.path.exists("./DATA"):
            os.mkdir("./DATA")
        # root_name
        self.args.root_name = os.path.join("./DATA", self.args.root_name)
        # visdom
        self.args.env_name = (self.args.to if self.args.env_name is None
                              else self.args.env_name)
        # save
        self.args.to = os.path.join("./RESULTS", self.args.to)
        if not os.path.exists(self.args.to):
            os.mkdir(self.args.to)
        # source_files
        self.args.source_files = [
            self.args.cli_data, self.args.seq_data, self.args.graph_data]

        return self.args

    def save(self, path):
        if os.path.exists(path):
            with open(path, "r") as f:
                content = json.load(f)
            content.append(self.args.__dict__)
            with open(path, "w") as f:
                json.dump(content, f)
        else:
            with open(path, "w") as f:
                json.dump([self.args.__dict__], f)

    def __repr__(self):
        strings = ""
        for k, v in self.args.__dict__.items():
            if isinstance(v, int):
                strings += ("%s: %d\n" % (k, v))
            elif isinstance(v, float):
                strings += ("%s: %.4f\n" % (k, v))
            else:
                strings += ("%s: %s\n" % (k, v))
        return strings
