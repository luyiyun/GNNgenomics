#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse


class Config:

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description=("如果上次已经建立了root folder，则提供的source files"
                         "可以只是文件名，而不是完成路径")
        )

        # base
        self.parser.add_argument("--task", default="train",
                                 help="train(default) or test or pred")
        self.parser.add_argument("--device", default="cuda:0",
                                 help="cuda:0(default) or cpu")
        self.parser.add_argument("-rs", "--random_seed", type=int,
                                 default=2019,
                                 help="random seed, default is 2019")

        # datasets args
        self.parser.add_argument("-rn", "--root_name", default="pan_surv",
                                 help="root name, default pan_surv")
        self.parser.add_argument("-s", "--split", default=[0.8, 0.1, 0.2],
                                 type=int, nargs=3,
                                 help=("train val test prop, "
                                       "default are 0.8, 0.1, 0.2,"
                                       "if task is not train, just first "
                                       "is used."))
        self.parser.add_argument("-bs", "--batch_size", default=8, type=int,
                                 help="batch size, default 8")
        self.parser.add_argument(
            "-sd", "--seq_data", default="HiSeqV2",
            help="the genomics seq data path, default pan rnaseq"
        )
        self.parser.add_argument(
            "-cd", "--cli_data", default="PANCAN_clinicalMatrix",
            help="the cli data path, default pan cli"
        )
        self.parser.add_argument(
            "-gd", "--graph_data", default="STRING.csv",
            help="the graph data path, default string"
        )
        self.parser.add_argument(
            "--processed_name", default="surv_tcgaPan_ppi_99",
            help="processed name, default surv_tcgaPan_ppi_99"
        )

        # network architecture
        self.parser.add_argument("-bh", "--block_hiddens", default=[10],
                                 nargs="+",
                                 help="res block channels, default [10]")
        self.parser.add_argument("--act", default="relu")
        self.parser.add_argument("--norm", default="batch")
        self.parser.add_argument("--bias", default=True, type=bool)
        self.parser.add_argument("--conv", default="gcn")
        self.parser.add_argument("--heads", default=1, type=int)
        self.parser.add_argument("--n_blocks", default=10, type=int)
        self.parser.add_argument("--neighbors", default=10, type=int)
        self.parser.add_argument("--block", default="res",
                                 help="the type of blocks, default res")
        self.parser.add_argument("--dropout", default=0.2, type=float,
                                 help="dropout rate, default 0.2")
        self.parser.add_argument("--out_dim", default=1, type=int,
                                 help="output dims, default 1")
        self.parser.add_argument("--prediction_hidden_units", nargs="+",
                                 default=[1024, 512, 256], type=int,
                                 help="the hidden units in prediction")

        # training args
        self.parser.add_argument("-e", "--epoch", type=int, default=100,
                                 help="train epoch")
        self.parser.add_argument("--optimizer", default="Adam",
                                 help="optimizer, default Adam")
        self.parser.add_argument("-lr", "--learning_rate", type=float,
                                 default=0.001,
                                 help="learning rate, default=0.001")

        # pred or eval args
        self.parser.add_argument("-lm", "--load_model", default=None,
                                 help="path of loaded model, default None")

        # results args
        self.parser.add_argument("-t", "--to", default="default_save",
                                 help="save to path, default default_save")
        self.parser.add_argument("--scores", default=["c_index"], nargs="+",
                                 help="evaluation scores, default c_index")
        self.parser.add_argument("-c", "--criterion", default="svm_loss",
                                 help="loss function, default cox_loss")
        self.parser.add_argument("--svm_loss_r", default=1.0, type=float,
                                 help=("if svm_loss, r is the prop of "
                                       "ranking loss, default 1.0"))
        # early stop args
        self.parser.add_argument("--early_stop", action="store_true",
                                 help="use early stop?")
        self.parser.add_argument("--early_stop_score", default="c_index",
                                 help="early stop using this, default c_index")
        self.parser.add_argument(
            "--early_stop_tolerance", default=10, type=int,
            help="the tolerance of early stop, default 10 epoch"
        )

        # visual args
        self.parser.add_argument("--use_visdom", default=True, type=bool,
                                 help="use visdom(for train), default True")
        self.parser.add_argument("-vp", "--visdom_port", default=8097,
                                 type=int,
                                 help="the port of visdom, default 8097")
        self.parser.add_argument("-en", "--env_name", default=None,
                                 help=("the name of enviroment of visdom, "
                                       "default --to"))

    def initialize(self):
        self.args = self.parser.parse_args()
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
        # if don't have raw files, must give source files;
        # if have raw files, source files is not necessary
        bool_raw_exist = os.path.exists(
            os.path.join(self.args.root_name, "raw"))
        bool_source_data = all([f is not None for f in self.args.source_files])
        if (not bool_raw_exist) and (not bool_source_data):
            raise ValueError("one of source files or root_name/raw "
                             "must be existed.")

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
