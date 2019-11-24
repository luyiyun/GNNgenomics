#!/usr/bin/env python
# -*- coding: utf-8 -*-

from config import Config


class PanSurvConfig(Config):
    def __init__(self):
        super(PanSurvConfig, self).__init__()

        # datasets args
        self.parser.add_argument("-sd", "--seq_data",
                            default="/home/dl/NewDisk/TCGA/Pan/HiSeqV2",
                            help="the genomics seq data path, default pan rnaseq"
                            )
        self.parser.add_argument("-cd", "--cli_data",
                            default="/home/dl/NewDisk/TCGA/Pan/PANCAN_clinicalMatrix",
                            help="the cli data path, default pan cli")
        self.parser.add_argument("-gd", "--graph_data",
                            default="/home/dl/NewDisk/Graph/STRING.csv",
                            help="the graph data path, default string")
        self.parser.add_argument("--processed_name", default="surv_tcgaPan_ppi",
                            help="processed name, default surv_tcgaPan_ppi")
