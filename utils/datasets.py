#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np
import pandas as pd
import torch
import torch_geometric.data as pyg_data
from sklearn.model_selection import train_test_split


def surv_tcgaPan_ppi(source_files, std_filter=0.99):
    tcgaPan_cli, tcgaPan_seq, ppi = source_files
    print("----- reading raw datasets")
    tcgaPan_cli = pd.read_csv(tcgaPan_cli, sep="\t", index_col=0)
    tcgaPan_seq = pd.read_csv(tcgaPan_seq, sep="\t", index_col=0).T
    ppi = pd.read_csv(ppi)

    print("----- preprocessing raw datasets")
    # common samples and same order for cli and seq
    seq_samples, cli_samples = set(tcgaPan_seq.index), set(tcgaPan_cli.index)
    common_samples = seq_samples.intersection(cli_samples)  # cli > seq
    common_samples = list(common_samples)
    tcgaPan_seq = tcgaPan_seq.loc[common_samples, :]
    tcgaPan_cli = tcgaPan_cli.loc[common_samples, :]
    # just primary cancer
    primary_mask = tcgaPan_cli.sample_type == "Primary Tumor"
    tcgaPan_seq = tcgaPan_seq[primary_mask]
    tcgaPan_cli = tcgaPan_cli[primary_mask]
    # just OS and OS time, and remove NA
    tcgaPan_cli = tcgaPan_cli[["_OS", "_OS_IND"]]
    mask = tcgaPan_cli.notna().all(axis=1)
    tcgaPan_cli, tcgaPan_seq = tcgaPan_cli[mask], tcgaPan_seq[mask]
    # remove 20% genes whose sd are small
    genes_std = tcgaPan_seq.agg("std", axis=0)
    genes_mask = genes_std > genes_std.quantile(std_filter)
    tcgaPan_seq = tcgaPan_seq.loc[:, genes_mask]
    # conmmon genes for ppi and seq
    ppi_genes = set(ppi.iloc[:, -1]).union(set(ppi.iloc[:, -2]))
    common_genes = ppi_genes.intersection(set(tcgaPan_seq.columns))
    common_genes = list(common_genes)
    tcgaPan_seq = tcgaPan_seq.loc[:, common_genes]
    mask = ppi.iloc[:, -2:].isin(common_genes).all(axis=1)
    ppi = ppi[mask]
    # ppi genes names -> number
    common_genes = pd.Series(range(len(common_genes)), index=common_genes)
    ppi_number = []
    ppi_number.append(common_genes.loc[ppi.iloc[:, -2]].values)
    ppi_number.append(common_genes.loc[ppi.iloc[:, -1]].values)
    ppi_number.append(ppi.values[:, 0].astype("int64"))
    ppi = np.stack(ppi_number, axis=1)
    # df --> ndarray
    tcgaPan_seq, tcgaPan_cli = tcgaPan_seq.values, tcgaPan_cli.values

    return tcgaPan_cli, tcgaPan_seq, ppi


def cls_tcgaMela_ppi(source_files):
    tcgaPan_cli, tcgaPan_seq, ppi, gene_subsets = source_files
    return


processes_dict = {
    "surv_tcgaPan_ppi": surv_tcgaPan_ppi,
    "cls_tcgaMela_ppi": cls_tcgaMela_ppi,
}


class GenomicsData(pyg_data.Dataset):
    def __init__(
        self, root, source_files=None, pre_transform=None, phase="train",
        process_data_name="processed", val_prop=0.1, test_prop=0.2,
        random_seed=None
    ):
        """
        split stratify is the last columns of cli
        """
        if (test_prop == 0.0) & (val_prop > 0.0):
            raise ValueError("Ensure the existence of test set firstly.")
        if not os.path.exists(root):
            os.mkdir(root)
        self.source_files = source_files
        self.process_data_name = process_data_name
        self.val_prop, self.test_prop = val_prop, test_prop
        self.random_seed = random_seed
        super(GenomicsData, self).__init__(root, None, pre_transform)
        if phase == "train":
            use_data_path = self.processed_paths[1]
        elif phase == "test":
            use_data_path = self.processed_paths[2]
        elif phase == "val":
            use_data_path = self.processed_paths[3]
        else:
            raise ValueError("phase must be one of train, test and val.")
        self.xs, self.ys = torch.load(use_data_path)
        self.edge_index, self.edge_weight = torch.load(self.processed_paths[0])
        self.features_dim = 1 if self.xs.ndim == 2 else self.xs.size(-1)

    def get(self, idx):
        return pyg_data.Data(x=self.xs[idx, :].unsqueeze(-1),
                             y=self.ys[idx, :].unsqueeze(0),
                             edge_index=self.edge_index,
                             edge_weight=self.edge_weight)

    def __len__(self):
        return self.xs.size(0)

    @property
    def raw_file_names(self):
        return [os.path.split(f)[-1] for f in self.source_files]

    @property
    def processed_file_names(self):
        data_names = ["%s_graph.pth", "%s_train.pth"]
        if self.test_prop > 0.0:
            data_names.append("%s_test.pth")
        if self.val_prop > 0.0:
            data_names.append("%s_val.pth")
        return [dn % self.process_data_name for dn in data_names]

    def download(self):
        print("Download...")
        for src, target in zip(self.source_files, self.raw_paths):
            os.symlink(src, target)

    def process(self):
        cli, seq, ppi = self.pre_transform(self.raw_paths)
        assert cli.ndim <= 2
        assert seq.ndim == 2
        assert ppi.ndim == 2
        edge_index = torch.tensor(ppi[:, :2].T).long()
        edge_weight = torch.tensor(ppi[:, 2]).float()
        torch.save([edge_index, edge_weight], self.processed_paths[0])
        # split data
        if self.test_prop > 0.0:
            xs_train, xs_test, ys_train, ys_test = train_test_split(
                seq, cli, test_size=self.test_prop,
                random_state=self.random_seed, shuffle=True,
                stratify=cli[:, -1] if cli.ndim == 2 else cli)
            torch.save(
                [torch.tensor(xs_test).float(),
                 torch.tensor(ys_test).float()],
                self.processed_paths[2])
            if self.val_prop > 0.0:
                val_prop_inner = self.val_prop / (1 - self.test_prop)
                xs_train, xs_val, ys_train, ys_val = train_test_split(
                    xs_train, ys_train, test_size=val_prop_inner,
                    random_state=self.random_seed, shuffle=True,
                    stratify=ys_train[:, -1] if cli.ndim == 2 else cli
                )
                torch.save([torch.tensor(xs_val).float(),
                            torch.tensor(ys_val).float()],
                           self.processed_paths[3])
            torch.save([torch.tensor(xs_train).float(),
                        torch.tensor(ys_train).float()],
                       self.processed_paths[1])
        else:
            torch.save([torch.tensor(seq).float(),
                        torch.tensor(cli).float()],
                       self.processed_paths[1])
