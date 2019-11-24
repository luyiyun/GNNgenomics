#!/usr/bin/env python
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import torch


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


processes_dict = {
    "surv_tcgaPan_ppi": surv_tcgaPan_ppi,
}


