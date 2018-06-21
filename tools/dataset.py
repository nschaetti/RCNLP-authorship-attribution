#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

# Imports
import torchlanguage.transforms
import torch


#########################################
# Dataset
#########################################


# Load dataset
def load_dataset(dataset_size=100, dataset_start=0):
    """
    Load dataset
    :return:
    """
    # Load from directory
    reutersc50_dataset = torchlanguage.datasets.ReutersC50Dataset(
        n_authors=15,
        download=True,
        dataset_size=dataset_size,
        dataset_start=dataset_start
    )

    # Reuters C50 dataset training
    reuters_loader_train = torch.utils.data.DataLoader(
        torchlanguage.utils.CrossValidation(reutersc50_dataset),
        batch_size=1,
        shuffle=True
    )

    # Reuters C50 dataset test
    reuters_loader_test = torch.utils.data.DataLoader(
        torchlanguage.utils.CrossValidation(reutersc50_dataset, train=False),
        batch_size=1,
        shuffle=True
    )
    return reutersc50_dataset, reuters_loader_train, reuters_loader_test
# end load_dataset
