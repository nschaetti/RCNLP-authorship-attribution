#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File : core.classifiers.RCNLPTextClassifier.py
# Description : Echo State Network for text classification.
# Auteur : Nils Schaetti <nils.schaetti@unine.ch>
# Date : 01.02.2017 17:59:05
# Lieu : Nyon, Suisse
#
# This file is part of the Reservoir Computing NLP Project.
# The Reservoir Computing Memory Project is a set of free software:
# you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Foobar is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
#

import nsNLP
import numpy as np
import torch.utils.data
from torch.autograd import Variable
from torchlanguage import datasets
import torchlanguage.transforms as transforms
import echotorch.nn as etnn
import torchlanguage.models as models
import echotorch.utils
import matplotlib.pyplot as plt
import argparse


####################################################
# Main
####################################################

# Arguments
parser = argparse.ArgumentParser(u"Feature selector visualisation")
parser.add_argument("--n-authors", type=int, default=15)
parser.add_argument("--model", type=str, required=True)
args = parser.parse_args()

# CUDA


# Feature selector
if args.model == "cgfs":
    # CNN Glove Feature Selector
    cgfs = models.cgfs(pretrained=True, n_gram=2, n_features=60)
    
    # Remove last linear layer
    cgfs.linear2 = echotorch.nn.Identity()
    
    # Transformer
    transformer = transforms.Compose([
        transforms.GloveVector(),
        transforms.ToNGram(n=2, overlapse=True),
        transforms.Reshape((-1, 1, 2, 300)),
        transforms.FeatureSelector(cgfs, 60, to_variable=True),
        transforms.Reshape((1, -1, 60)),
        transforms.Normalize(mean=-4.56512329954, std=0.911449706065)
    ])
elif args.model == "ccsaa":
    pass
else:
    raise NotImplementedError("Other model than CFGS or CCSAA is not implemented")
# end if

# Reuters C50 dataset
reutersloader = torch.utils.data.DataLoader(
    datasets.ReutersC50Dataset(download=True, n_authors=args.n_authors, transform=transformer),
    batch_size=1, shuffle=False)

# Get training data for this fold
for i, data in enumerate(reutersloader):
    # Inputs and labels
    inputs, labels, time_labels = data
    print(labels)
    plt.imshow(inputs[0, 0].t().numpy(), cmap='Greys')
    plt.show()
# end for
