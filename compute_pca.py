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

import os
import torch.utils.data
from echotorch import datasets
import torch.utils
from echotorch.transforms import text
import argparse
from sklearn.decomposition import PCA
import pickle


# Parser
parser = argparse.ArgumentParser(u"Create PCA")

# Argument
parser.add_argument(u"--dataset", type=str, help="Root directory of dataset", required=True)
parser.add_argument(u"--transformer", type=str, help="Converter type", required=True)
parser.add_argument(u"--output", type=str, help="Output file", required=True)
args = parser.parse_args()

# Reuters C50 dataset
reutersloader = torch.utils.data.DataLoader(
    datasets.ReutersC50Dataset(root=args.dataset, download=True, n_authors=50, dataset_size=25),
    batch_size=1, shuffle=True)

# Choose the right transformer
if "wv" in args.transformer:
    reutersloader.dataset.transform = text.GloveVector()
elif "pos" in args.transformer:
    reutersloader.dataset.transform = text.PartOfSpeech()
elif "tag" in args.transformer:
    reutersloader.dataset.transform = text.Tag()
elif "character" in args.transformer:
    reutersloader.dataset.transform = text.Character()
elif "fw" in args.transformer:
    reutersloader.dataset.transform = text.FunctionWord()
else:
    print(u"No transformer set!")
    exit()
# end if

# Total
data_serie = torch.FloatTensor()

# Set fold and training mode
reutersloader.dataset.set_fold(0)
reutersloader.dataset.set_train(True)

# Get training data for this fold
for i, data in enumerate(reutersloader):
    print(i)
    # Inputs and labels
    inputs, labels, time_labels = data
    if i == 0:
        data = inputs[0]
    else:
        data_serie = torch.cat((data_serie, inputs[0]), dim=0)
    # end if
# end for

# Set test mode
reutersloader.dataset.set_train(False)

# Get test data for this fold
for i, data in enumerate(reutersloader):
    print(i)
    # Inputs and labels
    inputs, labels, time_labels = data
    data_serie = torch.cat((data_serie, inputs[0]), dim=0)
# end for

for ncomponents in [40, 90, 140]:
    # PCA
    pca = PCA(n_components=ncomponents)

    # Fit
    pca.fit(data_serie.numpy())

    # Save
    pickle.dump(pca, open(os.path.join(args.output, args.transformer + str(ncomponents) + ".p"), 'wb'))
# end for
