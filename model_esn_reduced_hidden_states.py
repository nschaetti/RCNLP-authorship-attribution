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

import numpy as np
import torch.utils.data
from torch.autograd import Variable
import echotorch.nn as etnn
import echotorch.utils
from tools import argument_parsing, dataset, functions, features
from sklearn.decomposition import PCA, IncrementalPCA
import argparse
import os

# Parser
parser = argparse.ArgumentParser()
parser.add_argument(u"--data-dir", type=str, help=u"Where is the data ?")
parser.add_argument(u"--dim", type=int, help=u"New dimension size")
args = parser.parse_args()

# Incremental PCA
transformer = IncrementalPCA(n_components=args.dim)

# For each file in the data dir
for data_file in os.listdir(args.data_dir):
    # It an hidden state file
    if u".npy" in data_file and u"ESN" in data_file:
        # Load the file
        print(u"Learning from {}".format(data_file))
        hidden_states = np.load(os.path.join(args.data_dir, data_file))[0]

        # Incremental PCA
        if hidden_states.shape[0] > args.dim:
            transformer.fit(hidden_states)
        # end if
    # end if
# end for

# Transforming
print(u"Transforming")

# For each file in the data dir
for data_file in os.listdir(args.data_dir):
    # It an hidden state file
    if u".npy" in data_file and u"ESN" in data_file:
        # Parse file name
        file_name, txt_ext, esn_name, np_ext = data_file.split(u".")

        # Load the file
        print(u"Transforming {}".format(data_file))
        hidden_states = np.load(os.path.join(args.data_dir, data_file))[0]

        # Incremental PCA
        reduced = transformer.transform(hidden_states)

        # Save
        np.save(os.path.join(args.data_dir, u"{}.pca{}".format(file_name, args.dim)), reduced)
    # end if
# end for
