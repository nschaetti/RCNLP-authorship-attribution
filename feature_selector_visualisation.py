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

import matplotlib.pyplot as plt
import argparse
from tools import features, dataset


####################################################
# Main
####################################################

# Arguments
parser = argparse.ArgumentParser(u"Feature selector visualisation")
parser.add_argument("--n-authors", type=int, default=15)
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--sub-type", type=str, required=True)
args = parser.parse_args()

# Load from directory
reutersc50_dataset, reuters_loader_train, reuters_loader_test = dataset.load_dataset()

# Load transformer
if args.model == 'cgfs':
    reutersc50_dataset.transform = features.create_transformer(feature='cgfs', n_gram=args.sub_type, fold=0)
elif args.model == 'ccsaa':
    reutersc50_dataset.transform = features.create_transformer(feature='ccsaa', fold=0)
# end if

# Get training data for this fold
for i, data in enumerate(reuters_loader_train):
    # Inputs and labels
    inputs, labels, time_labels = data
    plt.imshow(inputs[0, 0].t().numpy(), cmap='Greys')
    plt.show()
# end for
