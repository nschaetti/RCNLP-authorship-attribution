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
from torchlanguage import datasets
import torchlanguage.transforms as transforms
import echotorch.nn as etnn
import torchlanguage.models as models
import echotorch.utils
import argparse
import torchlanguage.utils

# Settings
embedding_dim = 300
n_authors = 15
n_gram = 2
use_cuda = True

# Argument parser
parser = argparse.ArgumentParser(description="CNN feature extraction")

# Argument
parser.add_argument("--fold", type=int, help="Starting fold", default=0)
parser.add_argument("--no-cuda", action='store_true', default=False, help="Enables CUDA training")
args = parser.parse_args()

# Use CUDA?
args.cuda = not args.no_cuda and torch.cuda.is_available()

# CNN Glove Feature Selector
cgfs = models.cgfs(pretrained=True, n_gram=n_gram, n_features=60)

# Remove last linear layer
cgfs.linear2 = echotorch.nn.Identity()

# Transformer
transformers = dict()
transformers['linear'] = transforms.Compose([
    transforms.GloveVector(),
    transforms.ToNGram(n=n_gram, overlapse=True),
    transforms.Reshape((-1, 600)),
])
transformers['cnn'] = transforms.Compose([
    transforms.GloveVector(),
    transforms.ToNGram(n=n_gram, overlapse=True),
    transforms.Reshape((-1, 1, 2, 300)),
    transforms.FeatureSelector(cgfs, 60, to_variable=True),
    transforms.Reshape((-1, 60)),
    transforms.Normalize(mean=-4.56512329954, std=0.911449706065)
])

# Average accuracy
average_accuracy = dict()
average_accuracy['linear'] = np.zeros(10)
average_accuracy['cnn'] = np.zeros(10)

# Average accuracy
global_average_accuracy = dict()
global_average_accuracy['linear'] = np.zeros(10)
global_average_accuracy['cnn'] = np.zeros(10)

# For each model
for model_type in ['linear', 'cnn']:
    print(u"Model type {}".format(model_type))

    # Reuters C50 dataset
    reutersloader = torch.utils.data.DataLoader(datasets.ReutersC50Dataset(download=True, n_authors=15,
                                                                           transform=transformers[model_type]),
                                                batch_size=1, shuffle=False)

    # Data set
    reuters_dataset = datasets.ReutersC50Dataset(root='./data', download=True, n_authors=n_authors,
                                                 transform=transformers[model_type])

    # Training dataset
    reutersloader_train = torch.utils.data.DataLoader(
        torchlanguage.utils.CrossValidation(reuters_dataset, k=10, train=True),
        batch_size=1, shuffle=False)

    # Eval. dataset
    reutersloader_val = torch.utils.data.DataLoader(
        torchlanguage.utils.CrossValidation(reuters_dataset, k=10, train=False),
        batch_size=1, shuffle=False)

    # 10-CV
    for k in np.arange(args.fold, 10):
        # Model
        if model_type == 'linear':
            model = etnn.RRCell(n_gram * 300, n_authors)
        else:
            model = etnn.RRCell(60, n_authors)
        # end if

        # Get test data for this fold
        step = 0
        for i, data in enumerate(reutersloader_train):
            # Inputs and labels
            inputs, labels, time_labels = data

            # View
            if model_type == 'linear':
                inputs = inputs.view(1, -1, 600)
            else:
                inputs = inputs.view(1, -1, 60)
            # end if
            time_labels = time_labels.view(1, -1, n_authors)

            # To variable
            inputs, time_labels = Variable(inputs), Variable(time_labels)

            # Give to RR
            model(inputs, time_labels)
        # end for

        # Training
        model.finalize()

        # Counters
        total = 0.0
        success = 0.0
        global_total = 0.0
        global_success = 0.0

        # For each test sample
        for i, data in enumerate(reutersloader_val):
            # Inputs and labels
            inputs, labels, time_labels = data

            # Outputs
            outputs = torch.LongTensor(1, inputs.size(1)).fill_(labels[0])

            # To variable
            inputs, time_labels = Variable(inputs), Variable(time_labels)

            # Forward
            model_outputs = model(inputs)

            # Normalized
            y_predicted = echotorch.utils.max_average_through_time(model_outputs, dim=1)

            # Compare
            if torch.equal(y_predicted.data, labels):
                global_success += 1.0
            # end if

            # Take the max as predicted
            _, predicted = torch.max(model_outputs.data, 2)

            # Add to correctly classified word
            success += (predicted == outputs).sum()
            total += predicted.size(1)
            global_total += 1.0
        # end for

        # Print and save loss
        print(u"\t\tModel {}, Fold {}, accuracy {} / {}".format(model_type, k, success / total * 100.0, global_success / global_total * 100.0))

        # Save accuracy
        average_accuracy[model_type][k] = success / total * 100.0
        global_average_accuracy[model_type][k] = global_success / global_total * 100.0

        # Reset model
        model.reset()

        # Next fold
        reutersloader_train.dataset.next_fold()
        reutersloader_val.dataset.next_fold()
    # end for

    print(u"\t10-fold cross validation for {} : {} / {}".format(model_type, np.average(average_accuracy[model_type]), np.average(global_average_accuracy[model_type])))
# end for

print(u"Difference : {} / {}".format(np.average(average_accuracy['cnn']) - np.average(average_accuracy['linear']), np.average(global_average_accuracy['cnn']) - np.average(global_average_accuracy['linear'])))
