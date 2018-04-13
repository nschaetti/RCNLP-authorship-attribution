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

# Imports
import os
import argparse
import torch.utils.data
from torch.autograd import Variable
from echotorch import datasets
import echotorch.nn as etnn
from echotorch.transforms import text
from torch import optim
import torch.nn as nn
import numpy as np

# Settings
n_epoch = 600
embedding_dim = 300
n_authors = 15
use_cuda = True

# Argument parser
parser = argparse.ArgumentParser(description="CNN feature extraction")

# Argument
parser.add_argument("--fold", type=int, help="Starting fold", default=0)
parser.add_argument("--n-gram", type=int, help="N-gram", default=1)
parser.add_argument("--no-cuda", action='store_true', default=False, help="Enables CUDA training")
parser.add_argument("--steps", type=int, help="Steps", default=20)
args = parser.parse_args()

# Use CUDA?
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Word embedding
transform = text.GloveVector(model='en_vectors_web_lg')

# Reuters C50 dataset
reutersloader = torch.utils.data.DataLoader(datasets.ReutersC50Dataset(download=True, n_authors=15,
                                                                       transform=transform),
                                            batch_size=1, shuffle=False)

# Loss function
loss_function = nn.MSELoss()

# 10-CV
for k in np.arange(args.fold, 10):
    # Model
    model = etnn.RRCell(args.n_gram * 300, n_authors)
    if args.cuda:
        model.cuda()
    # end if

    # Total losses
    training_loss = 0.0
    test_loss = 0.0

    # Set fold and training mode
    reutersloader.dataset.set_fold(k)
    reutersloader.dataset.set_train(True)

    # Get test data for this fold
    step = 0
    for i, data in enumerate(reutersloader):
        # Inputs and labels
        sample_inputs, labels, time_labels = data

        # Remove useless dim
        time_labels = time_labels.squeeze(0)

        # Create inputs
        inputs = torch.zeros(sample_inputs.size(1) - args.n_gram + 1, 1, args.n_gram, embedding_dim)
        for i in np.arange(args.n_gram, sample_inputs.size(1) + 1):
            inputs[i - args.n_gram, 0] = sample_inputs[0, i - args.n_gram:i]
        # end for

        # Channel
        inputs = inputs.view((-1, args.n_gram*embedding_dim))

        # Batch dimension
        inputs = inputs.unsqueeze(0)
        time_labels = time_labels.unsqueeze(0)

        # To variable
        inputs, time_labels = Variable(inputs), Variable(time_labels)
        if args.cuda:
            inputs, outputs = inputs.cuda(), time_labels.cuda()
        # end if

        # Give to RR
        model(inputs, time_labels)
    # end for

    # Training
    model.finalize()

    # Set test mode
    reutersloader.dataset.set_train(False)

    # Counters
    total = 0.0
    success = 0.0

    # For each test sample
    for i, data in enumerate(reutersloader):
        # Inputs and labels
        inputs, labels, time_labels = data

        # Remove useless dim
        time_labels = time_labels.squeeze(0)

        # Outputs
        outputs = torch.LongTensor(inputs.size(1)).fill_(labels[0])

        # Channel
        inputs = inputs.view((-1, embedding_dim))

        # Batch dimension
        inputs = inputs.unsqueeze(0)
        time_labels = time_labels.unsqueeze(0)

        # To variable
        inputs, time_labels = Variable(inputs), Variable(time_labels)
        if args.cuda:
            inputs, time_labels = inputs.cuda(), outputs.cuda()
        # end if

        # Forward
        model_outputs = model(inputs)

        # Take the max as predicted
        _, predicted = torch.max(model_outputs.data, 1)

        # Add to correctly classified word
        success += (predicted == outputs).sum()
        total += predicted.size(0)
    # end for

    # Print and save loss
    print(u"Fold {}, training loss {}, test loss {}, accuracy {}".format(k, training_loss, test_loss,
                                                                         success / total * 100.0))

    # Reset model
    model = None
# end for
