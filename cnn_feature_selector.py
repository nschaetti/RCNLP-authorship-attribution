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
from echotorch.transforms import text
from modules import CNNDeepFeatureSelector
from torch import optim
import torch.nn as nn

# Settings
n_epoch = 600
embedding_dim = 300
n_authors = 15
use_cuda = True

# Argument parser
parser = argparse.ArgumentParser(description="CNN feature extraction")

# Argument
parser.add_argument("--output", type=str, help="Embedding output file", default='.')
parser.add_argument("--n-features", type=int, help="Number of features", default=10)
parser.add_argument("--fold", type=int, help="Starting fold", default=0)
parser.add_argument("--steps", type=int, help="Steps", default=1)
parser.add_argument("--no-cuda", action='store_true', default=False, help="Enables CUDA training")
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
# loss_function = nn.NLLLoss()
loss_function = nn.CrossEntropyLoss()

# 10-CV
for k in range(10):
    # Model
    # model = CNNFeatureSelector(embedding_dim=embedding_dim, n_authors=n_authors)
    model = CNNDeepFeatureSelector(n_authors=n_authors, n_features=args.n_features)
    if args.cuda:
        model.cuda()
    # end if

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Epoch
    for epoch in range(n_epoch):
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
            inputs, labels, time_labels = data

            # Outputs
            outputs = torch.LongTensor(inputs.size(1)).fill_(labels[0])

            # Channel
            inputs = inputs.view((-1, 1, embedding_dim))

            # To variable
            inputs, outputs = Variable(inputs), Variable(outputs)
            if args.cuda:
                inputs, outputs = inputs.cuda(), outputs.cuda()
            # end if

            # Zero grad
            if step == 0:
                model.zero_grad()
            # end if

            # Compute output
            log_probs = model(inputs)

            # Loss
            if step == 0:
                loss = loss_function(log_probs, outputs)
            else:
                loss += loss_function(log_probs, outputs)
            # end if

            # Backward and step
            if step == args.steps - 1:
                loss.backward()
                optimizer.step()

                # Add
                training_loss += loss.data[0]
            # end if

            # Step
            step += 1
            if step == args.steps:
                step = 0
            # end if
        # end for

        # Set test mode
        reutersloader.dataset.set_train(False)

        # Counters
        total = 0.0
        success = 0.0

        # For each test sample
        for i, data in enumerate(reutersloader):
            # Inputs and labels
            inputs, labels, time_labels = data

            # Outputs
            outputs = torch.LongTensor(inputs.size(1)).fill_(labels[0])

            # Channel
            inputs = inputs.view((-1, 1, embedding_dim))

            # To variable
            inputs, outputs = Variable(inputs), Variable(outputs)
            if args.cuda:
                inputs, outputs = inputs.cuda(), outputs.cuda()
            # end if

            # Forward
            model_outputs = model(inputs)
            loss = loss_function(model_outputs, outputs)

            # Take the max as predicted
            _, predicted = torch.max(model_outputs.data, 1)

            # Add to correctly classified word
            success += (predicted == outputs.data).sum()
            total += predicted.size(0)

            # Add loss
            test_loss += loss.data[0]
        # end for

        # Print and save loss
        print(u"Fold {}, epoch {}, training loss {}, test loss {}, accuracy {}".format(k, epoch, training_loss, test_loss,
                                                                             success / total * 100.0))
    # end for

    # Save model
    torch.save(model, open(os.path.join(args.output, u"cnn_feature_extractor." + str(args.n_features) + u"." + str(k) + u".p"), 'wb'))

    # Reset model
    model = None
# end for
