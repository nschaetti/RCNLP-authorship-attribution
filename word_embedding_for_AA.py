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
from echotorch import datasets
from echotorch.transforms import text
from modules import CNNEmbedding
from torch import optim
import torch.nn as nn
import echotorch.nn as etnn
import echotorch.utils
import os
import argparse


# Settings
n_authors = 15
next_index = 0
voc_size = 56790

# Argument parser
parser = argparse.ArgumentParser(description="Word embedding for AA")

# Argument
parser.add_argument("--output", type=str, help="Embedding output file", default='.')
parser.add_argument("--dim", type=int, help="Embedding dimension", default=300)
parser.add_argument("--no-cuda", action='store_true', default=False, help="Enables CUDA training")
parser.add_argument("--epoch", type=int, help="Epoch", default=300)
args = parser.parse_args()

# Use CUDA?
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Word embedding
transform = text.Token()

# Reuters C50 dataset
reutersloader = torch.utils.data.DataLoader(datasets.ReutersC50Dataset(download=True, n_authors=15,
                                                                       transform=transform),
                                            batch_size=1, shuffle=False)

# Token to ix
token_to_ix = dict()
ix_to_token = dict()

# Model
model = CNNEmbedding(voc_size=voc_size, embedding_dim=args.dim)
if args.cuda:
    model.cuda()
# end if

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Loss function
# loss_function = nn.NLLLoss()
loss_function = nn.CrossEntropyLoss()

# Set fold and training mode
reutersloader.dataset.set_fold(0)

# For each fold
for k in range(10):
    # Epoch
    for epoch in range(args.epoch):
        # Total losses
        training_loss = 0.0
        test_loss = 0.0

        # Set training mode
        reutersloader.dataset.set_fold(k)
        reutersloader.dataset.set_train(True)

        # Get test data for this fold
        for i, data in enumerate(reutersloader):
            # Inputs and labels
            sample_inputs, sample_label = data[0], data[1]

            # Inputs
            inputs = torch.LongTensor(len(sample_inputs), 1)

            # For each token
            j = 0
            for token in sample_inputs:
                if token not in token_to_ix:
                    token_to_ix[token] = next_index
                    ix_to_token[voc_size] = token
                    next_index += 1
                # end if
                inputs[j, 0] = token_to_ix[token]
                j += 1
            # end for

            # Outputs
            outputs = torch.LongTensor(inputs.size(0)).fill_(sample_label[0])

            # To variable
            inputs, outputs = Variable(inputs), Variable(outputs)
            if args.cuda:
                inputs, outputs = inputs.cuda(), outputs.cuda()
            # end if

            # Zero grad
            model.zero_grad()
            print(inputs)
            print(inputs[0, 0][0])
            print(ix_to_token[inputs[0, 0]])
            print(ix_to_token[inputs[1, 0]])
            print(ix_to_token[inputs[2, 0]])
            exit()
            # Compute output
            log_probs = model(inputs)

            # Loss
            loss = loss_function(log_probs, outputs)

            # Backward and step
            loss.backward()
            optimizer.step()

            # Add
            training_loss += loss.data[0]
        # end for

        # Set test mode
        reutersloader.dataset.set_train(False)

        # Counters
        total = 0.0
        success = 0.0

        # For each test sample
        for i, data in enumerate(reutersloader):
            # Inputs and labels
            sample_inputs, sample_label = data[0], data[1]

            # Inputs
            inputs = torch.LongTensor(len(sample_inputs))

            # For each token
            j = 0
            for token in sample_inputs:
                if token not in token_to_ix:
                    token_to_ix[token] = next_index
                    ix_to_token[voc_size] = token
                    next_index += 1
                # end if
                inputs[j] = token_to_ix[token]
                j += 1
            # end for

            # Outputs
            outputs = torch.LongTensor(inputs.size(0)).fill_(sample_label[0])

            # Shape
            inputs = inputs.squeeze(0)

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
        print(u"Epoch {}, training loss {}, test loss {}, accuracy {}".format(epoch, training_loss, test_loss,
                                                                              success / total * 100.0))
    # end for

    # Save model
    torch.save((token_to_ix, model), open(os.path.join(args.output, u"word_embedding_AA." + str(k) + u".p")))
    break
# end for

print(voc_size)
