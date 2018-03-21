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
from echotorch import datasets
from echotorch.transforms import text
from modules import CNNEmbedding, CNNDeepFeatureSelector, CNNFeatureSelector
from torch import optim
import torch.nn as nn
import echotorch.nn as etnn
import echotorch.utils
import os


# Settings
n_epoch = 1
embedding_dim = 10
n_authors = 15
use_cuda = True
voc_size = 29395
n_gram = 3

# Word embedding
transform = text.Token()

# Reuters C50 dataset
reutersloader = torch.utils.data.DataLoader(datasets.ReutersC50Dataset(download=True, n_authors=15,
                                                                       transform=transform),
                                            batch_size=1, shuffle=False)

# Token to ix
token_to_ix = dict()
ix_to_token = dict()
voc_size = 0

# Model
model = CNNEmbedding(voc_size=voc_size, embedding_dim=embedding_dim)

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Loss function
# loss_function = nn.NLLLoss()
loss_function = nn.CrossEntropyLoss()

# Set fold and training mode
reutersloader.dataset.set_fold(0)

# Epoch
for epoch in range(n_epoch):
    # Total losses
    training_loss = 0.0
    test_loss = 0.0

    # Set training mode
    reutersloader.dataset.set_train(True)

    # Get test data for this fold
    for i, data in enumerate(reutersloader):
        # Inputs and labels
        sample_inputs, sample_label = data[0], data[1]

        # Inputs
        inputs = torch.LongTensor(len(sample_inputs)-n_gram+1, n_gram)

        # For each token
        input_index = 0
        for j in np.arange(n_gram-1, len(sample_inputs)):
            gram_index = 0
            for k in np.arange(n_gram-1, -1, -1):
                token = sample_inputs[j-k]
                if token not in token_to_ix:
                    token_to_ix[token] = voc_size
                    ix_to_token[voc_size] = token
                    voc_size += 1
                # end if
                inputs[input_index, gram_index] = token_to_ix[token]
                gram_index += 1
            # end for
            input_index += 1
        # end for

        # Outputs
        outputs = torch.LongTensor(inputs.size(0)).fill_(sample_label[0])

        # To variable
        inputs, outputs = Variable(inputs), Variable(outputs)
        if use_cuda:
            inputs, outputs = inputs.cuda(), outputs.cuda()
        # end if

        # Zero grad
        model.zero_grad()

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
                token_to_ix[token] = voc_size
                ix_to_token[voc_size] = token
                voc_size += 1
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
        if use_cuda:
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
