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

import argparse
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import codecs
import numpy as np
from modules import CharacterLanguageModel
import os


####################################################
# Function
####################################################


# Compute token to ix and voc size
def token_to_ix_voc_size(dataset_path):
    """
    Compute token to ix and voc size
    :param text_data:
    :return:
    """
    token_to_ix = {}
    index = 0
    # For each file
    for file_name in os.listdir(dataset_path):
        text_data = codecs.open(os.path.join(dataset_path, file_name), 'r', encoding='utf-8').read()
        for i in range(len(text_data)):
            character = text_data[i]
            if character not in token_to_ix:
                token_to_ix[character] = index
                index += 1
                print(index)
            # end if
        # end for
    # end for
    return token_to_ix, index
# end token_to_ix_voc_size

####################################################
# Main function
####################################################


# Argument parser
parser = argparse.ArgumentParser(description="Character embedding extraction")

# Argument
parser.add_argument("--dataset", type=str, help="Input file")
parser.add_argument("--dim", type=int, help="Embedding dimension")
parser.add_argument("--context-size", type=int, help="Content size")
parser.add_argument("--epoch", type=int, help="Epoch", default=300)
parser.add_argument("--output", type=str, help="Embedding output file", default='char_embedding.p')
args = parser.parse_args()

# Init random seed
torch.manual_seed(1)

# Token to ix and voc size
token_to_ix, voc_size = token_to_ix_voc_size(args.dataset)
print(voc_size)
print(token_to_ix)
exit()

# Embedding layer
embedding_layer = nn.Embedding(voc_size, args.dim)

# Grams
grams = list()

# Build tuple with (preceding chars, target char)
for i in np.arange(args.context_size, len(text)):
    context = list()
    for j in np.arange(i-args.context_size, i):
        context.append(text[j])
    # end for
    grams.append((context, text[i]))
# end for

# Losses
losses = []

# Objective function
loss_function = nn.NLLLoss()

# Our model
model = CharacterLanguageModel(voc_size, args.dim, args.context_size)

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001)

# For each epoch
for epoch in range(args.epoch):
    total_loss = torch.Tensor([0])
    for context, target in grams:
        # Prepare inputs
        context_idxs = [token_to_ix[c] for c in context]
        context_var = autograd.Variable(torch.LongTensor(context_idxs))

        # Reset gradients
        model.zero_grad()

        # Forward pass
        log_probs = model(context_var)

        # Compute loss function
        loss = loss_function(log_probs, autograd.Variable(
            torch.LongTensor([token_to_ix[target]])
        ))

        # Backward pass
        loss.backward()
        optimizer.step()

        # Add total loss
        total_loss += loss.data
    # end for
    losses.append(total_loss)
    print(total_loss[0])
# end for

print(token_to_ix['a'])
print(model.embeddings(autograd.Variable(torch.LongTensor([token_to_ix['a']]))))

print(token_to_ix['b'])
print(model.embeddings(autograd.Variable(torch.LongTensor([token_to_ix['b']]))))

# Save
torch.save(model.embeddings, open(args.output, 'wb'))
