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
import torchlanguage.datasets
from torch.autograd import Variable
from echotorch import datasets
from echotorch.transforms import text
import echotorch.nn as etnn
import echotorch.utils
from torch.utils.data.dataloader import DataLoader
from tools import argument_parsing, dataset, functions, features

####################################################
# Functions
####################################################


# Converter in
def converter_in(converters_desc, converter):
    """
    Is the converter in the desc
    :param converters_desc:
    :param converter:
    :return:
    """
    for converter_desc in converters_desc:
        if converter in converter_desc:
            return True
        # end if
    # end for
    return False
# end converter_in


# Load character embedding
def load_character_embedding(emb_path):
    """
    Load character embedding
    :param emb_path:
    :return:
    """
    token_to_ix, weights = torch.load(open(emb_path, 'rb'))
    return token_to_ix, weights
# end load_character_embedding


# Create matrices
def create_matrices(rc_size, rc_w_sparsity, n_layers):
    """
    Create matrices
    :param n_layers:
    :return:
    """
    for i in range(n_layers):
        # Create W matrix
        if i == 0:
            base_w = etnn.ESNCell.generate_w(rc_size, rc_w_sparsity).unsqueeze(0)
        else:
            base_w = torch.cat((base_w, etnn.ESNCell.generate_w(rc_size, rc_w_sparsity).unsqueeze(0)), dim=0)
        # end if
    # end for
    return base_w
# end create_matrices

####################################################
# Main
####################################################


# Parse args
args, use_cuda, param_space, xp = argument_parsing.parser_esn_training()

# Load from directory
reutersc50_dataset, reuters_loader_train, reuters_loader_test = dataset.load_dataset()

# Print authors
xp.write(u"Authors : {}".format(reutersc50_dataset.authors), log_level=0)

# Last space
last_space = dict()

# Create W matrices
base_w = create_matrices(int(args.get_space()['reservoir_size'][-1]), float(args.get_space()['w_sparsity'][-1]), int(args.get_space()['n_layers'][-1]))

# Iterate
for space in param_space:
    # Params
    reservoir_size, w_sparsity, leak_rate, input_scaling, \
    input_sparsity, spectral_radius, feature, aggregation, \
    state_gram, feedbacks_sparsity, lang, embedding = functions.get_params(space)
    n_layers = int(space['n_layers'])
    if n_layers == 1:
        leaky_rates = leak_rate
    else:
        leaky_rates = np.linspace(1.0, leak_rate, n_layers)
    # end if
    w = base_w[:n_layers]

    # Choose the right transformer
    reutersc50_dataset.transform = features.create_transformer(feature, embedding, args.embedding_path, lang)

    # Set experience state
    xp.set_state(space)

    # Average sample
    average_sample = np.array([])

    # For each sample
    for n in range(args.n_samples):
        # Set sample
        xp.set_sample_state(n)

        # Stacked ESN
        esn = etnn.StackedESN(
            input_dim=reutersc50_dataset.transform.input_dim,
            hidden_dim=[reservoir_size]*n_layers,
            output_dim=reutersc50_dataset.n_authors,
            spectral_radius=spectral_radius,
            sparsity=input_sparsity,
            input_scaling=input_scaling,
            w=w,
            w_sparsity=w_sparsity,
            leaky_rate=leaky_rates
        )
        if use_cuda:
            esn.cuda()
        # end if

        # Average
        average_k_fold = np.array([])

        # OOV
        oov = np.array([])

        # For each batch
        for k in range(10):
            # Choose fold
            xp.set_fold_state(k)
            reuters_loader_train.dataset.set_fold(k)
            reuters_loader_test.dataset.set_fold(k)

            # Get training data for this fold
            for i, data in enumerate(reuters_loader_train):
                # Inputs and labels
                inputs, labels, time_labels = data

                # To variable
                inputs, time_labels = Variable(inputs), Variable(time_labels)
                if use_cuda: inputs, time_labels = inputs.cuda(), time_labels.cuda()

                # Accumulate xTx and xTy
                esn(inputs, time_labels)
            # end for

            # Finalize training
            esn.finalize()

            # Counters
            successes = 0.0
            count = 0.0

            # Get test data for this fold
            for i, data in enumerate(reuters_loader_test):
                # Inputs and labels
                inputs, labels, time_labels = data

                # To variable
                inputs, labels = Variable(inputs), Variable(labels)
                if use_cuda: inputs, labels = inputs.cuda(), labels.cuda()

                # Predict
                y_predicted = esn(inputs)

                # Normalized
                y_predicted -= torch.min(y_predicted)
                y_predicted /= torch.max(y_predicted) - torch.min(y_predicted)

                # Sum to one
                sums = torch.sum(y_predicted, dim=2)
                for t in range(y_predicted.size(1)):
                    y_predicted[0, t, :] = y_predicted[0, t, :] / sums[0, t]
                # end for

                # Normalized
                y_predicted = echotorch.utils.max_average_through_time(y_predicted, dim=1)

                # Compare
                if torch.equal(y_predicted, labels):
                    successes += 1.0
                # end if

                count += 1.0
            # end for

            # Print success rate
            xp.add_result(successes / count)

            # Reset learning
            esn.reset()
        # end for
    # end for

    # Last space
    last_space = space
# end for

# Save experiment results
xp.save()
