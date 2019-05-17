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


####################################################
# Main
####################################################


# Parse args
args, use_cuda, param_space, xp = argument_parsing.parser_esn_training()

# Load from directory
reutersc50_dataset, reuters_loader_train, reuters_loader_test = dataset.load_dataset(args.dataset_size)

# Print authors
xp.write(u"Authors : {}".format(reutersc50_dataset.authors), log_level=0)

# First params
w = functions.manage_w(xp, args, args.keep_w)

# W index
w_index = 0

# Last space
last_space = dict()

# Iterate
for space in param_space:
    # Params
    reservoir_size, w_sparsity, leak_rate, input_scaling, \
    input_sparsity, spectral_radius, feature, aggregation, \
    state_gram, feedbacks_sparsity, lang, embedding, \
    dataset_start, window_size, ridge_param, washout = functions.get_params(space)

    # Choose the right transformer
    reutersc50_dataset.transform = features.create_transformer(feature, embedding, args.embedding_path, lang)

    # Dataset start
    reutersc50_dataset.set_start(dataset_start)

    # Set experience state
    xp.set_state(space)

    # Average sample
    average_sample = np.array([])

    # New W?
    if len(last_space) > 0 and last_space['reservoir_size'] != space['reservoir_size']:
        w = etnn.ESNCell.generate_w(int(space['reservoir_size']), space['w_sparsity'])
    # end if

    # Certainty data
    certainty_data = np.zeros((2, args.n_samples * 1500))
    certainty_index = 0

    # Set sample
    xp.set_sample_state(0)

    # ESN cell
    esn = etnn.LiESN(
        input_dim=reutersc50_dataset.transform.input_dim,
        hidden_dim=reservoir_size,
        output_dim=reutersc50_dataset.n_authors,
        spectral_radius=spectral_radius,
        sparsity=input_sparsity,
        input_scaling=input_scaling,
        w_sparsity=w_sparsity,
        learning_algo='inv',
        leaky_rate=leak_rate,
        feedbacks=args.feedbacks,
        wfdb_sparsity=feedbacks_sparsity,
        seed=1 if args.keep_w else None,
        ridge_param=ridge_param,
        washout=washout
    )
    if use_cuda:
        esn.cuda()
    # end if

    # Average
    average_k_fold = np.array([])

    # OOV
    oov = np.array([])

    # Choose fold
    xp.set_fold_state(0)
    reuters_loader_train.dataset.set_fold(0)
    reuters_loader_test.dataset.set_fold(0)

    # Choose the right transformer
    reutersc50_dataset.transform = features.create_transformer(feature, embedding, args.embedding_path, lang, 0, use_cuda, args.dataset_size, dataset_start)

    # Get training data for this fold
    for i, data in enumerate(reuters_loader_train):
        # Inputs and labels
        inputs, labels, time_labels = data

        # To variable
        inputs, time_labels = Variable(inputs), Variable(time_labels)
        if use_cuda: inputs, time_labels = inputs.cuda(), time_labels.cuda()

        # Accumulate xTx and xTy
        hidden_states = esn(inputs, time_labels)

        # Save to last file
        np.save(reutersc50_dataset.last_file[0] + u".ESNnx{}".format(reservoir_size), hidden_states.numpy())
    # end for

    # Get test data for this fold
    for i, data in enumerate(reuters_loader_test):
        # Inputs and labels
        inputs, labels, time_labels = data

        # To variable
        inputs, labels, time_labels = Variable(inputs), Variable(labels), Variable(time_labels)
        if use_cuda: inputs, labels, time_labels = inputs.cuda(), labels.cuda(), time_labels.cuda()

        # Predict
        hidden_states = esn(inputs, time_labels)

        # Save to last file
        np.save(reutersc50_dataset.last_file[0] + u".ESNnx{}".format(reservoir_size), hidden_states.numpy())
    # end for

    # W index
    w_index += 1

    # Last space
    last_space = space
# end for
