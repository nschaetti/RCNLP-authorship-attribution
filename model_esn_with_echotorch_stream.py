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
    state_gram, feedbacks_sparsity, lang, embedding, dataset_start = functions.get_params(space)

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

    # For each sample
    for n in range(args.n_samples):
        # Set sample
        xp.set_sample_state(n)

        # ESN cell
        esn = etnn.LiESN(
            input_dim=reutersc50_dataset.transform.input_dim,
            hidden_dim=reservoir_size,
            output_dim=reutersc50_dataset.n_authors,
            spectral_radius=spectral_radius,
            sparsity=input_sparsity,
            input_scaling=input_scaling,
            w_sparsity=w_sparsity,
            w=w if args.keep_w else None,
            learning_algo='inv',
            leaky_rate=leak_rate,
            feedbacks=args.feedbacks,
            wfdb_sparsity=feedbacks_sparsity
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

            # Choose the right transformer
            reutersc50_dataset.transform = features.create_transformer(feature, embedding, args.embedding_path, lang, k, use_cuda)

            # Stream data
            stream_inputs = torch.FloatTensor()
            stream_time_labels = torch.FloatTensor()
            stream_local_labels = torch.LongTensor()

            # Get training data for this fold
            for i, data in enumerate(reuters_loader_train):
                # Inputs and labels
                inputs, labels, time_labels = data

                if i % 2 == 1:
                    # Inputs and labels
                    stream_inputs = torch.cat((stream_inputs, inputs), dim=1)
                    stream_time_labels = torch.cat((stream_time_labels, time_labels), dim=1)

                    # To variable
                    stream_inputs, stream_time_labels = Variable(stream_inputs), Variable(stream_time_labels)
                    if use_cuda: stream_inputs, stream_time_labels = stream_inputs.cuda(), stream_time_labels.cuda()

                    # Accumulate xTx and xTy
                    esn(stream_inputs, stream_time_labels)
                else:
                    stream_inputs = inputs
                    stream_time_labels = time_labels
                # end if
            # end for

            # Finalize training
            try:
                esn.finalize()
            except RuntimeError:
                continue
            # end try

            # Counters
            success = 0.0
            count = 0.0

            # Get test data for this fold
            for i, data in enumerate(reuters_loader_test):
                # Inputs and labels
                inputs, labels, time_labels = data

                # Time labels
                local_labels = torch.LongTensor(1, time_labels.size(1)).fill_(labels[0])

                if i % 2 == 1:
                    # Inputs and labels
                    stream_inputs = torch.cat((stream_inputs, inputs), dim=1)
                    stream_time_labels = torch.cat((stream_time_labels, time_labels), dim=1)
                    stream_local_labels = torch.cat((stream_local_labels, local_labels), dim=1)

                    # To variable
                    stream_inputs, stream_time_labels, stream_local_labels = Variable(stream_inputs), Variable(stream_time_labels), Variable(stream_local_labels)
                    if use_cuda: stream_inputs, stream_time_labels, stream_local_labels = stream_inputs.cuda(), stream_time_labels.cuda(), stream_local_labels.cuda()

                    # Predict
                    y_predicted = esn(stream_inputs)

                    # Normalized
                    y_predicted -= torch.min(y_predicted)
                    y_predicted /= torch.max(y_predicted) - torch.min(y_predicted)

                    # Sum to one
                    sums = torch.sum(y_predicted, dim=2)
                    for t in range(y_predicted.size(1)):
                        y_predicted[0, t, :] = y_predicted[0, t, :] / sums[0, t]
                    # end for

                    # Predictions
                    _, predicted = torch.max(y_predicted, dim=2)

                    # Compare local
                    success += float((predicted == stream_local_labels).sum())

                    # Count
                    count += float(stream_local_labels.size(1))
                else:
                    stream_inputs = inputs
                    stream_time_labels = time_labels
                    stream_local_labels = local_labels
                # end if
            # end for

            # Compute accuracy
            accuracy = success / count

            # Print success rate
            xp.add_result(accuracy)

            # Reset learning
            esn.reset()
        # end for
    # end for

    # W index
    w_index += 1

    # Last space
    last_space = space
# end for

# Save experiment results
xp.save()
