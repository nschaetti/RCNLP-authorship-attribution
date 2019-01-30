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
import matplotlib.pyplot as plt
import math
import sys


####################################################
# Main
####################################################

d_prob_average = 0.00435502867848
d_prob_std = 0.00580516507418

# Parse args
args, use_cuda, param_space, xp = argument_parsing.parser_esn_training()

# Load from directory
reutersc50_dataset, reuters_loader_train, reuters_loader_test = dataset.load_dataset(args.dataset_size, n_authors=3)

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
            learning_algo='inv',
            leaky_rate=leak_rate,
            feedbacks=args.feedbacks,
            wfdb_sparsity=feedbacks_sparsity,
            seed=1 if args.keep_w else None,
            ridge_param=ridge_param,
            washout=washout,
            softmax_output=True
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
            reutersc50_dataset.transform = features.create_transformer(feature, embedding, args.embedding_path, lang, k, use_cuda, args.dataset_size, dataset_start)

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
            local_success = 0.0
            local_count = 0.0

            # List of probs and d_probs
            probs_list = list()
            d_probs_list = list()

            # Get test data for this fold
            for i, data in enumerate(reuters_loader_test):
                # Inputs and labels
                inputs, labels, time_labels = data

                # Time labels
                local_labels = torch.LongTensor(1, time_labels.size(1)).fill_(labels[0])

                # To variable
                inputs, labels, time_labels, local_labels = Variable(inputs), Variable(labels), Variable(time_labels), Variable(local_labels)
                if use_cuda: inputs, labels, time_labels, local_labels = inputs.cuda(), labels.cuda(), time_labels.cuda(), local_labels.cuda()

                # Predict
                y_predicted = esn(inputs)

                # Normalized
                global_predicted = echotorch.utils.max_average_through_time(y_predicted, dim=1)

                # Compare
                if torch.equal(global_predicted, labels):
                    successes += 1.0
                    certainty_data[1, certainty_index] = 1
                else:
                    certainty_data[1, certainty_index] = 0
                # end if

                # Certainty
                y_max = float(torch.max(torch.mean(y_predicted, dim=1), dim=1)[0])
                certainty_data[0, certainty_index] = y_max
                certainty_index += 1
                if y_max > 0.6:
                    # print(u" ".join(reuters_loader_test.dataset.dataset.last_tokens))
                    # Show prob for each
                    for t, token in enumerate(reuters_loader_test.dataset.dataset.last_tokens):
                        prob = float(y_predicted[:, t, int(labels[0])])
                        if t == 0:
                            d_prob = 0.0
                        else:
                            prob_t1 = float(y_predicted[:, t - 1, int(labels[0])])
                            d_prob = abs(prob - prob_t1)
                        # end if

                        # Check
                        bold = u"normal"
                        under = u"normal"
                        if t != 0 and d_prob >= d_prob_average + 2.0 * d_prob_std:
                            if prob - prob_t1 < 0:
                                under = u"underline"
                            # end if
                            if prob - prob_t1 > 0:
                                bold = u"bold"
                            # end if
                        # end if

                        # Transparency
                        transparency = (prob - 0.5) * 4
                        if transparency < 0.0:
                            transparency = 0.0
                        # end if
                        if transparency > 1.0:
                            transparency = 1.0
                        # end if
                        transparency = transparency**2

                        # Color
                        r = int(y_predicted[:, t, 0] * 255.0)
                        g = int(y_predicted[:, t, 1] * 255.0)
                        b = int(y_predicted[:, t, 2] * 255.0)

                        # Print
                        sys.stdout.write(u'<span style="background-color: rgba(0, 255, 0, {}); font-weight: {}; text-decoration: {}">{}&nbsp;</span>'.format(transparency, bold, under, token))

                        # Add all probs
                        # probs_list.append(float(y_predicted[:, t, 0]))
                        # probs_list.append(float(y_predicted[:, t, 1]))
                        # probs_list.append(float(y_predicted[:, t, 2]))

                        # Add all d_probs
                        """if t != 0:
                            d_probs_list.append(abs(float(y_predicted[:, t, 0]) - float(y_predicted[:, t-1, 0])))
                            d_probs_list.append(abs(float(y_predicted[:, t, 1]) - float(y_predicted[:, t - 1, 1])))
                            d_probs_list.append(abs(float(y_predicted[:, t, 2]) - float(y_predicted[:, t - 1, 2])))
                        # end if"""
                    # end for

                    # Sum to one
                    """sums = torch.sum(y_predicted, dim=2)
                    for t in range(y_predicted.size(1)):
                        y_predicted[0, t, :] = y_predicted[0, t, :] / sums[0, t]
                    # end for"""
                    sys.stdout.flush()
                    print(u"\n")
                    print(int(labels[0]))
                    print(y_max)
                    print(u"##################################")
                # end if
                # Local predictions
                _, local_predicted = torch.max(y_predicted, dim=2)

                # Compare local
                local_success += float((local_predicted == local_labels).sum())

                # Count
                count += 1.0
                local_count += float(time_labels.size(1))
            # end for

            # Compute accuracy
            if args.measure == 'global':
                accuracy = successes / count
            else:
                accuracy = local_success / local_count
            # end if

            # Print success rate
            xp.add_result(accuracy)

            # Reset learning
            esn.reset()
        # end for

        # To numpy
        """probs_array = np.array(probs_list)
        d_probs_array = np.array(d_probs_list)

        # Show average and std
        print(u"probs average : {}".format(np.average(probs_array)))
        print(u"probs std : {}".format(np.std(probs_array)))
        print(u"d_probs average : {}".format(np.average(d_probs_array)))
        print(u"d_probs std : {}".format(np.std(d_probs_array)))"""
    # end for

    # Save certainty
    if args.certainty != "":
        print(certainty_data)
        np.save(open(args.certainty, "wb"), certainty_data)
    # end if

    # W index
    w_index += 1

    # Last space
    last_space = space
# end for

# Save experiment results
xp.save()
