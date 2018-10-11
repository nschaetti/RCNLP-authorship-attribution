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
import torch.nn.functional as F


####################################################
# Main
####################################################


# Parse args
args, use_cuda, param_space, xp = argument_parsing.parser_esn_training()

# Load from directory
reutersc50_dataset, reuters_loader_train, reuters_loader_test = dataset.load_dataset(args.dataset_size, shuffle=False)

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
    # Params
    reservoir_size = int(space['reservoir_size'])
    w_sparsity = space['w_sparsity']
    leak_rate = space['leak_rate']
    input_scaling = space['input_scaling']
    input_sparsity = space['input_sparsity']
    spectral_radius = space['spectral_radius']
    feature = space['transformer']
    aggregation = space['aggregation'][0][0]
    state_gram = space['state_gram']
    feedbacks_sparsity = space['feedbacks_sparsity']
    lang = space['lang'][0][0]
    embedding = space['embedding'][0][0]
    dataset_start = space['dataset_start']

    # Number of model
    n_models = len(feature)

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

        # Average
        average_k_fold = np.array([])

        # OOV
        oov = np.array([])

        # Models
        models = list()

        # Create models
        for m in range(n_models):
            # ESN cell
            esn = etnn.LiESN(
                input_dim=300 if feature[m][0] == 'wv' else 60,
                hidden_dim=reservoir_size,
                output_dim=reutersc50_dataset.n_authors,
                spectral_radius=spectral_radius,
                sparsity=input_sparsity,
                input_scaling=input_scaling,
                w_sparsity=w_sparsity,
                w=w if args.keep_w else None,
                learning_algo='inv',
                leaky_rate=float(leak_rate[m][0]),
                feedbacks=args.feedbacks,
                wfdb_sparsity=feedbacks_sparsity
            )
            if use_cuda:
                esn.cuda()
            # end if
            models.append(esn)
        # end for

        # For each batch
        for k in range(10):
            # Choose fold
            xp.set_fold_state(k)
            reuters_loader_train.dataset.set_fold(k)
            reuters_loader_test.dataset.set_fold(k)

            # Models outputs
            model_outputs = list()
            model_targets = list()
            model_local_targets = list()

            # For each model
            for m in range(n_models):
                # Leak rate and features
                model_feature = feature[m][0]
                print(u"Model {}".format(model_feature))

                # Choose the right transformer
                reutersc50_dataset.transform = features.create_transformer(model_feature, embedding, args.embedding_path, lang, k, use_cuda)

                # Outputs
                model_outputs.append(list())

                # ESN cell
                esn = models[m]

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
                try:
                    esn.finalize()
                except RuntimeError:
                    continue
                # end try

                # Model counters
                model_success = 0.0
                model_total = 0.0

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
                    y_predicted -= torch.min(y_predicted)
                    y_predicted /= torch.max(y_predicted) - torch.min(y_predicted)

                    # Sum to one
                    sums = torch.sum(y_predicted, dim=2)
                    for t in range(y_predicted.size(1)):
                        y_predicted[0, t, :] = y_predicted[0, t, :] / sums[0, t]
                    # end for

                    # Normalized
                    global_predicted = echotorch.utils.max_average_through_time(y_predicted, dim=1)

                    # Compare
                    if torch.equal(global_predicted, labels):
                        model_success += 1.0
                    # end if

                    # Count
                    model_total += 1.0

                    # Save
                    model_outputs[m].append(y_predicted[0])

                    # Save targets
                    if m == 0:
                        model_targets.append(labels)
                        model_local_targets.append(local_labels)
                    # end if
                # end for

                # Show
                print(u"Model {} : {}".format(model_feature, model_success / model_total * 100.0))

                # Reset learning
                esn.reset()
            # end for

            # Counters
            successes = 0.0
            count = 0.0
            local_success = 0.0
            local_count = 0.0

            # For each outputs
            for i in range(len(model_outputs[0])):
                # Resize output
                len1 = model_outputs[0][i].size(0)
                resize_output = model_outputs[1][i].view(1, 1, -1, 15)
                resize_output = F.upsample(resize_output, size=(len1, 15), mode='bilinear')
                resize_output = resize_output.view(-1, 15)

                # Averate output
                y_predicted = (model_outputs[0][i] + resize_output) / 2.0

                # Show
                """plt.plot(model_outputs[0][i][:, 0].data.numpy())
                plt.plot(resize_output[:, 0].data.numpy())
                plt.plot(y_predicted[:, 0].data.numpy())
                plt.show()"""

                # Normalized
                global_predicted = echotorch.utils.max_average_through_time(y_predicted, dim=0)

                # Compare
                if torch.equal(global_predicted, model_targets[i]):
                    successes += 1.0
                # end if

                # Local predictions
                _, local_predicted = torch.max(y_predicted, dim=1)

                # Compare local
                local_success += float((local_predicted == model_local_targets[i]).sum())

                # Count
                count += 1.0
                local_count += float(model_local_targets[i].size(1))
            # end for

            # Compute accuracy
            if args.measure == 'global':
                accuracy = successes / count
            else:
                accuracy = local_success / local_count
            # end if

            # Print success rate
            xp.add_result(accuracy)
        # end for
    # end for

    # W index
    w_index += 1

    # Last space
    last_space = space
# end for

# Save experiment results
xp.save()
