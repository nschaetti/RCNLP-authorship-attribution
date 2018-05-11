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
import argparse
import torchlanguage.utils
from tools import dataset, features, settings


# Create accuracy table
def create_accuracy_table(model_types):
    """
    Create accuracy table
    :param model_types:
    :return:
    """
    average_table = dict()
    for t in model_types:
        average_table[t] = np.zeros(settings.k)
    # end for
    return average_table
# end create_accuracy_table


# Model type
model_types = ['linear', 'cgfs', 'ccsaa']

# Model subtype
model_subtypes = {
    'linear': {1: 300, 2: 600, 3: 900},
    'cgfs': {'c2': 60},
    'ccsaa': {'c1': 150}
}

# Argument parser
parser = argparse.ArgumentParser(description="CNN feature extraction")
parser.add_argument("--no-cuda", action='store_true', default=False, help="Enables CUDA training")
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Average accuracy tables
average_accuracy = create_accuracy_table(model_types)
global_average_accuracy = create_accuracy_table(model_types)

# For each model
for model_type in model_types:
    for model_subtype in model_subtypes[model_type]:
        # Log model type
        print(u"Model type {}, subtype {}".format(model_type, model_subtype))

        # Load from directory
        reutersc50_dataset, reuters_loader_train, reuters_loader_test = dataset.load_dataset()

        # Input dim
        input_dim = model_subtypes[model_type][model_subtype]

        # 10-CV
        for k in np.arange(0, settings.k):
            # Load transformer
            if model_type == 'linear':
                reutersc50_dataset.transform = torchlanguage.transforms.Compose([
                    torchlanguage.transforms.GloveVector(model='en_vectors_web_lg'),
                    torchlanguage.transforms.ToNGram(n=model_subtype),
                    torchlanguage.transforms.Reshape((-1, input_dim))
                ])
            elif model_type == 'cgfs':
                reutersc50_dataset.transform = features.create_transformer(feature='cgfs', n_gram=model_subtype, fold=k)
            elif model_type == 'ccsaa':
                reutersc50_dataset.transform = features.create_transformer(feature='ccsaa', fold=k)
            # end if

            # Linear classifier
            model = etnn.RRCell(input_dim, settings.n_authors)

            # Get test data for this fold
            step = 0
            for i, data in enumerate(reuters_loader_train):
                # Inputs and labels
                inputs, labels, time_labels = data

                # View
                inputs = inputs.view(1, -1, input_dim)

                # end if
                time_labels = time_labels.view(1, -1, settings.n_authors)

                # To variable
                inputs, time_labels = Variable(inputs), Variable(time_labels)

                # Give to RR
                model(inputs, time_labels)
            # end for

            # Training
            model.finalize()

            # Counters
            total = 0.0
            success = 0.0
            global_total = 0.0
            global_success = 0.0

            # For each test sample
            for i, data in enumerate(reuters_loader_test):
                # Inputs and labels
                inputs, labels, time_labels = data

                # Outputs
                outputs = torch.LongTensor(1, inputs.size(1)).fill_(labels[0])

                # To variable
                inputs, time_labels = Variable(inputs), Variable(time_labels)

                # Forward
                model_outputs = model(inputs)

                # Normalized
                y_predicted = echotorch.utils.max_average_through_time(model_outputs, dim=1)

                # Compare
                if torch.equal(y_predicted.data, labels):
                    global_success += 1.0
                # end if

                # Take the max as predicted
                _, predicted = torch.max(model_outputs.data, 2)

                # Add to correctly classified word
                success += (predicted == outputs).sum()
                total += predicted.size(1)
                global_total += 1.0
            # end for

            # Print and save loss
            print(u"\t\tModel {}, Subtype {}, Fold {}, accuracy {} / {}".format(
                model_type,
                model_subtype,
                k,
                success / total * 100.0,
                global_success / global_total * 100.0)
            )

            # Save accuracy
            average_accuracy[model_type][k] = success / total * 100.0
            global_average_accuracy[model_type][k] = global_success / global_total * 100.0

            # Reset model
            model.reset()

            # Next fold
            reuters_loader_train.dataset.next_fold()
            reuters_loader_test.dataset.next_fold()
        # end for

        print(u"\t10-fold cross validation for {} : {} / {}".format(
            model_type,
            np.average(average_accuracy[model_type]),
            np.average(global_average_accuracy[model_type]))
        )
    # end for
# end for
