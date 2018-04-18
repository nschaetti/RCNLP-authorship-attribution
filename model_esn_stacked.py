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
def create_matrices(n_layers):
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


# Argument builder
args = nsNLP.tools.ArgumentBuilder(desc=u"Argument test")

# Dataset arguments
args.add_argument(command="--dataset", name="dataset", type=str, default="data/",
                  help="JSON file with the file description for each authors", required=False, extended=False)
args.add_argument(command="--k", name="k", type=int, help="K-Fold Cross Validation", extended=False, default=10)

# Author parameters
args.add_argument(command="--n-authors", name="n_authors", type=int,
                  help="Number of authors to include in the test", default=15, extended=False)
for i in range(15):
    args.add_argument(command="--author{}".format(i), name="author{}".format(i), type=str,
                      help="{}th author to test".format(i), extended=False)
# end for

# ESN arguments
args.add_argument(command="--reservoir-size", name="reservoir_size", type=float, help="Reservoir's size",
                  required=True, extended=True)
args.add_argument(command="--spectral-radius", name="spectral_radius", type=float, help="Spectral radius",
                  default="1.0", extended=True)
args.add_argument(command="--input-scaling", name="input_scaling", type=str, help="Input scaling", extended=True,
                  default="0.5")
args.add_argument(command="--input-sparsity", name="input_sparsity", type=str, help="Input sparsity", extended=True,
                  default="0.05")
args.add_argument(command="--w-sparsity", name="w_sparsity", type=str, help="W sparsity", extended=True,
                  default="0.05")
args.add_argument(command="--transformer", name="transformer", type=str,
                  help="The text transformer to use (wv, cnn)", default='wv', extended=True)
args.add_argument(command="--keep-w", name="keep_w", action='store_true', help="Keep W matrix", default=False,
                  extended=False)
args.add_argument(command="--n-layers", name="n_layers", type=int, help="Number of layers",
                  default=5, extended=True)

# Tokenizer and word vector parameters
args.add_argument(command="--lang", name="lang", type=str, help="Tokenizer language parameters",
                  default='en_vectors_web_lg', extended=True)

# Experiment output parameters
args.add_argument(command="--name", name="name", type=str, help="Experiment's name", extended=False, required=True)
args.add_argument(command="--description", name="description", type=str, help="Experiment's description",
                  extended=False, required=True)
args.add_argument(command="--output", name="output", type=str, help="Experiment's output directory", required=True,
                  extended=False)
args.add_argument(command="--n-samples", name="n_samples", type=int, help="Number of different reservoir to test",
                  default=1, extended=False)
args.add_argument(command="--verbose", name="verbose", type=int, help="Verbose level", default=2, extended=False)
args.add_argument(command="--cuda", name="cuda", action='store_true',
                  help="Use CUDA?", default=False, extended=False)

# Parse arguments
args.parse()

# CUDA
use_cuda = torch.cuda.is_available() if args.cuda else False

# Parameter space
param_space = nsNLP.tools.ParameterSpace(args.get_space())

# Experiment
xp = nsNLP.tools.ResultManager\
(
    args.output,
    args.name,
    args.description,
    args.get_space(),
    args.n_samples,
    args.k,
    verbose=args.verbose
)

# First params
rc_size = int(args.get_space()['reservoir_size'][0])
rc_w_sparsity = args.get_space()['w_sparsity'][0]

# Last space
last_space = dict()

# Create W matrices
base_w = create_matrices(int(args.get_space()['n_layers'][-1]))

# Iterate
for space in param_space:
    # Params
    reservoir_size = int(space['reservoir_size'])
    w_sparsity = space['w_sparsity']
    input_scaling = space['input_scaling']
    input_sparsity = space['input_sparsity']
    spectral_radius = space['spectral_radius']
    transformer = space['transformer'][0][0]
    aggregation = space['aggregation'][0][0]
    lang = space['lang'][0][0]
    n_layers = space['n_layers']
    leaky_rates = np.linspace(1.0, 0.01, n_layers)
    w = base_w[:n_layers]

    # Choose the right transformer
    if "wv" in transformer:
        transform = torchlanguage.transforms.GloveVector(model=lang)
    # end if

    # Load from directory
    dataset = torchlanguage.datasets.ReutersC50Dataset(
        n_authors=15,
        download=True,
        transform=transform
    )

    # Cross validation
    cross_val_dataset = {'train': torchlanguage.utils.CrossValidation(dataset, k=k),
                         'test': torchlanguage.utils.CrossValidation(dataset, k=k, train=False)}

    # Data loader
    data_loader_train = DataLoader(cross_val_dataset['train'], batch_size=1, shuffle=False, num_workers=1)
    data_loader_test = DataLoader(cross_val_dataset['test'], batch_size=1, shuffle=False, num_workers=1)

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
            input_dim=transform.input_dim,
            hidden_dim=reservoir_size,
            output_dim=dataset.n_authors,
            spectral_radius=spectral_radius,
            sparsity=input_sparsity,
            input_scaling=input_scaling,
            w=w,
            w_sparsity=w_sparsity
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
            # Set k
            xp.set_fold_state(k)

            # Get training data for this fold
            for i, data in enumerate(data_loader_train):
                # Inputs and labels
                inputs, labels, time_labels = data
                print(inputs)
                print(time_labels)
                print(inputs.size())
                print(time_labels.size())
                print(inputs[0, 0])
                print(time_labels[0, 0])
                exit()
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
            for i, data in enumerate(data_loader_test):
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

            # Next fold
            cross_val_dataset['train'].next_fold()
            cross_val_dataset['test'].next_fold()

            # Reset learning
            esn.reset()
        # end for
    # end for

    # Last space
    last_space = space
# end for

# Save experiment results
xp.save()
