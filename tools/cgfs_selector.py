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

# Imports
import argparse
import os
import numpy as np
import torch.utils.data
from torch.autograd import Variable
from echotorch import datasets
from modules import CNN2DDeepFeatureSelector
from torch import optim
import torch.nn as nn
import torchlanguage.models
from tools import dataset, features, settings
import echotorch.nn as etnn


# Create CGFS transformer
def create_cgfs_transformer(cgfs_model, use_cuda=True):
    # Remove last linear layer
    cgfs_model.linear2 = etnn.Identity()

    # Transformer
    if use_cuda:
        transformer = torchlanguage.transforms.Compose([
            torchlanguage.transforms.GloveVector(),
            torchlanguage.transforms.ToNGram(n=3, overlapse=True),
            torchlanguage.transforms.Reshape((-1, 1, 3, settings.cgfs_input_dim)),
            torchlanguage.transforms.ToCUDA(),
            torchlanguage.transforms.FeatureSelector(cgfs_model, settings.cgfs_output_dim['c3'], to_variable=True),
            torchlanguage.transforms.Reshape((-1, settings.cgfs_output_dim['c3'])),
            torchlanguage.transforms.Normalize(mean=settings.cgfs_mean, std=settings.cgfs_std, input_dim=90)
        ])
    else:
        transformer = torchlanguage.transforms.Compose([
            torchlanguage.transforms.GloveVector(),
            torchlanguage.transforms.ToNGram(n=3, overlapse=True),
            torchlanguage.transforms.Reshape((-1, 1, 3, settings.cgfs_input_dim)),
            torchlanguage.transforms.FeatureSelector(cgfs_model, settings.cgfs_output_dim['c3'], to_variable=True),
            torchlanguage.transforms.Reshape((-1, settings.cgfs_output_dim['c3'])),
            torchlanguage.transforms.Normalize(mean=settings.cgfs_mean, std=settings.cgfs_std, input_dim=90)
        ])
    # end if
    return transformer
# end create_ccsaa_transformer


# Train a CGFS selector
def train_cgfs(fold=0, cgfs_epoch=100, n_gram='c3', dataset_size=100, dataset_start=0, cuda=True):
    """
    Train a CGFS selector
    :param fold:
    :param cgfs_epoch:
    :param n_gram:
    :param dataset_size:
    :param dataset_start:
    :param cuda:
    :return:
    """
    # Word embedding
    transform = torchlanguage.transforms.Compose([
        torchlanguage.transforms.GloveVector(model='en_vectors_web_lg'),
        torchlanguage.transforms.ToNGram(n=3, overlapse=True),
        torchlanguage.transforms.Reshape((-1, 3, settings.glove_embedding_dim))
    ])

    # Load from directory
    reutersc50_dataset, reuters_loader_train, reuters_loader_test = dataset.load_dataset(dataset_size=dataset_size,
                                                                                         dataset_start=dataset_start)
    reutersc50_dataset.transform = transform

    # Loss function
    loss_function = nn.NLLLoss()

    # Set fold
    reuters_loader_train.dataset.set_fold(fold)
    reuters_loader_test.dataset.set_fold(fold)

    # Model
    model = torchlanguage.models.CGFS(
        n_gram=3,
        n_authors=settings.n_authors,
        n_features=settings.cgfs_output_dim[n_gram]
    )
    if cuda:
        model.cuda()
    # end if

    # Best model
    best_acc = 0.0
    best_model = model.state_dict()

    # Optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=settings.cgfs_lr,
        momentum=settings.cgfs_momentum
    )

    # Epoch
    for epoch in range(cgfs_epoch):
        # Total losses
        training_loss = 0.0
        training_total = 0.0
        test_loss = 0.0
        test_total = 0.0

        # Get test data for this fold
        for i, data in enumerate(reuters_loader_train):
            # Inputs and labels
            inputs, labels, time_labels = data

            # View
            inputs = inputs.view((-1, 1, 3, settings.glove_embedding_dim))

            # Outputs
            outputs = torch.LongTensor(inputs.size(0)).fill_(labels[0])

            # To variable
            inputs, outputs = Variable(inputs), Variable(outputs)
            if cuda:
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
            training_total += 1.0
        # end for

        # Counters
        total = 0.0
        success = 0.0

        # For each test sample
        for i, data in enumerate(reuters_loader_test):
            # Inputs and labels
            inputs, labels, time_labels = data

            # View
            inputs = inputs.view((-1, 1, 3, settings.glove_embedding_dim))

            # Outputs
            outputs = torch.LongTensor(inputs.size(0)).fill_(labels[0])

            # To variable
            inputs, outputs = Variable(inputs), Variable(outputs)
            if cuda:
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
            test_total += 1.0
        # end for

        # Accuracy
        accuracy = success / total * 100.0

        # Print and save loss
        print(u"Epoch {}, train loss {}, test loss {}, accuracy {}".format(epoch, training_loss / training_total,
                                                                           test_loss / test_total, accuracy))

        # Save if best
        if accuracy > best_acc:
            best_acc = accuracy
            best_model = model.state_dict()
        # end if
    # end for

    # Load best
    model.load_state_dict(best_model)

    return model
# end train_cgfs
