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
from torchlanguage import datasets
from torch import optim
import torch.nn as nn
import torchlanguage.models
from torchlanguage import transforms

# Settings
n_epoch = 150
n_authors = 15
use_cuda = True

# Argument parser
parser = argparse.ArgumentParser(description="CNN Character Feature Selector for AA")

# Argument
parser.add_argument("--output", type=str, help="Embedding output file", default='.')
parser.add_argument("--n-gram", type=str, help="N-gram (c1, c2)", default='c1')
parser.add_argument("--fold", type=int, help="Starting fold", default=0)
parser.add_argument("--text-length", type=int, help="Text length", default=20)
parser.add_argument("--batch-size", type=int, help="Batch-size", default=64)
parser.add_argument("--no-cuda", action='store_true', default=False, help="Enables CUDA training")
args = parser.parse_args()

# Use CUDA?
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Transforms
if args.n_gram == 'c1':
    transform = transforms.Compose([
        transforms.Character(),
        transforms.ToIndex(start_ix=0),
        transforms.ToNGram(n=args.text_length, overlapse=True),
        transforms.Reshape((-1, args.text_length))
    ])
else:
    transform = transforms.Compose([
        transforms.Character2Gram(),
        transforms.ToIndex(start_ix=0),
        transforms.ToNGram(n=args.text_length, overlapse=True),
        transforms.Reshape((-1, args.text_length))
    ])
# end if

# Dataset
dataset = datasets.ReutersC50Dataset(download=True, n_authors=15, transform=transform)

# Reuters C50 dataset
dataloader_train = torch.utils.data.DataLoader(torchlanguage.utils.CrossValidation(dataset), batch_size=1, shuffle=True)
dataloader_valid = torch.utils.data.DataLoader(torchlanguage.utils.CrossValidation(dataset, train=False), batch_size=1, shuffle=True)

# Loss function
loss_function = nn.CrossEntropyLoss()

# 10-CV
for k in np.arange(args.fold, 10):
    # Model
    model = torchlanguage.models.CCSAA(text_length=args.text_length, vocab_size=84, embedding_dim=50, n_classes=15)
    if args.cuda:
        model.cuda()
    # end if

    # Best model
    best_acc = 0.0

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Epoch
    for epoch in range(n_epoch):
        # Total losses
        training_loss = 0.0
        test_loss = 0.0

        # Get test data for this fold
        for i, data in enumerate(dataloader_train):
            # Inputs and labels
            inputs, labels, time_labels = data

            # Reshape
            inputs = inputs.view(-1, args.text_length)

            # Outputs
            outputs = torch.LongTensor(inputs.size(0)).fill_(labels[0])

            # To variable
            inputs, outputs = Variable(inputs), Variable(outputs)
            if args.cuda:
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

        # Counters
        total = 0.0
        success = 0.0

        # For each test sample
        for i, data in enumerate(dataloader_valid):
            # Inputs and labels
            inputs, labels, time_labels = data

            # Reshape
            inputs = inputs.view(-1, args.text_length)

            # Outputs
            outputs = torch.LongTensor(inputs.size(0)).fill_(labels[0])

            # To variable
            inputs, outputs = Variable(inputs), Variable(outputs)
            if args.cuda:
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

        # Accuracy
        accuracy = success / total * 100.0

        # Print and save loss
        print(
            u"Fold {}, epoch {}, training loss {}, test loss {}, accuracy {}".format(k, epoch, training_loss, test_loss,
                                                                                     accuracy))

        # Save if best
        if accuracy > best_acc:
            best_acc = accuracy
            # Save model
            print(u"Saving model with best accuracy {}".format(best_acc))
            torch.save(model.state_dict(), open(
                os.path.join(args.output, u"cnn_" + str(args.n_gram) + u"character_extractor." + str(k) + u".pth"),
                'wb'))
            torch.save(transform.transforms[1].token_to_ix, open(
                os.path.join(args.output, u"cnn_" + str(args.n_gram) + u"character_extractor." + str(k) + u".voc.pth"),
                'wb'))
        # end if
    # end for

    # Next fold
    dataloader_train.dataset.next_fold()
    dataloader_valid.dataset.next_fold()

    # Reset model
    model = None
# edn for
