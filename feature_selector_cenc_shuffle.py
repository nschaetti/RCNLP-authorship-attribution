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
from torch import optim
import torch.nn as nn
import torchlanguage.models
from torchlanguage import transforms
from tools import dataset, settings
import echotorch.utils
import matplotlib.pyplot as plt


last_embeddings = torch.FloatTensor()


def get_embeddings(self, inputs, outputs):
    global last_embeddings
    last_embeddings = inputs[0]
# end get_embeddings


# Argument parser
parser = argparse.ArgumentParser(description="CNN Character Feature Selector for AA (cenc)")

# Argument
parser.add_argument("--output", type=str, help="Embedding output file", default='.')
parser.add_argument("--start-fold", type=int, help="Starting fold", default=0)
parser.add_argument("--end-fold", type=int, help="Ending fold", default=9)
parser.add_argument("--text-length", type=int, help="Text length", default=20)
parser.add_argument("--batch-size", type=int, help="Batch-size", default=64)
parser.add_argument("--n-gram", type=str, help="Character n-gram", default='c1')
parser.add_argument("--no-cuda", action='store_true', default=False, help="Enables CUDA training")
parser.add_argument("--dataset-size", type=int, help="Ratio of the data set to use (100 percent by default)", default=100)
parser.add_argument("--dataset-start", type=int, help="Where to start in the data set", default=0)
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

# Load from directory
reutersc50_dataset, reuters_loader_train, reuters_loader_test = dataset.load_dataset()
reutersc50_dataset.transform = transform

# Loss function
loss_function = nn.CrossEntropyLoss()

# 10-CV
for k in np.arange(args.start_fold, args.end_fold+1):
    # Log
    print(u"Starting fold {}".format(k))

    # Set fold
    reuters_loader_train.dataset.set_fold(k)
    reuters_loader_test.dataset.set_fold(k)

    # Dataset start
    reutersc50_dataset.set_start(args.dataset_start)

    # Model
    model = torchlanguage.models.cEnc(
        text_length=args.text_length,
        vocab_size=settings.cenc_voc_size,
        embedding_dim=settings.cenc_embedding_dim,
        n_classes=settings.n_authors,
        n_features=300
    )
    if args.cuda:
        model.cuda()
    # end if

    # Optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=settings.cenc_lr,
        momentum=settings.cenc_momentum
    )

    # Forward hook
    model.linear2.register_forward_hook(get_embeddings)

    # Best model
    best_acc = 0.0

    # Epoch
    for epoch in range(settings.cenc_epoch):
        # Total losses
        training_loss = 0.0
        training_total = 0.0
        training_files = 0
        test_total = 0.0
        test_loss = 0.0
        test_files = 0

        # Samples
        samples_inputs = torch.FloatTensor()
        samples_outputs = torch.LongTensor()

        # Train
        model.train()

        # Get test data for this fold
        for i, data in enumerate(reuters_loader_train):
            # Inputs and labels
            inputs, labels, time_labels = data

            # Reshape
            inputs = inputs.view(-1, args.text_length)

            # Outputs
            outputs = torch.LongTensor(inputs.size(0)).fill_(labels[0])

            # Run training
            if i % settings.cenc_files == settings.cenc_files - 1:
                # Append
                samples_inputs = torch.cat((samples_inputs, inputs), dim=0)
                samples_outputs = torch.cat((samples_outputs, outputs), dim=0)

                # Random permutation
                random_permutation = torch.randperm(samples_inputs.size(0))

                # Shuffle inputs and outputs
                samples_inputs = samples_inputs[random_permutation]
                samples_outputs = samples_outputs[random_permutation]

                # For each batch
                for batch_i in np.arange(0, samples_inputs.size(0), settings.cenc_batch_size):
                    # Batch samples
                    batch_inputs = samples_inputs[int(batch_i):int(batch_i) + settings.cenc_batch_size]
                    batch_outputs = samples_outputs[int(batch_i):int(batch_i) + settings.cenc_batch_size]

                    # To variable
                    batch_inputs, batch_outputs = Variable(batch_inputs), Variable(batch_outputs)
                    if args.cuda:
                        batch_inputs, batch_outputs = batch_inputs.cuda(), batch_outputs.cuda()
                    # end if

                    # Zero grad
                    model.zero_grad()

                    # Compute output
                    log_probs = model(batch_inputs)

                    # Loss
                    loss = loss_function(log_probs, batch_outputs)

                    # Backward and step
                    loss.backward()
                    optimizer.step()

                    # Add
                    training_loss += loss.data[0]
                    training_total += 1.0
                # end for
            elif i % settings.cgfs_files == 0:
                # Create
                samples_inputs = inputs
                samples_outputs = outputs
            else:
                # Append
                samples_inputs = torch.cat((samples_inputs, inputs), dim=0)
                samples_outputs = torch.cat((samples_outputs, outputs), dim=0)
            # end if

            # Training files
            training_files += 1
        # end for

        # Counters
        total = 0.0
        success = 0.0
        doc_total = 0.0
        doc_success = 0.0

        # Eval
        model.eval()

        # For each test sample
        for i, data in enumerate(reuters_loader_test):
            # Inputs and labels
            inputs, labels, time_labels = data

            # Reshape
            inputs = inputs.view(-1, args.text_length)

            # Outputs
            outputs = torch.LongTensor(inputs.size(0)).fill_(labels[0])

            # To variable
            inputs, outputs, labels = Variable(inputs), Variable(outputs), Variable(labels)
            if args.cuda:
                inputs, outputs, labels = inputs.cuda(), outputs.cuda(), labels.cuda()
            # end if

            # Forward
            model_outputs = model(inputs)
            loss = loss_function(model_outputs, outputs)

            # Take the max as predicted
            _, predicted = torch.max(model_outputs.data, 1)

            # Add to correctly classified word
            success += (predicted == outputs.data).sum()
            total += predicted.size(0)

            # Normalized
            y_predicted = echotorch.utils.max_average_through_time(model_outputs, dim=0)

            # Compare
            if torch.equal(y_predicted, labels):
                doc_success += 1.0
            # end if
            doc_total += 1.0

            # Add loss
            test_loss += loss.data[0]
            test_total += 1.0

            # Test files
            test_files += 1
        # end for

        # Accuracy
        accuracy = success / total * 100.0
        doc_accuracy = doc_success / doc_total * 100.0

        # Print and save loss
        print(
        u"Fold {}, epoch {}, training files {}, training loss {}, test files {}, test loss {}, accuracy {}, doc accuracy {}".format(
            k,
            epoch,
            training_files,
            training_loss / training_total,
            test_files,
            test_loss / test_total,
            accuracy,
            doc_accuracy)
        )

        # Save if best
        if accuracy > best_acc:
            plt.imsave(os.path.join(args.output,"cenc_outputs.{}.png".format(epoch)), last_embeddings.data.t().cpu().numpy(), cmap='Greys')
            best_acc = accuracy
            # Save model
            print(u"Saving model with best accuracy {}".format(best_acc))
            torch.save(model.state_dict(), open(
                os.path.join(args.output, u"cenc." + str(k) + u".pth"),
                'wb'))
            torch.save(transform.transforms[1].token_to_ix, open(
                os.path.join(args.output, u"cenc." + str(k) + u".voc.pth"),
                'wb'))
        # end if
    # end for

    # Log best accuracy
    print(u"Fold {} with best accuracy {}".format(k, best_acc))

    # Reset model
    model = None
# edn for
