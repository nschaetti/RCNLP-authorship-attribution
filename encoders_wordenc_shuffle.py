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
import echotorch.utils
import matplotlib.pyplot as plt


last_input = torch.FloatTensor()
last_code = torch.FloatTensor()


def get_inputs(self, inputs, outputs):
    global last_input
    last_input = inputs[0]
# end get_embeddings


def get_code(self, inputs, outputs):
    global last_code
    last_code = outputs
# end get_embeddings


# Argument parser
parser = argparse.ArgumentParser(description="CNN feature extraction")

# Argument
parser.add_argument("--output", type=str, help="Embedding output directory", default='.')
parser.add_argument("--input", type=str, help="Pre trained model", default='')
parser.add_argument("--n-gram", type=int, help="Word n-gram", default=3)
parser.add_argument("--hidden", type=int, help="Hidden size", default=600)
parser.add_argument("--size", type=int, help="Encoding size", default=300)
parser.add_argument("--no-cuda", action='store_true', default=False, help="Enables CUDA training")
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Word embedding
transform = torchlanguage.transforms.Compose([
    torchlanguage.transforms.GloveVector(model='en_vectors_web_lg'),
    torchlanguage.transforms.ToNGram(n=args.n_gram),
    torchlanguage.transforms.Reshape((-1, args.n_gram, settings.glove_embedding_dim))
])

# Load from directory
reutersc50_dataset, reuters_loader_train, reuters_loader_test = dataset.load_pretrain_dataset()
reutersc50_dataset.transform = transform

print(reutersc50_dataset.authors)

# Loss function
loss_function = nn.MSELoss()

if args.input != "":
    # wEnc
    model = torchlanguage.models.WordEncoder(
        n_gram=3,
        n_features=(args.hidden, args.size)
    )

    # Load states
    print(u"Load {}".format(args.input))
    state_dict = torch.load(open(args.input, 'rb'))

    # Load dict
    model.load_state_dict(state_dict)
else:
    # Model
    model = torchlanguage.models.WordEncoder(
        n_gram=args.n_gram,
        n_features=(args.hidden, args.size)
    )
# end if
if args.cuda:
    model.cuda()
# end if

# Forward hook
model.encoder.register_forward_hook(get_inputs)
model.encoder.register_forward_hook(get_code)

# Best model
best_acc = 1000000.0

# Optimizer
optimizer = optim.SGD(
    model.parameters(),
    lr=settings.wordencoder_lr,
    momentum=settings.wordencoder_momentum
)

# Epoch
for epoch in range(settings.wordencoder_epoch):
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
        # print(i)
        # Inputs and labels
        inputs, labels, time_labels = data

        # View
        inputs = inputs.view((-1, args.n_gram * settings.glove_embedding_dim))

        # To variable
        inputs = Variable(inputs)
        if args.cuda:
            inputs = inputs.cuda()
        # end if

        # Zero grad
        model.zero_grad()

        # Compute output
        model_outputs = model(inputs)

        # Loss
        loss = loss_function(model_outputs, inputs)

        # Backward and step
        loss.backward()
        optimizer.step()

        # Add
        training_loss += loss.data[0]
        training_total += 1.0
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

        # View
        inputs = inputs.view((-1, args.n_gram * settings.glove_embedding_dim))

        # To variable
        inputs = Variable(inputs)
        if args.cuda:
            inputs = inputs.cuda()
        # end if

        # Forward
        model_outputs = model(inputs)
        loss = loss_function(model_outputs, inputs)

        # Add loss
        test_loss += loss.data[0]
        test_total += 1.0
        test_files += 1
    # end for

    # Print and save loss
    print(u"Epoch {}, training files {}, training loss {}, test files {}, test loss {}".format(
        epoch,
        training_files,
        training_loss / training_total,
        test_files,
        test_loss / test_total)
    )

    # Save if best
    if test_loss / test_total < best_acc:
        plt.imsave(os.path.join(args.output, "WAE.{}.png".format(epoch)),
                   torch.cat((last_input, model_outputs), dim=0).data.t().cpu().numpy(), cmap='Greys')
        plt.show()
        plt.imsave(os.path.join(args.output, "WAE_code.{}.png".format(epoch)),
                   last_code.data.t().cpu().numpy(), cmap='Greys')
        plt.show()
        best_acc = test_loss / test_total
        # Save model
        print(u"Saving model with best loss {}".format(best_acc))
        torch.save(model.state_dict(), open(
            os.path.join(args.output, u"WAE.p"),
            'wb'))
    # end if
# end for
