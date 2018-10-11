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
import os
import torch.utils.data
from torch.autograd import Variable
from torch import optim
import torch.nn as nn
import torchlanguage.models
from torchlanguage import models
from torchlanguage import transforms
from tools import dataset, settings
import echotorch.nn as etnn


# Load CCSAA 35
def load_ccsaa35(use_cuda=False):
    """
    Load CNN Character Selector for Authorship Attribution
    :param fold:
    :return:
    """
    # Path
    path = os.path.join("feature_selectors", "ccsaa", "ccsaa35.pth")
    voc_path = os.path.join("feature_selectors", "ccsaa", "ccsaa35.voc.pth")
    print(path)
    print(voc_path)
    # CNN Character Selector for Authorship Attribution
    ccsaa = models.CCSAA(
        text_length=settings.ccsaa_text_length,
        vocab_size=settings.ccsaa_pretrain_voc_size,
        n_classes=settings.n_pretrain_authors,
        n_features=settings.ccsaa_output_dim
    )
    if use_cuda:
        ccsaa.cuda()
    # end if

    # Eval
    ccsaa.eval()

    # Load dict and voc
    ccsaa.load_state_dict(torch.load(open(path, 'rb')))
    voc = torch.load(open(voc_path, 'rb'))

    # Remove last linear layer
    ccsaa.linear2 = etnn.Identity()

    # Transformer
    if use_cuda:
        transformer = torchlanguage.transforms.Compose([
            torchlanguage.transforms.Character(),
            torchlanguage.transforms.ToIndex(token_to_ix=voc),
            torchlanguage.transforms.ToNGram(n=settings.ccsaa_text_length, overlapse=True),
            torchlanguage.transforms.Reshape((-1, settings.ccsaa_text_length)),
            torchlanguage.transforms.ToCUDA(),
            torchlanguage.transforms.FeatureSelector(ccsaa, settings.ccsaa_output_dim, to_variable=True),
            torchlanguage.transforms.Reshape((-1, settings.ccsaa_output_dim))
        ])
    else:
        transformer = torchlanguage.transforms.Compose([
            torchlanguage.transforms.Character(),
            torchlanguage.transforms.ToIndex(token_to_ix=voc),
            torchlanguage.transforms.ToNGram(n=settings.ccsaa_text_length, overlapse=True),
            torchlanguage.transforms.Reshape((-1, settings.ccsaa_text_length)),
            torchlanguage.transforms.FeatureSelector(ccsaa, settings.ccsaa_output_dim, to_variable=True),
            torchlanguage.transforms.Reshape((-1, settings.ccsaa_output_dim))
        ])
    # end if
    return ccsaa, transformer
# end load_ccsaa


# Load CCSAA
def load_ccsaa(fold=0, dataset_size=100, dataset_start=0, use_cuda=False):
    """
    Load CNN Character Selector for Authorship Attribution
    :param fold:
    :return:
    """
    # Path
    path = os.path.join("feature_selectors", "ccsaa", str(int(dataset_size)), str(int(dataset_start)), "ccsaa.{}.pth".format(fold))
    voc_path = os.path.join("feature_selectors", "ccsaa", str(int(dataset_size)), str(int(dataset_start)), "ccsaa.{}.voc.pth".format(fold))
    # print(path)
    # print(voc_path)
    # CNN Character Selector for Authorship Attribution
    ccsaa = models.CCSAA(
        text_length=settings.ccsaa_text_length,
        vocab_size=settings.ccsaa_voc_size,
        n_classes=settings.n_authors,
        n_features=settings.ccsaa_output_dim
    )
    if use_cuda:
        ccsaa.cuda()
    # end if

    # Eval
    ccsaa.eval()

    # Load dict and voc
    ccsaa.load_state_dict(torch.load(open(path, 'rb')))
    voc = torch.load(open(voc_path, 'rb'))

    # Remove last linear layer
    ccsaa.linear2 = etnn.Identity()

    # Transformer
    if use_cuda:
        transformer = torchlanguage.transforms.Compose([
            torchlanguage.transforms.Character(),
            torchlanguage.transforms.ToIndex(token_to_ix=voc),
            torchlanguage.transforms.ToNGram(n=settings.ccsaa_text_length, overlapse=True),
            torchlanguage.transforms.Reshape((-1, settings.ccsaa_text_length)),
            torchlanguage.transforms.ToCUDA(),
            torchlanguage.transforms.FeatureSelector(ccsaa, settings.ccsaa_output_dim, to_variable=True),
            torchlanguage.transforms.Reshape((-1, settings.ccsaa_output_dim))
        ])
    else:
        transformer = torchlanguage.transforms.Compose([
            torchlanguage.transforms.Character(),
            torchlanguage.transforms.ToIndex(token_to_ix=voc),
            torchlanguage.transforms.ToNGram(n=settings.ccsaa_text_length, overlapse=True),
            torchlanguage.transforms.Reshape((-1, settings.ccsaa_text_length)),
            torchlanguage.transforms.FeatureSelector(ccsaa, settings.ccsaa_output_dim, to_variable=True),
            torchlanguage.transforms.Reshape((-1, settings.ccsaa_output_dim))
        ])
    # end if
    return ccsaa, transformer
# end load_ccsaa


# Create CCSAA transformer
def create_ccsaa_transformer(ccsaa_model, ccsaa_voc, use_cuda=True):
    # Remove last linear layer
    ccsaa_model.linear = etnn.Identity()

    # Transformer
    if use_cuda:
        transformer = torchlanguage.transforms.Compose([
            torchlanguage.transforms.Character(),
            torchlanguage.transforms.ToIndex(token_to_ix=ccsaa_voc),
            torchlanguage.transforms.ToNGram(n=settings.ccsaa_text_length, overlapse=True),
            torchlanguage.transforms.Reshape((-1, settings.ccsaa_text_length)),
            torchlanguage.transforms.ToCUDA(),
            torchlanguage.transforms.FeatureSelector(ccsaa_model, settings.ccsaa_output_dim, to_variable=True),
            torchlanguage.transforms.Reshape((-1, settings.ccsaa_output_dim)),
            torchlanguage.transforms.Normalize(mean=settings.ccsaa_mean, std=settings.ccsaa_std, input_dim=150)
        ])
    else:
        transformer = torchlanguage.transforms.Compose([
            torchlanguage.transforms.Character(),
            torchlanguage.transforms.ToIndex(token_to_ix=ccsaa_voc),
            torchlanguage.transforms.ToNGram(n=settings.ccsaa_text_length, overlapse=True),
            torchlanguage.transforms.Reshape((-1, settings.ccsaa_text_length)),
            torchlanguage.transforms.FeatureSelector(ccsaa_model, settings.ccsaa_output_dim, to_variable=True),
            torchlanguage.transforms.Reshape((-1, settings.ccsaa_output_dim)),
            torchlanguage.transforms.Normalize(mean=settings.ccsaa_mean, std=settings.ccsaa_std, input_dim=150)
        ])
    # end if
    return transformer
# end create_ccsaa_transformer


# Train CCSAA
def train_ccsaa(fold=0, ccsaa_epoch=100, text_length=20, n_gram='c1', dataset_size=100, dataset_start=0, cuda=True, save=False, save_dir='.'):
    """
    Train CCSAA
    :param fold:
    :param ccsaa_epoch:
    :param text_length:
    :param n_gram:
    :param dataset_size:
    :param dataset_start:
    :param cuda:
    :return:
    """
    # Save path
    save_path = os.path.join(save_dir, str(int(dataset_size)), str(int(dataset_start)))

    # Transforms
    if n_gram == 'c1':
        transform = transforms.Compose([
            transforms.Character(),
            transforms.ToIndex(start_ix=0),
            transforms.ToNGram(n=text_length, overlapse=True),
            transforms.Reshape((-1, text_length))
        ])
    else:
        transform = transforms.Compose([
            transforms.Character2Gram(),
            transforms.ToIndex(start_ix=0),
            transforms.ToNGram(n=text_length, overlapse=True),
            transforms.Reshape((-1, text_length))
        ])
    # end if

    # Load from directory
    reutersc50_dataset, reuters_loader_train, reuters_loader_test = dataset.load_dataset(dataset_size=dataset_size, dataset_start=dataset_start)
    reutersc50_dataset.transform = transform

    # Loss function
    loss_function = nn.CrossEntropyLoss()

    # Set fold
    reuters_loader_train.dataset.set_fold(fold)
    reuters_loader_test.dataset.set_fold(fold)

    # Model
    model = torchlanguage.models.CCSAA(
        text_length=text_length,
        vocab_size=settings.ccsaa_voc_size,
        embedding_dim=settings.ccsaa_embedding_dim,
        n_classes=settings.n_authors
    )
    if cuda:
        model.cuda()
    # end if

    # Load
    if save and os.path.exists(os.path.join(save_path, u"ccsaa." + str(fold) + u".pth")) and os.path.exists(os.path.join(save_path, u"ccsaa." + str(fold) + u".voc.pth")):
        model.load_state_dict(
            torch.load(open(os.path.join(save_path, u"ccsaa." + str(fold) + u".pth"), 'rb'))
        )
        voc = torch.load(
            open(os.path.join(save_path, u"ccsaa." + str(fold) + u".voc.pth"), 'rb')
        )
        return model, voc
    # end if

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=settings.ccsaa_lr, momentum=settings.ccsaa_momentum)

    # Best model
    best_acc = 0.0
    best_model = model.state_dict()

    # Fail count
    fail_count = 0

    # Epoch
    for epoch in range(10000):
        # Total losses
        training_loss = 0.0
        training_total = 0.0
        test_loss = 0.0
        test_total = 0.0

        # Get test data for this fold
        for i, data in enumerate(reuters_loader_train):
            # Inputs and labels
            inputs, labels, time_labels = data

            # Reshape
            inputs = inputs.view(-1, text_length)

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

            # Reshape
            inputs = inputs.view(-1, text_length)

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
        # print(u"Epoch {}, train loss {}, test loss {}, accuracy {}".format(epoch, training_loss / training_total, test_loss / test_total, accuracy))

        # Save if best
        if accuracy > best_acc and epoch > 10:
            best_acc = accuracy
            best_model = model.state_dict()
            fail_count = 0
        elif epoch > 10:
            fail_count += 1
        # end if

        if fail_count > ccsaa_epoch:
            break
        # end if
    # end for

    # Load best
    model.load_state_dict(best_model)

    # Save
    if save:
        # Create dir if not exists
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        # end if

        # Save
        torch.save(
            model.state_dict(),
            open(os.path.join(save_path, u"ccsaa." + str(fold) + u".pth"), 'wb')
        )

        # Save doc
        torch.save(
            transform.transforms[1].token_to_ix,
            open(os.path.join(save_path, u"ccsaa." + str(fold) + u".voc.pth"), 'wb')
        )
    # end if

    return model, transform.transforms[1].token_to_ix
# end train_ccsaa
