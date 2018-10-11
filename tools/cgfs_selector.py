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
from tools import dataset, settings
from torchlanguage import models
import echotorch.nn as etnn
import echotorch.utils


# Statistics
stats_sum = 0.0
stats_sd = 0.0


# Load CGFS 35
def load_cgfs35(use_cuda=False):
    """
    Load CGFS
    :param fold:
    :return:
    """
    # Path
    path = os.path.join("feature_selectors", "cgfs", "c3", "cgfs35.p")
    # print(path)
    # CNN Glove Feature Selector
    cgfs = models.CGFS(
        n_gram=3,
        n_features=settings.cgfs_output_dim['c3'],
        n_authors=settings.n_pretrain_authors
    )
    if use_cuda:
        cgfs.cuda()
    # end if

    # Eval
    cgfs.eval()

    # Load states
    # print(u"Load {}".format(path))
    state_dict = torch.load(open(path, 'rb'))

    # Load dict
    cgfs.load_state_dict(state_dict)

    # Remove last linear layer
    cgfs.linear3 = etnn.Identity()

    # Transformer
    if use_cuda:
        transformer = torchlanguage.transforms.Compose([
            torchlanguage.transforms.GloveVector(),
            torchlanguage.transforms.ToNGram(n=3, overlapse=True),
            torchlanguage.transforms.Reshape((-1, 1, 3, settings.cgfs_input_dim)),
            torchlanguage.transforms.ToCUDA(),
            torchlanguage.transforms.FeatureSelector(cgfs, settings.cgfs_output_dim['c3'], to_variable=True),
            torchlanguage.transforms.Reshape((-1, settings.cgfs_output_dim['c3'])),
        ])
    else:
        transformer = torchlanguage.transforms.Compose([
            torchlanguage.transforms.GloveVector(),
            torchlanguage.transforms.ToNGram(n=3, overlapse=True),
            torchlanguage.transforms.Reshape((-1, 1, 3, settings.cgfs_input_dim)),
            torchlanguage.transforms.FeatureSelector(cgfs, settings.cgfs_output_dim['c3'], to_variable=True),
            torchlanguage.transforms.Reshape((-1, settings.cgfs_output_dim['c3'])),
        ])
    # end if
    return cgfs, transformer
# end load_cgfs


# Load CGFS
def load_cgfs(fold=0, dataset_size=100, dataset_start=0, use_cuda=False):
    """
    Load CGFS
    :param fold:
    :return:
    """
    # Path
    path = os.path.join("feature_selectors", "cgfs", "c3", str(int(dataset_size)), str(int(dataset_start)), "cgfs.{}.p".format(fold))
    # print(path)
    # CNN Glove Feature Selector
    cgfs = models.CGFS(n_gram=3, n_features=settings.cgfs_output_dim['c3'])
    if use_cuda:
        cgfs.cuda()
    # end if

    # Eval
    cgfs.eval()

    # Load states
    # print(u"Load {}".format(path))
    state_dict = torch.load(open(path, 'rb'))

    # Load dict
    cgfs.load_state_dict(state_dict)

    # Remove last linear layer
    cgfs.linear2 = etnn.Identity()
    cgfs.linear3 = etnn.Identity()

    # Transformer
    if use_cuda:
        transformer = torchlanguage.transforms.Compose([
            torchlanguage.transforms.GloveVector(),
            torchlanguage.transforms.ToNGram(n=3, overlapse=True),
            torchlanguage.transforms.Reshape((-1, 1, 3, settings.cgfs_input_dim)),
            torchlanguage.transforms.ToCUDA(),
            torchlanguage.transforms.FeatureSelector(cgfs, settings.cgfs_output_dim['c3'], to_variable=True),
            torchlanguage.transforms.Reshape((-1, settings.cgfs_output_dim['c3'])),
            # torchlanguage.transforms.Normalize(mean=settings.cgfs_mean, std=settings.cgfs_std, input_dim=90)
        ])
    else:
        transformer = torchlanguage.transforms.Compose([
            torchlanguage.transforms.GloveVector(),
            torchlanguage.transforms.ToNGram(n=3, overlapse=True),
            torchlanguage.transforms.Reshape((-1, 1, 3, settings.cgfs_input_dim)),
            torchlanguage.transforms.FeatureSelector(cgfs, settings.cgfs_output_dim['c3'], to_variable=True),
            torchlanguage.transforms.Reshape((-1, settings.cgfs_output_dim['c3'])),
            # torchlanguage.transforms.Normalize(mean=settings.cgfs_mean, std=settings.cgfs_std, input_dim=90)
        ])
    # end if
    return cgfs, transformer
# end load_cgfs


# Create CGFS transformer
def create_cgfs_transformer(cgfs_model, cgfs_info, use_cuda=True):
    # Remove last linear layer
    cgfs_model.linear3 = etnn.Identity()

    # Eval
    cgfs_model.eval()

    # Transformer
    if use_cuda:
        transformer = torchlanguage.transforms.Compose([
            torchlanguage.transforms.GloveVector(),
            torchlanguage.transforms.ToNGram(n=3, overlapse=True),
            torchlanguage.transforms.Reshape((-1, 1, 3, settings.cgfs_input_dim)),
            torchlanguage.transforms.ToCUDA(),
            torchlanguage.transforms.FeatureSelector(cgfs_model, settings.cgfs_output_dim['c3'], to_variable=True),
            torchlanguage.transforms.Reshape((-1, settings.cgfs_output_dim['c3'])),
            # torchlanguage.transforms.Normalize(mean=cgfs_info['mean'], std=cgfs_info['std'], input_dim=90)
            # torchlanguage.transforms.Normalize(mean=settings.cgfs_mean, std=settings.cgfs_std, input_dim=90)
        ])
    else:
        transformer = torchlanguage.transforms.Compose([
            torchlanguage.transforms.GloveVector(),
            torchlanguage.transforms.ToNGram(n=3, overlapse=True),
            torchlanguage.transforms.Reshape((-1, 1, 3, settings.cgfs_input_dim)),
            torchlanguage.transforms.FeatureSelector(cgfs_model, settings.cgfs_output_dim['c3'], to_variable=True),
            torchlanguage.transforms.Reshape((-1, settings.cgfs_output_dim['c3'])),
            # torchlanguage.transforms.Normalize(mean=cgfs_info['mean'], std=cgfs_info['std'], input_dim=90)
            # torchlanguage.transforms.Normalize(mean=settings.cgfs_mean, std=settings.cgfs_std, input_dim=90)
        ])
    # end if
    return transformer
# end create_ccsaa_transformer


# Train a CGFS selector
def train_cgfs(fold=0, cgfs_epoch=100, n_gram='c3', dataset_size=100, dataset_start=0, cuda=True, save=False, save_dir='.'):
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
    # Global
    global stats_sum, stats_sd

    # Save path
    save_path = os.path.join(save_dir, str(int(dataset_size)), str(int(dataset_start)))

    # Word embedding
    transform = torchlanguage.transforms.Compose([
        torchlanguage.transforms.GloveVector(model='en_vectors_web_lg'),
        torchlanguage.transforms.ToNGram(n=3),
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

    # FS file
    fs_file = os.path.join(save_path, u"cgfs." + str(fold) + u".p")

    # Load
    if save and os.path.exists(fs_file):
        print(fs_file)
        model.load_state_dict(
            torch.load(open(fs_file, 'rb'))
        )
        return model
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
        doc_total = 0.0
        doc_success = 0.0

        # Statistics
        stats_sum = 0.0
        stats_sd = 0.0

        # For each test sample
        for i, data in enumerate(reuters_loader_test):
            # Inputs and labels
            inputs, labels, time_labels = data

            # View
            inputs = inputs.view((-1, 1, 3, settings.glove_embedding_dim))

            # Outputs
            outputs = torch.LongTensor(inputs.size(0)).fill_(labels[0])

            # To variable
            inputs, outputs, labels = Variable(inputs), Variable(outputs), Variable(labels)
            if cuda:
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

            # Add loss
            test_loss += loss.data[0]
            test_total += 1.0

            # Normalized
            y_predicted = echotorch.utils.max_average_through_time(model_outputs, dim=0)

            # Compare
            if torch.equal(y_predicted, labels):
                doc_success += 1.0
            # end if
            doc_total += 1.0
        # end for

        # Accuracy
        accuracy = success / total * 100.0
        doc_accuracy = doc_success / doc_total * 100.0

        # Print and save loss
        print(u"Epoch {}, training total {}, train loss {}, test total {}, test loss {}, accuracy {}, doc accuracy {} (mean {}. std {})".format(
            epoch,
            training_total,
            training_loss,
            test_total,
            test_loss,
            accuracy,
            doc_accuracy,
            stats_sum / test_total,
            stats_sd / test_total)
        )

        # Save if best
        if accuracy > best_acc and epoch > 10:
            best_acc = accuracy
            best_model = model.state_dict()
            fail_count = 0
        elif epoch > 10:
            fail_count += 1
        # end if

        # Fail
        if fail_count > cgfs_epoch:
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
            open(os.path.join(save_path, u"cgfs." + str(fold) + u".pth"), 'wb')
        )

        # Save info
        torch.save(
            {'mean': stats_sum, 'std': stats_sd},
            open(os.path.join(save_path, u"cgfs.info." + str(fold) + u".pth"), 'wb')
        )
    # end if

    return model, {'mean': stats_sum, 'std': stats_sd}
# end train_cgfs
