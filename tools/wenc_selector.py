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


# Load wenc 35
def load_wenc35(use_cuda=False):
    """
    Load wenc
    :param fold:
    :return:
    """
    # Path
    path = os.path.join("feature_selectors", "wenc", "wenc35.p")
    # print(path)
    # wEnc
    wenc = torchlanguage.models.wEnc(
        n_classes=settings.n_authors,
        n_gram=3
    )
    if use_cuda:
        wenc.cuda()
    # end if

    # Eval
    wenc.eval()

    # Load states
    # print(u"Load {}".format(path))
    state_dict = torch.load(open(path, 'rb'))

    # Load dict
    wenc.load_state_dict(state_dict)

    # Remove last linear layer
    wenc.linear2 = etnn.Identity()

    # Transformer
    if use_cuda:
        transformer = torchlanguage.transforms.Compose([
            torchlanguage.transforms.GloveVector(),
            torchlanguage.transforms.ToNGram(n=3, overlapse=True),
            torchlanguage.transforms.Reshape((-1, 1, 3, settings.cgfs_input_dim)),
            torchlanguage.transforms.ToCUDA(),
            torchlanguage.transforms.FeatureSelector(wenc, settings.wenc_output_dim, to_variable=True),
            torchlanguage.transforms.Reshape((-1, settings.wenc_output_dim)),
        ])
    else:
        transformer = torchlanguage.transforms.Compose([
            torchlanguage.transforms.GloveVector(),
            torchlanguage.transforms.ToNGram(n=3, overlapse=True),
            torchlanguage.transforms.Reshape((-1, 1, 3, settings.cgfs_input_dim)),
            torchlanguage.transforms.FeatureSelector(wenc, settings.wenc_output_dim, to_variable=True),
            torchlanguage.transforms.Reshape((-1, settings.wenc_output_dim)),
        ])
    # end if
    return wenc, transformer
# end load_wenc


# Load wEnc
def load_wenc(fold=0, dataset_size=100, dataset_start=0, use_cuda=False):
    """
    Load wEnc
    :param fold:
    :return:
    """
    # Path
    path = os.path.join("feature_selectors", "wenc", str(int(dataset_size)), str(int(dataset_start)), "wenc.{}.p".format(fold))

    # wEnc
    wenc = torchlanguage.models.wEnc(
        n_classes=settings.n_authors,
        n_gram=3
    )
    if use_cuda:
        wenc.cuda()
    # end if

    # Eval
        wenc.eval()

    # Load states
    # print(u"Load {}".format(path))
    state_dict = torch.load(open(path, 'rb'))

    # Load dict
    wenc.load_state_dict(state_dict)

    # Remove last linear layer
    wenc.linear2 = etnn.Identity()

    # Transformer
    if use_cuda:
        transformer = torchlanguage.transforms.Compose([
            torchlanguage.transforms.GloveVector(),
            torchlanguage.transforms.ToNGram(n=3, overlapse=True),
            torchlanguage.transforms.Reshape((-1, 1, 3, settings.cgfs_input_dim)),
            torchlanguage.transforms.ToCUDA(),
            torchlanguage.transforms.FeatureSelector(wenc, settings.wenc_output_dim, to_variable=True),
            torchlanguage.transforms.Reshape((-1, settings.wenc_output_dim)),
        ])
    else:
        transformer = torchlanguage.transforms.Compose([
            torchlanguage.transforms.GloveVector(),
            torchlanguage.transforms.ToNGram(n=3, overlapse=True),
            torchlanguage.transforms.Reshape((-1, 1, 3, settings.cgfs_input_dim)),
            torchlanguage.transforms.FeatureSelector(wenc, settings.wenc_output_dim, to_variable=True),
            torchlanguage.transforms.Reshape((-1, settings.wenc_output_dim)),
        ])
    # end if
    return wenc, transformer
# end load_wenc


# Create wEnc transformer
def create_wenc_transformer(wenc_model, cgfs_info, use_cuda=True):
    # Remove last linear layer
    wenc_model.linear2 = etnn.Identity()

    # Eval
    wenc_model.eval()

    # Transformer
    if use_cuda:
        transformer = torchlanguage.transforms.Compose([
            torchlanguage.transforms.GloveVector(),
            torchlanguage.transforms.ToNGram(n=3, overlapse=True),
            torchlanguage.transforms.Reshape((-1, 1, 3, settings.cgfs_input_dim)),
            torchlanguage.transforms.ToCUDA(),
            torchlanguage.transforms.FeatureSelector(wenc_model, settings.wenc_output_dim, to_variable=True),
            torchlanguage.transforms.Reshape((-1, settings.wenc_output_dim)),
        ])
    else:
        transformer = torchlanguage.transforms.Compose([
            torchlanguage.transforms.GloveVector(),
            torchlanguage.transforms.ToNGram(n=3, overlapse=True),
            torchlanguage.transforms.Reshape((-1, 1, 3, settings.cgfs_input_dim)),
            torchlanguage.transforms.FeatureSelector(wenc_model, settings.wenc_output_dim, to_variable=True),
            torchlanguage.transforms.Reshape((-1, settings.wenc_output_dim)),
        ])
    # end if
    return transformer
# end create_wenc_transformer
