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


# Load WAE
def load_WAE(sub_dir="WAE", use_cuda=False):
    """
    Load WAE
    :return:
    """
    # Path
    path = os.path.join("encoders", sub_dir, "WAE.p")

    # wEnc
    encoder = torchlanguage.models.WordEncoder(
        n_gram=3,
        n_features=(675, 450)
    )
    if use_cuda:
        encoder.cuda()
    # end if

    # Load states
    print(u"Load {}".format(path))
    state_dict = torch.load(open(path, 'rb'))

    # Load dict
    encoder.load_state_dict(state_dict)

    # Eval
    encoder.eval()
    encoder.encode = True

    # Transformer
    if use_cuda:
        transformer = torchlanguage.transforms.Compose([
            torchlanguage.transforms.GloveVector(),
            torchlanguage.transforms.ToNGram(n=3, overlapse=False),
            torchlanguage.transforms.Reshape((-1, 1, 3, settings.cgfs_input_dim)),
            torchlanguage.transforms.ToCUDA(),
            torchlanguage.transforms.FeatureSelector(encoder, settings.wenc_output_dim, to_variable=True),
            torchlanguage.transforms.Reshape((-1, settings.wenc_output_dim)),
        ])
    else:
        transformer = torchlanguage.transforms.Compose([
            torchlanguage.transforms.GloveVector(),
            torchlanguage.transforms.ToNGram(n=3, overlapse=False),
            torchlanguage.transforms.Reshape((-1, 1, 3, settings.cgfs_input_dim)),
            torchlanguage.transforms.FeatureSelector(encoder, settings.wenc_output_dim, to_variable=True),
            torchlanguage.transforms.Reshape((-1, settings.wenc_output_dim)),
        ])
    # end if
    return encoder, transformer
# end load_WAE


# Create WAE transformer
def create_WAE_transformer(wae_model, use_cuda=True):
    """
    Create WAE transformer
    :param wae_model:
    :param use_cuda:
    :return:
    """
    # Eval
    wae_model.eval()
    wae_model.encode = True

    # Transformer
    if use_cuda:
        transformer = torchlanguage.transforms.Compose([
            torchlanguage.transforms.GloveVector(),
            torchlanguage.transforms.ToNGram(n=3, overlapse=True),
            torchlanguage.transforms.Reshape((-1, 1, 3, settings.cgfs_input_dim)),
            torchlanguage.transforms.ToCUDA(),
            torchlanguage.transforms.FeatureSelector(wae_model, settings.wenc_output_dim, to_variable=True),
            torchlanguage.transforms.Reshape((-1, settings.wenc_output_dim)),
        ])
    else:
        transformer = torchlanguage.transforms.Compose([
            torchlanguage.transforms.GloveVector(),
            torchlanguage.transforms.ToNGram(n=3, overlapse=True),
            torchlanguage.transforms.Reshape((-1, 1, 3, settings.cgfs_input_dim)),
            torchlanguage.transforms.FeatureSelector(wae_model, settings.wenc_output_dim, to_variable=True),
            torchlanguage.transforms.Reshape((-1, settings.wenc_output_dim)),
        ])
    # end if
    return transformer
# end create_WAE_transformer
