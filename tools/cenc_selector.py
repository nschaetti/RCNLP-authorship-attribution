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


# Load cEnc 35
def load_cenc35(use_cuda=False):
    """
    Character encoder
    :param fold:
    :return:
    """
    # Path
    path = os.path.join("feature_selectors", "cenc", "cenc35.pth")
    voc_path = os.path.join("feature_selectors", "cenc", "cenc35.voc.pth")
    # print(path)
    # print(voc_path)
    # cEnc
    cenc = torchlanguage.models.cEnc(
        text_length=settings.ccsaa_text_length,
        vocab_size=settings.cenc_voc_size,
        embedding_dim=settings.cenc_embedding_dim,
        n_classes=settings.n_authors,
        n_features=300
    )
    if use_cuda:
        cenc.cuda()
    # end if

    # Eval
    cenc.eval()

    # Load dict and voc
    cenc.load_state_dict(torch.load(open(path, 'rb')))
    voc = torch.load(open(voc_path, 'rb'))

    # Remove last linear layer
    cenc.linear2 = etnn.Identity()

    # Transformer
    if use_cuda:
        transformer = torchlanguage.transforms.Compose([
            torchlanguage.transforms.Character(),
            torchlanguage.transforms.ToIndex(token_to_ix=voc),
            torchlanguage.transforms.ToNGram(n=settings.cenc_text_length, overlapse=False),
            torchlanguage.transforms.Reshape((-1, settings.cenc_text_length)),
            torchlanguage.transforms.ToCUDA(),
            torchlanguage.transforms.FeatureSelector(cenc, settings.cenc_output_dim, to_variable=True),
            torchlanguage.transforms.Reshape((-1, settings.cenc_output_dim))
        ])
    else:
        transformer = torchlanguage.transforms.Compose([
            torchlanguage.transforms.Character(),
            torchlanguage.transforms.ToIndex(token_to_ix=voc),
            torchlanguage.transforms.ToNGram(n=settings.cenc_text_length, overlapse=False),
            torchlanguage.transforms.Reshape((-1, settings.cenc_text_length)),
            torchlanguage.transforms.FeatureSelector(cenc, settings.cenc_output_dim, to_variable=True),
            torchlanguage.transforms.Reshape((-1, settings.cenc_output_dim))
        ])
    # end if
    return cenc, transformer
# end load_cenc


# Load cenc
def load_cenc(fold=0, dataset_size=100, dataset_start=0, use_cuda=False):
    """
    Load CNN Character Selector for Authorship Attribution
    :param fold:
    :return:
    """
    # Path
    path = os.path.join("feature_selectors", "cenc", str(int(dataset_size)), str(int(dataset_start)), "cenc.{}.pth".format(fold))
    voc_path = os.path.join("feature_selectors", "cenc", str(int(dataset_size)), str(int(dataset_start)), "cenc.{}.voc.pth".format(fold))
    # print(path)
    # print(voc_path)
    # cEnc
    cenc = torchlanguage.models.cEnc(
        text_length=settings.ccsaa_text_length,
        vocab_size=settings.cenc_voc_size,
        embedding_dim=settings.cenc_embedding_dim,
        n_classes=settings.n_authors,
        n_features=300
    )
    if use_cuda:
        cenc.cuda()
    # end if

    # Eval
    cenc.eval()

    # Load dict and voc
    cenc.load_state_dict(torch.load(open(path, 'rb')))
    voc = torch.load(open(voc_path, 'rb'))

    # Remove last linear layer
    cenc.linear2 = etnn.Identity()

    # Transformer
    if use_cuda:
        transformer = torchlanguage.transforms.Compose([
            torchlanguage.transforms.Character(),
            torchlanguage.transforms.ToIndex(token_to_ix=voc),
            torchlanguage.transforms.ToNGram(n=settings.cenc_text_length, overlapse=True),
            torchlanguage.transforms.Reshape((-1, settings.cenc_text_length)),
            torchlanguage.transforms.ToCUDA(),
            torchlanguage.transforms.FeatureSelector(cenc, settings.cenc_output_dim, to_variable=True),
            torchlanguage.transforms.Reshape((-1, settings.cenc_output_dim))
        ])
    else:
        transformer = torchlanguage.transforms.Compose([
            torchlanguage.transforms.Character(),
            torchlanguage.transforms.ToIndex(token_to_ix=voc),
            torchlanguage.transforms.ToNGram(n=settings.cenc_text_length, overlapse=True),
            torchlanguage.transforms.Reshape((-1, settings.cenc_text_length)),
            torchlanguage.transforms.FeatureSelector(cenc, settings.cenc_output_dim, to_variable=True),
            torchlanguage.transforms.Reshape((-1, settings.cenc_output_dim))
        ])
    # end if
    return cenc, transformer
# end load_cenc


# Create cEnc transformer
def create_cenc_transformer(cenc_model, cenc_voc, use_cuda=True):
    # Remove last linear layer
    cenc_model.linear2 = etnn.Identity()

    # Transformer
    if use_cuda:
        transformer = torchlanguage.transforms.Compose([
            torchlanguage.transforms.Character(),
            torchlanguage.transforms.ToIndex(token_to_ix=cenc_voc),
            torchlanguage.transforms.ToNGram(n=settings.cenc_text_length, overlapse=True),
            torchlanguage.transforms.Reshape((-1, settings.cenc_text_length)),
            torchlanguage.transforms.ToCUDA(),
            torchlanguage.transforms.FeatureSelector(cenc_model, settings.cenc_output_dim, to_variable=True),
            torchlanguage.transforms.Reshape((-1, settings.cenc_output_dim))
        ])
    else:
        transformer = torchlanguage.transforms.Compose([
            torchlanguage.transforms.Character(),
            torchlanguage.transforms.ToIndex(token_to_ix=cenc_voc),
            torchlanguage.transforms.ToNGram(n=settings.cenc_text_length, overlapse=True),
            torchlanguage.transforms.Reshape((-1, settings.cenc_text_length)),
            torchlanguage.transforms.FeatureSelector(cenc_model, settings.cenc_output_dim, to_variable=True),
            torchlanguage.transforms.Reshape((-1, settings.cenc_output_dim))
        ])
    # end if
    return transformer
# end create_cenc_transformer
