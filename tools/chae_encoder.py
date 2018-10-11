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


# Load CHAE
def load_CHAE(embedding_path, sub_dir="CHAE", use_cuda=False):
    """
    Load CHAE
    :return:
    """
    # Path
    path = os.path.join("encoders", sub_dir, "CEAE.p")
    # print(u"Load {}/{}".format(path, voc_path))

    # Character embedding
    token_to_ix, weights = torch.load(open(embedding_path, 'rb'))
    embedding_dim = weights.size(1)

    # cEnc
    encoder = torchlanguage.models.CharacterEncoder(
        text_length=settings.charencoder_text_length,
        embedding_dim=embedding_dim
    )
    if use_cuda:
        encoder.cuda()
    # end if

    # Load dict and voc
    encoder.load_state_dict(torch.load(open(path, 'rb')))

    # Eval
    encoder.eval()
    encoder.encode = True

    # Transformer
    if use_cuda:
        transformer = torchlanguage.transforms.Compose([
            torchlanguage.transforms.Character(),
            torchlanguage.transforms.ToIndex(token_to_ix=token_to_ix),
            torchlanguage.transforms.Embedding(weights=weights, voc_size=len(token_to_ix)),
            torchlanguage.transforms.ToNGram(n=settings.charencoder_text_length, overlapse=True),
            torchlanguage.transforms.Reshape((-1, embedding_dim * settings.charencoder_text_length)),
            torchlanguage.transforms.ToCUDA(),
            torchlanguage.transforms.FeatureSelector(encoder, 400, to_variable=True),
            torchlanguage.transforms.Reshape((-1, 400))
        ])
    else:
        transformer = torchlanguage.transforms.Compose([
            torchlanguage.transforms.Character(),
            torchlanguage.transforms.ToIndex(token_to_ix=token_to_ix),
            torchlanguage.transforms.Embedding(weights=weights, voc_size=len(token_to_ix)),
            torchlanguage.transforms.ToNGram(n=settings.charencoder_text_length, overlapse=True),
            torchlanguage.transforms.Reshape((-1, embedding_dim * settings.charencoder_text_length)),
            torchlanguage.transforms.FeatureSelector(encoder, 400, to_variable=True),
            torchlanguage.transforms.Reshape((-1, 400))
        ])
    # end if
    return encoder, transformer
# end load_CHAE


# Create CHAE transformer
def create_CHAE_transformer(CHAE_model, CHAE_voc, use_cuda=True):
    # Eval
    CHAE_model.eval()
    CHAE_model.encode = True

    # Transformer
    if use_cuda:
        transformer = torchlanguage.transforms.Compose([
            torchlanguage.transforms.Character(),
            torchlanguage.transforms.ToIndex(token_to_ix=CHAE_voc),
            torchlanguage.transforms.ToNGram(n=settings.cenc_text_length, overlapse=True),
            torchlanguage.transforms.Reshape((-1, settings.cenc_text_length)),
            torchlanguage.transforms.ToCUDA(),
            torchlanguage.transforms.FeatureSelector(CHAE_model, settings.cenc_output_dim, to_variable=True),
            torchlanguage.transforms.Reshape((-1, settings.cenc_output_dim))
        ])
    else:
        transformer = torchlanguage.transforms.Compose([
            torchlanguage.transforms.Character(),
            torchlanguage.transforms.ToIndex(token_to_ix=CHAE_voc),
            torchlanguage.transforms.ToNGram(n=settings.cenc_text_length, overlapse=True),
            torchlanguage.transforms.Reshape((-1, settings.cenc_text_length)),
            torchlanguage.transforms.FeatureSelector(CHAE_model, settings.cenc_output_dim, to_variable=True),
            torchlanguage.transforms.Reshape((-1, settings.cenc_output_dim))
        ])
    # end if
    return transformer
# end create_CHAE_transformer
