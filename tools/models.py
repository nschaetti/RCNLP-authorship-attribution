#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

# Imports
import torchlanguage.transforms
import echotorch.nn as etnn
from torchlanguage import models
import torch
from tools import settings
import os


# Load CGFS
def load_cgfs(fold=0, use_cuda=False):
    """
    Load CGFS
    :param fold:
    :return:
    """
    # Path
    path = os.path.join("feature_selectors", "cgfs", "c3", "cgfs.{}.p".format(fold))

    # CNN Glove Feature Selector
    cgfs = models.CGFS(n_gram=3, n_features=settings.cgfs_output_dim['c3'])
    if use_cuda:
        cgfs.cuda()
    # end if

    # Load states
    state_dict = torch.load(open(path, 'rb'))

    # Load dict
    cgfs.load_state_dict(state_dict)

    # Remove last linear layer
    cgfs.linear2 = etnn.Identity()

    # Transformer
    if use_cuda:
        transformer = torchlanguage.transforms.Compose([
            torchlanguage.transforms.GloveVector(),
            torchlanguage.transforms.ToNGram(n=3, overlapse=True),
            torchlanguage.transforms.Reshape((-1, 1, 3, settings.cgfs_input_dim)),
            torchlanguage.transforms.ToCUDA(),
            torchlanguage.transforms.FeatureSelector(cgfs, settings.cgfs_output_dim['c3'], to_variable=True),
            torchlanguage.transforms.Reshape((-1, settings.cgfs_output_dim['c3'])),
            torchlanguage.transforms.Normalize(mean=settings.cgfs_mean, std=settings.cgfs_std, input_dim=90)
        ])
    else:
        transformer = torchlanguage.transforms.Compose([
            torchlanguage.transforms.GloveVector(),
            torchlanguage.transforms.ToNGram(n=3, overlapse=True),
            torchlanguage.transforms.Reshape((-1, 1, 3, settings.cgfs_input_dim)),
            torchlanguage.transforms.FeatureSelector(cgfs, settings.cgfs_output_dim['c3'], to_variable=True),
            torchlanguage.transforms.Reshape((-1, settings.cgfs_output_dim['c3'])),
            torchlanguage.transforms.Normalize(mean=settings.cgfs_mean, std=settings.cgfs_std, input_dim=90)
        ])
    # end if
    return cgfs, transformer
# end load_cgfs


# Load CCSAA
def load_ccsaa(fold=0, use_cuda=False):
    """
    Load CNN Character Selector for Authorship Attribution
    :param fold:
    :return:
    """
    # Path
    path = os.path.join("feature_selectors", "ccsaa", "ccsaa.{}.pth".format(fold))
    voc_path = os.path.join("feature_selectors", "ccsaa", "ccsaa.{}.voc.pth".format(fold))

    # CNN Character Selector for Authorship Attribution
    ccsaa = models.CCSAA(
        text_length=settings.ccsaa_text_length,
        vocab_size=settings.ccsaa_voc_size,
        n_classes=settings.n_authors
    )
    if use_cuda:
        ccsaa.cuda()
    # end if

    # Load dict and voc
    ccsaa.load_state_dict(torch.load(open(path, 'rb')))
    voc = torch.load(open(voc_path, 'rb'))

    # Remove last linear layer
    ccsaa.linear = etnn.Identity()

    # Transformer
    if use_cuda:
        transformer = torchlanguage.transforms.Compose([
            torchlanguage.transforms.Character(),
            torchlanguage.transforms.ToIndex(token_to_ix=voc),
            torchlanguage.transforms.ToNGram(n=settings.ccsaa_text_length, overlapse=True),
            torchlanguage.transforms.Reshape((-1, settings.ccsaa_text_length)),
            torchlanguage.transforms.ToCUDA(),
            torchlanguage.transforms.FeatureSelector(ccsaa, settings.ccsaa_output_dim, to_variable=True),
            torchlanguage.transforms.Reshape((-1, settings.ccsaa_output_dim)),
            torchlanguage.transforms.Normalize(mean=settings.ccsaa_mean, std=settings.ccsaa_std, input_dim=150)
        ])
    else:
        transformer = torchlanguage.transforms.Compose([
            torchlanguage.transforms.Character(),
            torchlanguage.transforms.ToIndex(token_to_ix=voc),
            torchlanguage.transforms.ToNGram(n=settings.ccsaa_text_length, overlapse=True),
            torchlanguage.transforms.Reshape((-1, settings.ccsaa_text_length)),
            torchlanguage.transforms.FeatureSelector(ccsaa, settings.ccsaa_output_dim, to_variable=True),
            torchlanguage.transforms.Reshape((-1, settings.ccsaa_output_dim)),
            torchlanguage.transforms.Normalize(mean=settings.ccsaa_mean, std=settings.ccsaa_std, input_dim=150)
        ])
    # end if
    return ccsaa, transformer
# end load_ccsaa
