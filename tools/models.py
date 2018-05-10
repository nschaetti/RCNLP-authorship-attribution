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
def load_cgfs(n_gram='c1', fold=0):
    """
    Load CGFS
    :param n_gram:
    :param fold:
    :return:
    """
    # CNN Glove Feature Selector
    cgfs = models.CGFS(n_gram=n_gram, n_features=settings.cgfs_output_dim[n_gram])

    # Load dict
    cgfs.load_state_dict(torch.load(open(path, 'rb')))

    # Remove last linear layer
    cgfs.linear2 = etnn.Identity()

    # Transformer
    transformer = torchlanguage.transforms.Compose([
        torchlanguage.transforms.GloveVector(),
        torchlanguage.transforms.ToNGram(n=n_gram, overlapse=True),
        torchlanguage.transforms.Reshape((-1, 1, n_gram, settings.cgfs_input_dim)),
        torchlanguage.transforms.FeatureSelector(cgfs, settings.cgfs_output_dim[n_gram], to_variable=True),
        torchlanguage.transforms.Reshape((-1, settings.cgfs_output_dim[n_gram])),
        torchlanguage.transforms.Normalize(mean=settings.cgfs_mean, std=settings.cgfs_std)
    ])
    return cgfs, transformer
# end load_cgfs


# Load CCSAA
def load_ccsaa(fold=0):
    """
    Load CNN Character Selector for Authorship Attribution
    :param fold:
    :return:
    """
    # Path
    path = os.path.join("feature_selector", "ccsaa", "cnn_c1character_extractor.{}.pth".format(fold))
    voc_path = os.path.join("feature_selector", "ccsaa", "cnn_c1character_extractor.{}.voc.pth".format(fold))

    # CNN Character Selector for Authorship Attribution
    ccsaa = models.CCSAA(
        text_length=settings.ccsaa_text_length,
        vocab_size=settings.ccsaa_voc_size,
        n_classes=settings.n_authors
    )

    # Load dict and voc
    ccsaa.load_state_dict(torch.load(open(path, 'rb')))
    voc = torch.load(open(voc_path, 'rb'))

    # Remove last linear layer
    ccsaa.linear = etnn.Identity()

    # Transformer
    transformer = torchlanguage.transforms.Compose([
        torchlanguage.transforms.Character(),
        torchlanguage.transforms.ToIndex(token_to_ix=voc),
        torchlanguage.transforms.ToNGram(n=settings.ccsaa_text_length, overlapse=True),
        torchlanguage.transforms.Reshape((-1, settings.ccsaa_text_length)),
        torchlanguage.transforms.FeatureSelector(ccsaa, settings.ccsaa_output_dim, to_variable=True),
        torchlanguage.transforms.Reshape((-1, settings.ccsaa_output_dim)),
        torchlanguage.transforms.Normalize(mean=settings.ccsaa_mean, std=settings.ccsaa_std)
    ])
    return ccsaa, transformer
# end load_ccsaa
