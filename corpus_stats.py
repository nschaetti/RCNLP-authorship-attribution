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

import os
import torch.utils.data
from echotorch import datasets
import torch.utils
from echotorch.transforms import text
import argparse
import torchlanguage.datasets
import spacy
import numpy as np

# Load model
nlp = spacy.load('en_vectors_web_lg')

# Load dataset
reutersc50_dataset = torchlanguage.datasets.ReutersC50Dataset()

# List of length
length_list = list()
length_list_bar = dict()

# Tokens per author
tokens_per_author = dict()

# For each sample
for sample in reutersc50_dataset:
    # Tokens
    tokens = nlp(sample[0])

    # Author
    author = sample[1]

    # Add to lengths
    length_list.append(len(tokens))

    # Centile
    centile = int(len(tokens) // 100.0)
    if centile not in length_list_bar.keys():
        length_list_bar[centile] = 1.0
    else:
        length_list_bar[centile] += 1.0

    # Add to dict
    if author not in tokens_per_author.keys():
        tokens_per_author[author] = len(tokens)
    else:
        tokens_per_author[author] += len(tokens)
    # end if
# end for

per_author_bar = dict()

author_counts = tokens_per_author.values()
author_counts.sort(reverse=True)
for i, count in enumerate(author_counts):
    print("({}, {})".format(i, count))
    mille = int(count // 2000.0)
    mille_key = mille * 2000 + 1000
    if mille_key not in per_author_bar.keys():
        per_author_bar[mille_key] = 1.0
    else:
        per_author_bar[mille_key] += 1.0
# end for

print(length_list_bar)
for key in sorted(per_author_bar.keys()):
    print("({}, {})".format(key, per_author_bar[key]))
# end for

print(np.min(np.array(length_list)))
print(np.max(np.array(length_list)))
print(np.average(np.array(length_list)))
print(np.std(np.array(length_list)))

print(np.min(np.array(tokens_per_author.values())))
print(np.max(np.array(tokens_per_author.values())))
print(np.average(np.array(tokens_per_author.values())))
print(np.std(np.array(tokens_per_author.values())))
