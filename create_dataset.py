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

import nsNLP
import os
import argparse
import json
import pickle
import numpy as np
from shutil import copyfile
from nsNLP.esn_models.converters.PosConverter import PosConverter
from nsNLP.esn_models.converters.TagConverter import TagConverter
from nsNLP.esn_models.converters.WVConverter import WVConverter
from nsNLP.esn_models.converters.FuncWordConverter import FuncWordConverter
from tools.Logging import Logging


####################################################
# Main function
####################################################

if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description="RCNLP - Authorship attribution with Echo State Network")

    # Argument
    parser.add_argument("--input", type=str, help="Input directory")
    parser.add_argument("--output", type=str, help="Output directory")
    args = parser.parse_args()

    # Author informations
    author_infos = dict()

    # Indexes
    text_index = 0
    author_index = 0

    # For each author directory
    for author_directory in os.listdir(args.input):
        # Author directory
        author_path = os.path.join(args.input, author_directory)

        # Add author to dict
        author_infos[author_index] = list()

        # For each text
        for author_text in os.listdir(author_path):
            # Text path
            text_path = os.path.join(author_path, author_text)

            # Destination
            destination_path = os.path.join(args.output, str(text_index) + ".txt")

            # Move
            copyfile(text_path, destination_path)

            # Add to author
            author_infos[author_index].append(str(text_index) + ".txt")

            # Next index
            text_index += 1
        # end for

        # Author index
        author_index += 1
    # end for

    # Write JSON
    with open(os.path.join(args.output, "authors.json"), 'w') as f:
        json.dump(author_infos, f, encoding='utf-8', indent=4)
    # end with

# end if
