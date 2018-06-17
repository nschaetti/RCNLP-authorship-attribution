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
import random
import string
from shutil import copyfile


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

    # For train and test
    for dset in ("C50test", "C50train"):
        # For each author directory
        for author_name in os.listdir(os.path.join(args.input, dset)):
            # Author directory
            author_path = os.path.join(args.input, dset, author_name)

            # Add author to dict
            if author_name not in author_infos.keys():
                author_infos[author_name] = list()
            # end if

            # For each text
            for author_text in os.listdir(author_path):
                # Text path
                text_path = os.path.join(author_path, author_text)

                # Random name
                random_name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))

                # Destination
                destination_path = os.path.join(args.output, random_name + ".txt")

                # Move
                copyfile(text_path, destination_path)

                # Add to author
                author_infos[author_name].append(random_name)
            # end for
        # end for
    # end for

    # Write JSON
    with open(os.path.join(args.output, "authors.json"), 'w') as f:
        json.dump(author_infos, f, encoding='utf-8', indent=4)
    # end with

# end if
