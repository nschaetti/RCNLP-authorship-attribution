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

import argparse
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import codecs
import numpy as np
from modules import CharacterLanguageModel
import os
import random
import string


####################################################
# Function
####################################################


####################################################
# Main function
####################################################

# Argument parser
parser = argparse.ArgumentParser(description="Create embedding data")

# Argument
parser.add_argument("--dataset", type=str, help="Input directory")
parser.add_argument("--uppercase", action='store_true', default=False, help="Input directory")
parser.add_argument("--output", type=str, help="Embedding output directory", default='embedding_data')
args = parser.parse_args()

# Open output file
output_file = codecs.open(args.output, 'ab', encoding='utf-8')

# List of subdirectories
subdirectories = list()
subdir_files = dict()
for subdir_name in os.listdir(args.dataset):
    subdirectories.append(subdir_name)
    subdir_files[subdir_name] = list()
    for file_name in os.listdir(os.path.join(args.dataset, subdir_name)):
        subdir_files[subdir_name].append(file_name)
    # end for
# end for

print(subdirectories)
print(subdir_files)
exit()

# List directory
for file_name in os.listdir(args.dataset):
    # Read the file
    text_data = codecs.open(os.path.join(args.dataset, file_name), 'rb', encoding='utf-8').read()

    # Split by lines
    lines = text_data.split(u"\n")

    # For each line
    for line in lines:
        if line != u"####################################################################################################":
            # Random filename
            random_filename = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(20))

            # Path
            file_path = os.path.join(args.output, random_filename + ".txt")

            # Open
            f = codecs.open(file_path, 'wb', encoding='utf-8')

            # Write
            f.write(line)

            # Close
            f.close()
        # end if
    # end for
# end for

output_file.close()