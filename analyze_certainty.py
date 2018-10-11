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


# Args
parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, help="Certainty",required=True)
parser.add_argument("--step", type=float, help="Step", default=0.05)
args = parser.parse_args()

# Load
certainty_data = np.load(open(args.file, 'rb'))

for pos in np.arange(0, 1.0, args.step):
    total = 0.0
    tp_count = 0.0
    fp_count = 0.0
    pos_point = (2.0 * pos + args.step) / 2.0

    for i in range(certainty_data.shape[1]):
        if pos <= certainty_data[0, i] <= pos + args.step:
            if certainty_data[1, i] == 0:
                fp_count += 1.0
            else:
                tp_count += 1.0
            # end if
            total += 1.0
        # end if
    # end for

    if total != 0:
        print(u"({}, {})".format(pos_point, tp_count / total))
    # end if
# end for

print(u"Average : {}".format(np.average(certainty_data[0])))
print(u"Standard deviation : {}".format(np.std(certainty_data[0])))
