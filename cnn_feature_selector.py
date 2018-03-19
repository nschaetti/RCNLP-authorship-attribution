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
import numpy as np
import torch.utils.data
from torch.autograd import Variable
from echotorch import datasets
from echotorch.transforms import text
import echotorch.nn as etnn
import echotorch.utils
import os


# Settings
n_epoch = 30

# Word embedding
transform = text.GloveVector(model='en_vectors_web_lg')

# Reuters C50 dataset
reutersloader = torch.utils.data.DataLoader(datasets.ReutersC50Dataset(download=True, n_authors=15,
                                                                       transform=transform),
                                            batch_size=1, shuffle=False)

# Set fold and training mode
reutersloader.dataset.set_fold(0)
reutersloader.dataset.set_train(True)

# Epoch
for epoch in range(n_epoch):
    # Get test data for this fold
    for i, data in enumerate(reutersloader):
        # Inputs and labels
        inputs, labels, time_labels = data

        # Outputs
        outputs = torch.LongTensor(inputs.size(1)).fill_(labels[0])

        # To variable
        inputs, outputs = Variable(inputs.squeeze(0)), Variable(outputs)
        # inputs, outputs = inputs.cuda(), outputs.cuda()

        print(inputs.size())
        print(outputs.size())
        exit()
    # end for
# end for
