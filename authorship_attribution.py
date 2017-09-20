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
import pickle
import numpy as np
from parameters.ArgumentBuilder import ArgumentBuilder
from parameters.ParameterSpace import ParameterSpace
from corpus.Author import Author
from corpus.Corpus import Corpus
from corpus.Text import Text

####################################################
# Main function
####################################################

# Main function
if __name__ == "__main__":

    # Arguments
    args = ArgumentBuilder(desc=u"Argument test", set_authors=2)
    args.parse()

    # Corpus
    reteursC50 = Corpus(args.get_dataset())

    # Parameter space
    param_space = ParameterSpace(args.get_reservoir_params())

    # Iterate
    for space in param_space:
        print(space)
        print(u"")
    # end for
# end if
