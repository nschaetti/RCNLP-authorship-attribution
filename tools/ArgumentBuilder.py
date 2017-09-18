#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Auteur : Nils Schaetti <nils.schaetti@unine.ch>
# Date : 01.02.2017 17:59:05
# Lieu : Nyon, Suisse
#
# This file is part of the Reservoir Computing Memory Project.
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
#

import argparse


# Log the result of an experiment
class ArgumentBuilder(object):

    # Create parser
    @staticmethod
    def parser(desc):
        # Argument parser
        parser = argparse.ArgumentParser(description=desc)

        # Argument
        parser.add_argument("--dataset", type=str, help="Dataset's directory.")
        parser.add_argument("--author", type=str, help="Author to test.", default="1")
        parser.add_argument("--training-size", type=int, help="Number of texts to train the model.", default=2)
        parser.add_argument("--test-size", type=int, help="Number of texts to assess the model.", default=20)
        parser.add_argument("--lang", type=str, help="Language (ar, en, es, pt)", default='en')
        parser.add_argument("--converter", type=str, help="The text converter to use (fw, pos, tag, wv).",
                            default='pos')
        parser.add_argument("--pca-model", type=str, help="PCA model to load", default=None)
        parser.add_argument("--in-components", type=int, help="Number of principal component to reduce inputs to.",
                            default=-1)
        parser.add_argument("--sentence", action='store_true', help="Test sentence classification rate?", default=False)

        return parser
    # end parser

# end ArgumentBuilder
