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
import numpy as np


# Build arguments for the experiments and interpret the values
class ArgumentBuilder(object):
    """
    Build arguments for the experiments and interpret the values
    """

    # Constructor
    def __init__(self, desc, set_authors=-1):
        """
        Constructor
        :param desc:
        :param set_authors:
        """
        # Argument parser
        self._parser = argparse.ArgumentParser(description=desc)
        self._set_authors = set_authors
        self._args = None
    # end __init__

    #################################################
    # Public
    #################################################

    # Parse arguments
    def parse(self):
        """
        Create a parser with all the arguments
        :return: A parser
        """
        # Dataset parameters
        self._parser.add_argument("--dataset", type=str, help="JSON file with the file description for each authors", required=True)
        self._parser.add_argument("--dataset-size", type=float, help="Ratio of the dataset to use (100 percent by default)", default=100.0)
        self._parser.add_argument("--k", type=int, help="K-Fold Cross Validation.", default=10)

        # Author parameters
        self._parser.add_argument("--n-authors", type=int, help="Number of authors to include in the test", default=15)
        if self._set_authors != 15:
            # For each possible author
            for i in range(self._set_authors):
                self._parser.add_argument("--author{}".format(i), type=str, help="{}th author to test".format(i))
            # end for
        # end if

        # ESN parameters
        self._parser.add_argument("--reservoir-size", type=str, help="Reservoir's size", required=True)
        self._parser.add_argument("--spectral-radius", type=str, help="Spectral radius", default="0.99")
        self._parser.add_argument("--leak-rate", type=str, help="Reservoir's leak rate", default="1.0")
        self._parser.add_argument("--input-scaling", type=str, help="Input scaling", default="0.5")
        self._parser.add_argument("--input-sparsity", type=str, help="Input sparsity", default="0.05")
        self._parser.add_argument("--w-sparsity", type=str, help="W sparsity", default="0.05")
        self._parser.add_argument("--converters", type=str, help="The text converters to use (fw, pos, tag, wv, oh)",
                                  default='oh')
        self._parser.add_argument("--pca-model", type=str, help="PCA model to load", default=None)
        self._parser.add_argument("--keep-w", action='store_true', help="Keep W matrix", default=False)

        # Tokenizer and word vector parameters
        self._parser.add_argument("--tokenizer", type=str, help="Which tokenizer to use (spacy, nltk, spacy-tokens)",
                                  default='nltk')
        # self._parser.add_argument("--word-embeddings", type=str, help="Which word embeddings to use (spacy, nltk)",
        #                          default='spacy')
        self._parser.add_argument("--lang", type=str, help="Tokenizer language parameters", default='en')

        # Experiment output parameters
        self._parser.add_argument("--name", type=str, help="Experiment's name", required=True)
        self._parser.add_argument("--description", type=str, help="Experiment's description", required=True)
        self._parser.add_argument("--output", type=str, help="Experiment's output directory", required=True)
        self._parser.add_argument("--sentence", action='store_true', help="Test sentence classification rate?",
                                  default=False)
        self._parser.add_argument("--samples", type=int, help="Number of different reservoir to test", default=1)
        self._parser.add_argument("--verbose", type=int, help="Verbose level", default=2)

        # Parse arguments
        self._args = self._parser.parse_args()
    # end parser

    # Keep W
    def keep_W(self):
        """
        Keep W matrix
        :return:
        """
        return self._args.keep_w
    # end keep_W

    # Verbose?
    def verbose(self):
        """
        Verbose
        :return:
        """
        return self._args.verbose
    # end verbose

    # Get output
    def get_output(self):
        """
        Get output
        :return:
        """
        return self._args.output
    # end get_output

    # Get tokenize/wv
    def get_tokenizer_params(self):
        """
        Get tokenizer/wv
        :return:
        """
        return {'tokenizer': self._args.tokenizer, 'word-embeddings': self._args.word_embeddings, 'lang': self._args.lang}
    # end get_tokenizer_params

    # Get number of authors
    def get_n_authors(self):
        """
        Get number of authors
        :return:
        """
        return self._args.n_authors
    # end get_n_authors

    # Get number of samples
    def get_n_samples(self):
        """
        Get number of samples
        :return:
        """
        return self._args.samples
    # end get_n_samples

    # Get nth author
    def get_th_author(self, i):
        """
        Get nth author
        :param i:
        :return:
        """
        return getattr(self._args, "author{}".format(i))
    # end get_th_author

    # Get dataset
    def get_dataset(self):
        """
        Get dataset
        :return:
        """
        return self._args.dataset
    # end get_dataset

    # Get dataset size
    def get_dataset_size(self):
        """
        Get dataset size
        :return:
        """
        return self._args.dataset_size
    # end get_dataset_size

    # Get fold
    def get_fold(self):
        """
        Get fold
        :return:
        """
        return self._args.k
    # end get_fold

    # Get reservoir parameters
    def get_reservoir_params(self):
        """
        Get reservoir parameters
        :return:
        """
        params_dict = dict()

        # Reservoir params
        params = ["reservoir_size", "spectral_radius", "leak_rate", "input_scaling", "input_sparsity", "w_sparsity", "converters"]

        # For each param
        for param in params:
            params_dict[param] = self._interpret_value(self.get_value(param))
        # end for

        return params_dict
    # end get_reservoir_params

    # Get input parameters
    def get_input_params(self):
        """
        Get input parameters
        :return:
        """
        values = list()

        # Converter
        converters = self._args.converters.split('+')

        # For each param
        for converter in converters:
            # one or two
            if ',' in converter:
                values.append(converter.split(','))
            else:
                values.append([converter])
            # end if
        # end for

        return values
    # end get_input_params

    # Get argument's value
    def get_value(self, param):
        """
        Get argument's value
        :param param:
        :return:
        """
        return getattr(self._args, param)
    # end get_value

    #################################################
    # Private
    #################################################

    # Interpet value
    def _interpret_value(self, value):
        """
        Interpret parameter value
        :param value:
        :return:
        """
        # Value type
        value_type = 'numeric'

        # Values array
        values = np.array([])
        values_str = list()

        # Split for addition
        additions = value.split(u'+')

        # For each addition
        for add in additions:
            try:
                if ',' in add:
                    # Split by params
                    params = add.split(',')

                    # Params
                    start = float(params[0])
                    end = float(params[1]) + 0.00001
                    step = float(params[2])

                    # Add values
                    values = np.append(values, np.arange(start, end, step))
                else:
                    # Add value
                    values = np.append(values, np.array([float(add)]))
                # end if
                value_type = 'numeric'
            except ValueError:
                values_str.append(add)
                value_type = 'str'
            # end try
        # end for

        if value_type == 'numeric':
            return np.sort(values)
        else:
            return values_str
        # end if
    # end _interpret_value

# end ArgumentBuilder
