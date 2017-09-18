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

import io
import os
import argparse
import pickle
import numpy as np
import spacy
from nsNLP.esn_models.converters.PosConverter import PosConverter
from nsNLP.esn_models.converters.TagConverter import TagConverter
from nsNLP.esn_models.converters.WVConverter import WVConverter
from nsNLP.esn_models.converters.FuncWordConverter import FuncWordConverter
from tools.Logging import Logging
from core.tools.Metrics import Metrics

#########################################################################
# Experience settings
#########################################################################

# Exp. info
ex_name = "Authorship Attribution"
ex_instance = "Two Authors Two Sided Inputs"

# Reservoir Properties
rc_leak_rate = 0.1  # Leak rate
rc_input_scaling = 0.25  # Input scaling
rc_size = 2000  # Reservoir size
rc_spectral_radius = 0.99  # Spectral radius
rc_w_sparsity = 0.5
rc_input_sparsity = 0.1

####################################################
# Functions
####################################################


def get_n_token(text_file):
    t_nlp = spacy.load(args.lang)
    doc = t_nlp(io.open(text_file, 'r').read())
    count = 0
    # For each token
    for index, word in enumerate(doc):
        count += 1
    # end for
    return count
# end get_n_token

####################################################
# Main function
####################################################

if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description="RCNLP - Authorship clustering with Echo State Network")

    # Argument
    parser.add_argument("--dataset", type=str, help="Dataset's directory.")
    parser.add_argument("--author1", type=str, help="Author 1' ID.")
    parser.add_argument("--author2", type=str, help="Author 2's ID.")
    parser.add_argument("--lang", type=str, help="Language (ar, en, es, pt)", default='en')
    parser.add_argument("--converter", type=str, help="The text converter to use (fw, pos, tag, wv).", default='pos')
    parser.add_argument("--pca-model", type=str, help="PCA model to load", default=None)
    parser.add_argument("--in-components", type=int, help="Number of principal component to reduce inputs to.",
                        default=-1)
    parser.add_argument("--K", type=int, help="n-Fold Cross Validation", default=10)
    parser.add_argument("--k", type=int, help="Fold position to use", default=0)
    args = parser.parse_args()

    # Logging
    logging = RCNLPLogging(exp_name=ex_name, exp_inst=ex_instance,
                           exp_value=RCNLPLogging.generate_experience_name(locals()))
    logging.save_globals()
    logging.save_variables(locals())

    # PCA model
    pca_model = None
    if args.pca_model is not None:
        pca_model = pickle.load(open(args.pca_model, 'r'))
    # end if

    # Base converter
    base_converter = ReverseConverter()

    # Reverse WV converter
    reverse_wv_converter = WVConverter(pca_model=pca_model, upper_level=base_converter)

    # WV converter
    wv_converter = WVConverter(pca_model=pca_model)

    # Prepare training and test set indexes.
    n_fold_samples = int(100 / args.K)
    indexes = np.arange(0, 100, 1)
    indexes.shape = (args.K, n_fold_samples)

    # Prepare training and test set.
    test_set_indexes = indexes[args.k]
    training_set_indexes = indexes
    training_set_indexes = np.delete(training_set_indexes, args.k, axis=0)
    training_set_indexes.shape = (100 - n_fold_samples)

    # Classifier
    classifier = EchoWordClassifier(classes=[0, 1], size=rc_size, input_scaling=rc_input_scaling,
                                    leak_rate=rc_leak_rate,
                                    input_sparsity=rc_input_sparsity, converter=wv_converter,
                                    spectral_radius=rc_spectral_radius, w_sparsity=rc_w_sparsity)

    # Add examples
    for author_index, author_id in enumerate((args.author1, args.author2)):
        author_path = os.path.join(args.dataset, "total", author_id)
        for file_index in training_set_indexes:
            file_path = os.path.join(author_path, str(file_index) + ".txt")
            classifier.train(io.open(file_path, 'r').read(), author_index)
            # end for
    # end for

    # Finalize model training
    classifier.finalize(verbose=True)

    # Init test epoch
    test_set = list()

    # Get text
    for author_index, author_id in enumerate((args.author1, args.author2)):
        author_path = os.path.join(args.dataset, "total", str(author_id))
        for file_index in test_set_indexes:
            file_path = os.path.join(author_path, str(file_index) + ".txt")
            test_set.append((io.open(file_path, 'r').read(), author_index))
        # end for
    # end for

    # Success rate
    success_rate = Metrics.success_rate(classifier, test_set, verbose=True, debug=True)
    print(u"Success rate : {}".format(success_rate))

# end if