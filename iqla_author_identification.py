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
import argparse
import pickle
import numpy as np
import itertools
from random import shuffle
import json
import logging as log
from nsNLP.esn_models.converters.PosConverter import PosConverter
from nsNLP.esn_models.converters.TagConverter import TagConverter
from nsNLP.esn_models.converters.WVConverter import WVConverter
from nsNLP.esn_models.converters.FuncWordConverter import FuncWordConverter
from tools.Logging import Logging

#########################################################################
# Experience settings
#########################################################################

# Exp. info
ex_name = "Authorship Attribution Experience"
ex_instance = "Author Attribution Multiple Authors"

# Reservoir Properties
rc_leak_rate = 0.1  # Leak rate
rc_input_scaling = 0.25  # Input scaling
rc_size = 100  # Reservoir size
rc_spectral_radius = 0.99  # Spectral radius
rc_w_sparsity = 0.1
rc_input_sparsity = 0.1

####################################################
# Functions
####################################################


def get_combinations(n_authors, n_samples):
    """
    Return n_samples random combinations of n_authors
    :param n_authors:
    :param n_samples:
    :return:
    """
    comb = itertools.combinations(np.arange(1, 51), n_authors)
    combl = list()
    for c in comb:
        combl.append(c)
    # end for
    return shuffle(combl)[:n_samples]
# end get_combinations

####################################################
# Main function
####################################################

if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description="RCNLP - Authorship attribution witn ESN on the IQLA dataset")

    # Argument
    parser.add_argument("--dataset", type=str, help="Dataset's directory")
    parser.add_argument("--lang", type=str, help="Language (ar, en, es, pt)", default='it')
    parser.add_argument("--author", type=str, help="The author to identify", required=True)
    parser.add_argument("--converter", type=str, help="The text converter to use (fw, pos, tag, wv)", default='pos')
    parser.add_argument("--pca-model", type=str, help="PCA model to load", default="")
    parser.add_argument("--in-components", type=int, help="Number of principal component to reduce inputs to",
                        default=-1)
    parser.add_argument("--k", type=int, help="n-Fold Cross Validation.", default=10)
    args = parser.parse_args()

    # Logging
    logging = RCNLPLogging(exp_name=ex_name, exp_inst=ex_instance,
                           exp_value=RCNLPLogging.generate_experience_name(locals()))
    logging.save_globals()
    logging.save_variables(locals())

    # PCA model
    pca_model = None
    if args.pca_model != "":
        pca_model = pickle.load(open(args.pca_model, 'r'))
    # end if

    # Choose a text to symbol converter.
    if args.converter == "pos":
        converter = RCNLPPosConverter(resize=args.in_components, pca_model=pca_model)
    elif args.converter == "tag":
        converter = RCNLPTagConverter(resize=args.in_components, pca_model=pca_model)
    elif args.converter == "fw":
        converter = RCNLPFuncWordConverter(resize=args.in_components, pca_model=pca_model)
    else:
        converter = RCNLPWordVectorConverter(resize=args.in_components, pca_model=pca_model)
    # end if

    # Load texts information
    with open(os.path.join(args.dataset, "texts.json"), 'r') as f:
        texts_data = json.load(f)
    # end with
    text_codes = texts_data.keys()

    # Load author data
    with open(os.path.join(args.dataset, "authors.json"), 'r') as f:
        authors_data = json.load(f)
    # end with

    # Prepare training and test set indexes.
    n_texts = len(texts_data)
    n_fold_samples = int(n_texts / args.k)
    indexes = np.arange(0, n_texts, 1)
    indexes.shape = (args.k, n_fold_samples)

    # Array for results
    average_success_rate = np.array([])

    # n-Fold cross validation
    for k in range(0, args.k):
        # Info
        print(u"K-Fold {}".format(k))

        # Prepare training and test set
        test_set_indexes = indexes[k]
        training_set_indexes = indexes
        training_set_indexes = np.delete(training_set_indexes, k, axis=0)
        training_set_indexes.shape = (n_texts - n_fold_samples)

        # Create Echo Word Classifier
        classifier = RCNLPEchoWordClassifier(size=rc_size, input_scaling=rc_input_scaling, leak_rate=rc_leak_rate,
                                             input_sparsity=rc_input_sparsity, converter=converter, n_classes=2,
                                             spectral_radius=rc_spectral_radius, w_sparsity=rc_w_sparsity)

        # Add examples
        print(u"Adding examples...")
        for training_index in training_set_indexes:
            training_text_path = os.path.join(args.dataset, text_codes[training_index] + ".txt")
            if texts_data[text_codes[training_index]] == args.author:
                training_text_author = args.author
            else:
                training_text_author = "Unknown"
            # end if
            classifier.add_example(training_text_path, training_text_author)
        # end for

        # Train model
        print(u"Training...")
        classifier.train()

        # Test model performance
        success = 0.0
        count = 1.0
        for test_index in test_set_indexes:
            test_text_path = os.path.join(args.dataset, text_codes[test_index] + ".txt")
            if texts_data[text_codes[test_index]] == args.author:
                observed_author = args.author
            else:
                observed_author = "Unknown"
            # end if
            predicted_author = classifier.pred(test_text_path)
            if observed_author == predicted_author:
                success += 1
            # end if
            count += 1
        # end for

        # >> Save results
        average_success_rate = np.append(average_success_rate, [(success / count) * 100.0])

        # Delete variables
        del classifier
    # end for

    # Log results
    logging.save_results(u"Average success rate ", np.average(average_success_rate), display=True)
    logging.save_results(u"Success rate std ", np.std(average_success_rate), display=True)

# end if