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
import matplotlib.pyplot as plt
import pickle
import numpy as np
import Oger
import mdp
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
ex_instance = "Author Attribution"

# Reservoir Properties
rc_leak_rate = 0.05  # Leak rate
rc_input_scaling = 0.25  # Input scaling
rc_size = 100  # Reservoir size
rc_spectral_radius = 0.99  # Spectral radius
rc_w_sparsity = 0.1
rc_input_sparsity = 0.1

####################################################
# Main function
####################################################

if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description="RCNLP - Authorship attribution with Echo State Network")

    # Argument
    parser.add_argument("--dataset", type=str, help="Dataset's directory.")
    parser.add_argument("--author1", type=str, help="Author 1' ID.")
    parser.add_argument("--author2", type=str, help="Author 2's ID.")
    parser.add_argument("--lang", type=str, help="Language (ar, en, es, pt)", default='en')
    parser.add_argument("--converter", type=str, help="The text converter to use (fw, pos, tag, wv).", default='pos')
    parser.add_argument("--pca-model", type=str, help="PCA model to load", default=None)
    parser.add_argument("--in-components", type=int, help="Number of principal component to reduce inputs to.",
                        default=-1)
    parser.add_argument("--lr1", type=float, help="First leaky rate", default=0.1)
    parser.add_argument("--lr2", type=float, help="Second leaky rate", default=0.05)
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

    # >> 1. Choose a text to symbol converter.
    if args.converter == "pos":
        converter = RCNLPPosConverter(resize=args.in_components, pca_model=pca_model)
    elif args.converter == "tag":
        converter = RCNLPTagConverter(resize=args.in_components, pca_model=pca_model)
    elif args.converter == "fw":
        converter = RCNLPFuncWordConverter(resize=args.in_components, pca_model=pca_model)
    else:
        converter = RCNLPWordVectorConverter(resize=args.in_components, pca_model=pca_model)
    # end if

    # W matrix
    w = mdp.numx.random.choice([0.0, 1.0], (rc_size, rc_size), p=[1.0 - rc_w_sparsity, rc_w_sparsity])
    w[w == 1] = mdp.numx.random.rand(len(w[w == 1]))

    # >> 4. For each leak rate
    for index, leaky_rate in enumerate((args.lr1, args.lr2)):
        print("Leaky rate %f" % leaky_rate)

        # >> 5. Prepare training and test set.
        training_set_indexes = np.arange(1, 100, 1)
        test_set_indexes = np.array([0])

        # >> 6. Create Echo Word Classifier
        classifier = RCNLPEchoWordClassifier(size=rc_size, input_scaling=rc_input_scaling, leak_rate=leaky_rate,
                                             input_sparsity=rc_input_sparsity, converter=converter, n_classes=2,
                                             spectral_radius=rc_spectral_radius, w_sparsity=rc_w_sparsity, w=w)

        # >> 7. Add examples
        for author_index, author_id in enumerate((args.author1, args.author2)):
            author_path = os.path.join(args.dataset, "total", author_id)
            for file_index in training_set_indexes:
                classifier.add_example(os.path.join(author_path, str(file_index) + ".txt"), author_index)
            # end for
        # end for

        # >> 8. Train model
        classifier.train()

        # >> 9. Test model performance
        author_index = 0
        author_id = args.author1
        author_path = os.path.join(args.dataset, "total", author_id)
        print("Testing model performances with %d text files for author from %s..." % (test_set_indexes.shape[0],
                                                                                       author_path))
        classifier.pred(os.path.join(author_path, str(0) + ".txt"), show_graph=False, print_outputs=True)

        # Delete variables
        del classifier
    # end for

# end if