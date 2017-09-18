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

#########################################################################
# Experience settings
#########################################################################

# Exp. info
ex_name = "Authorship Attribution Experience"
ex_instance = "Author Attribution Reservoir Sample"

# Reservoir Properties
rc_leak_rate = 0.1  # Leak rate
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
    parser.add_argument("--author", type=str, help="Author to test.", default="1")
    parser.add_argument("--training-size", type=int, help="Number of texts to train the model.", default=2)
    parser.add_argument("--test-size", type=int, help="Number of texts to assess the model.", default=20)
    parser.add_argument("--lang", type=str, help="Language (ar, en, es, pt)", default='en')
    parser.add_argument("--converter", type=str, help="The text converter to use (fw, pos, tag, wv).", default='pos')
    parser.add_argument("--pca-model", type=str, help="PCA model to load", default=None)
    parser.add_argument("--in-components", type=int, help="Number of principal component to reduce inputs to.",
                        default=-1)
    parser.add_argument("--sentence", action='store_true', help="Test sentence classification rate?", default=False)
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

    # >> 2. Array for results
    average_success_rate = np.array([])

    # >> 3. Prepare training and test set.
    training_set_indexes = np.arange(0, args.training_size, 1)
    test_set_indexes = np.arange(args.training_size, args.training_size + args.test_size, 1)

    # >> 6. Create Echo Word Classifier
    classifier = RCNLPEchoWordClassifier(size=rc_size, input_scaling=rc_input_scaling, leak_rate=rc_leak_rate,
                                         input_sparsity=rc_input_sparsity, converter=converter,
                                         n_classes=2,
                                         spectral_radius=rc_spectral_radius, w_sparsity=rc_w_sparsity)

    # >> 7. Positive examples
    author_path = os.path.join(args.dataset, "total", args.author)
    for file_index in training_set_indexes:
        classifier.add_example(os.path.join(author_path, str(file_index) + ".txt"), 0)
    # end for

    # Negative examples
    for file_index in training_set_indexes:
        negative_authors = np.arange(1, 51, 1)
        np.delete(negative_authors, args.author)
        author_path = os.path.join(args.dataset, "total", np.random.choice(negative_authors, 1)[0])
        classifier.add_example(os.path.join(author_path, str(file_index) + ".txt"), 1)
    # end for

    # >> 8. Train model
    classifier.train()

    # >> 9. Test truth author
    success = 0.0
    count = 0.0
    author_path = os.path.join(args.dataset, "total", args.author)
    for file_index in test_set_indexes:
        file_path = os.path.join(author_path, str(file_index) + ".txt")

        # Document success rate
        if not args.sentence:
            author_pred, _ = classifier.pred(file_path)
            if author_pred == 0:
                success += 1.0
            # end if
            count += 1.0
        else:
            # Sentence success rate
            nlp = spacy.load(args.lang)
            doc = nlp(io.open(file_path, 'r').read())
            for sentence in doc.sents:
                sentence_pred, _ = classifier.pred_text(sentence.text)
                if sentence_pred == 0:
                    success += 1.0
                # end if
                count += 1.0
            # end for
        # end if
    # end for

    # Test engative authors
    negative_authors = np.arange(1, 51, 1)
    np.delete(negative_authors, args.author)
    for file_index in test_set_indexes:
        author_path = os.path.join(args.dataset, "total", np.random.choice(negative_authors, 1)[0])
        file_path = os.path.join(author_path, str(file_index) + ".txt")

        # Document success rate
        if not args.sentence:
            author_pred, _ = classifier.pred(file_path)
            if author_pred == 1:
                success += 1.0
            # end if
            count += 1.0
        else:
            # Sentence success rate
            nlp = spacy.load(args.lang)
            doc = nlp(io.open(file_path, 'r').read())
            for sentence in doc.sents:
                sentence_pred, _ = classifier.pred_text(sentence.text)
                if sentence_pred == 1:
                    success += 1.0
                # end if
                count += 1.0
            # end for
        # end if
    # end for

    # >> 10. Log success
    logging.save_results("Success rate ", (success / count) * 100.0, display=True)

# end if