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
import spacy
from nsNLP.esn_models.converters.PosConverter import PosConverter
from nsNLP.esn_models.converters.TagConverter import TagConverter
from nsNLP.esn_models.converters.WVConverter import WVConverter
from nsNLP.esn_models.converters.FuncWordConverter import FuncWordConverter
from tools.Logging import Logging
from core.tools.RCNLPPlotGenerator import RCNLPPlotGenerator

#########################################################################
# Experience settings
#########################################################################

# Exp. info
ex_name = "Authorship Attribution"
ex_instance = "Author Attribution n authors training size"

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
    parser = argparse.ArgumentParser(description="RCNLP - Authorship clustering with Echo State Network")

    # Argument
    parser.add_argument("--dataset", type=str, help="Dataset's directory.")
    parser.add_argument("--lang", type=str, help="Language (ar, en, es, pt)", default='en')
    parser.add_argument("--converter", type=str, help="The text converter to use (fw, pos, tag, wv).", default='pos')
    parser.add_argument("--pca-model", type=str, help="PCA model to load", default=None)
    parser.add_argument("--in-components", type=int, help="Number of principal component to reduce inputs to.",
                        default=-1)
    parser.add_argument("--samples", type=int, help="Samples", default=20)
    parser.add_argument("--step", type=int, help="Step for training size value", default=5)
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

    # >> 3. Array for results
    doc_success_rate_avg = np.array([])
    sen_success_rate_avg = np.array([])
    doc_success_rate_std = np.array([])
    sen_success_rate_std = np.array([])

    # Training set sizes
    training_set_sizes = np.arange(1, 96, args.step)

    # For each training size
    for training_size in training_set_sizes:
        logging.save_results("Training size ", training_size, display=True)

        # Average success rate for this training size
        training_size_average_doc_success_rate = np.array([])
        training_size_average_sen_success_rate = np.array([])

        # >> 4. Try n time
        for s in range(0, args.samples):

            # >> 5. Prepare training and test set.
            training_set_indexes = np.arange(0, training_size, 1)
            test_set_indexes = np.arange(training_size, 100, 1)

            # >> 6. Create Echo Word Classifier
            classifier = RCNLPEchoWordClassifier(size=rc_size, input_scaling=rc_input_scaling, leak_rate=rc_leak_rate,
                                                 input_sparsity=rc_input_sparsity, converter=converter, n_classes=2,
                                                 spectral_radius=rc_spectral_radius, w_sparsity=rc_w_sparsity)

            # >> 7. Add authors examples
            author_indexes = np.random.choice(np.arange(1, 51, 1), 2, replace=False)
            for author_index, author_id in enumerate((author_indexes[0], author_indexes[1])):
                author_path = os.path.join(args.dataset, "total", str(author_id))
                for file_index in training_set_indexes:
                    classifier.add_example(os.path.join(author_path, str(file_index) + ".txt"), author_index)
                # end for
            # end for

            # >> 8. Train model
            classifier.train()

            # >> 9. Test model performance
            doc_success = sen_success = 0.0
            doc_count = sen_count = 0.0
            for author_index, author_id in enumerate((author_indexes[0], author_indexes[1])):
                author_path = os.path.join(args.dataset, "total", str(author_id))
                for file_index in test_set_indexes:
                    file_path = os.path.join(author_path, str(file_index) + ".txt")

                    # Doc. success rate
                    author_pred = classifier.pred(os.path.join(author_path, str(file_index) + ".txt"))
                    if author_pred == author_index:
                        doc_success += 1.0
                    # end if
                    doc_count += 1.0

                    # Sentence success rate
                    nlp = spacy.load(args.lang)
                    doc = nlp(io.open(file_path, 'r').read())
                    for sentence in doc.sents:
                        sentence_pred, _, _ = classifier.pred_text(sentence.text)
                        if sentence_pred == author_index:
                            sen_success += 1.0
                        # end if
                        sen_count += 1.0
                    # end for
                # end for
            # end for

            # >> 11. Save results
            training_size_average_doc_success_rate = np.append(training_size_average_doc_success_rate,
                                                               [(doc_success / doc_count) * 100.0])
            training_size_average_sen_success_rate = np.append(training_size_average_sen_success_rate,
                                                               [(sen_success / sen_count) * 100.0])

            # Delete variables
            del classifier
        # end for

        # >> 10. Log success
        logging.save_results("Doc. success rate ", np.average(training_size_average_doc_success_rate), display=True)
        logging.save_results("Sen. success rate ", np.average(training_size_average_sen_success_rate), display=True)

        # Save results
        doc_success_rate_avg = np.append(doc_success_rate_avg, np.average(training_size_average_doc_success_rate))
        doc_success_rate_std = np.append(doc_success_rate_std, np.std(training_size_average_doc_success_rate))
        sen_success_rate_avg = np.append(sen_success_rate_avg, np.average(training_size_average_sen_success_rate))
        sen_success_rate_std = np.append(sen_success_rate_std, np.std(training_size_average_sen_success_rate))
    # end for

    # Plot perfs
    plot = RCNLPPlotGenerator(title=ex_name, n_plots=1)
    plot.add_sub_plot(title=ex_instance + ", success rates vs training size.", x_label="Nb. text file",
                      y_label="Success rates", ylim=[-10, 120])
    plot.plot(y=doc_success_rate_avg, x=training_set_sizes, yerr=doc_success_rate_std, label="Doc. success rate", subplot=1,
              marker='o', color='b')
    plot.plot(y=sen_success_rate_avg, x=training_set_sizes, yerr=sen_success_rate_std, label="Sen. success rate", subplot=1,
              marker='o', color='r')
    logging.save_plot(plot)

    # Open logging dir
    logging.open_dir()

# end if