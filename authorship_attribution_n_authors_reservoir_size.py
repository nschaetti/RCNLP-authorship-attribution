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
    parser.add_argument("--n-authors", type=int, help="Number of authors.", default=50)
    parser.add_argument("--samples", type=int, help="Number of samples to use to assess accuracy.", default=20)
    parser.add_argument("--training-size", type=int, help="Number of texts to train the model.", default=2)
    parser.add_argument("--test-size", type=int, help="Number of texts to assess the model.", default=20)
    parser.add_argument("--lang", type=str, help="Language (ar, en, es, pt)", default='en')
    parser.add_argument("--converter", type=str, help="The text converter to use (fw, pos, tag, wv).", default='pos')
    parser.add_argument("--pca-model", type=str, help="PCA model to load", default=None)
    parser.add_argument("--in-components", type=int, help="Number of principal component to reduce inputs to.",
                        default=-1)
    parser.add_argument("--sentence", action='store_true', help="Test sentence classification rate?", default=False)
    parser.add_argument("--step", type=int, help="Step for reservoir size value", default=50)
    parser.add_argument("--min", type=int, help="Minimum reservoir size value", default=10)
    parser.add_argument("--max", type=int, help="Maximum reservoir size value", default=1000)
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
    success_rate_avg = np.array([])
    success_rate_std = np.array([])

    # >> 3. Prepare training and test set.
    training_set_indexes = np.arange(0, args.training_size, 1)
    test_set_indexes = np.arange(args.training_size, args.training_size + args.test_size, 1)

    # Reservoir sizes
    reservoir_sizes = np.arange(args.min, args.max + 1, args.step)

    # >> 4. For each samples
    for reservoir_size in reservoir_sizes:
        print("Reservoir size %d" % reservoir_size)

        # Average success rate for this training size
        reservoir_size_average_success_rate = np.array([])

        # >> 4. Try n time
        for s in range(0, args.samples):
            try:
                # >> 6. Create Echo Word Classifier
                classifier = RCNLPEchoWordClassifier(size=reservoir_size, input_scaling=rc_input_scaling,
                                                     leak_rate=rc_leak_rate,
                                                     input_sparsity=rc_input_sparsity, converter=converter,
                                                     n_classes=args.n_authors,
                                                     spectral_radius=rc_spectral_radius, w_sparsity=rc_w_sparsity)

                # >> 7. Add examples
                for author_index, author_id in enumerate(np.arange(1, args.n_authors+1, 1)):
                    author_path = os.path.join(args.dataset, "total", author_id)
                    for file_index in training_set_indexes:
                        classifier.add_example(os.path.join(author_path, str(file_index) + ".txt"), author_index)
                    # end for
                # end for

                # >> 8. Train model
                classifier.train()

                # >> 9. Test model performance
                success = 0.0
                count = 0.0
                for author_index, author_id in enumerate(np.arange(1, args.n_authors+1, 1)):
                    author_path = os.path.join(args.dataset, "total", author_id)
                    for file_index in test_set_indexes:
                        file_path = os.path.join(author_path, str(file_index) + ".txt")

                        # Document success rate
                        if not args.sentence:
                            author_pred, _ = classifier.pred(file_path)
                            if author_pred == author_index:
                                success += 1.0
                            # end if
                            count += 1.0
                        else:
                            # Sentence success rate
                            nlp = spacy.load(args.lang)
                            doc = nlp(io.open(file_path, 'r').read())
                            for sentence in doc.sents:
                                sentence_pred, _ = classifier.pred_text(sentence.text)
                                if sentence_pred == author_index:
                                    success += 1.0
                                # end if
                                count += 1.0
                            # end for
                        # end if

                    # end for
                # end for

                # >> 10. Log success
                logging.save_results("Success rate ", (success / count) * 100.0, display=True)

                # >> 11. Save results
                reservoir_size_average_success_rate = np.append(reservoir_size_average_success_rate,
                                                                [(success / count) * 100.0])

                # Delete variables
                del classifier
            except:
                pass
            # end try
        # end for

        # >> 10. Log success
        logging.save_results("Reservoir size ", reservoir_size, display=True)
        logging.save_results("Success rate ", np.average(reservoir_size_average_success_rate), display=True)
        logging.save_results("Success rate std ", np.std(reservoir_size_average_success_rate), display=True)

        # Save results
        success_rate_avg = np.append(success_rate_avg, np.average(reservoir_size_average_success_rate))
        success_rate_std = np.append(success_rate_std, np.std(reservoir_size_average_success_rate))
    # end for

    for index, success_rate in enumerate(success_rate_avg):
        print("(%d, %f)" % (reservoir_sizes[index], success_rate))
    # end for

    for index, success_rate in enumerate(success_rate_std):
        print("(%d, %f)" % (reservoir_sizes[index], success_rate))
    # end for

    # Plot perfs
    plot = RCNLPPlotGenerator(title=ex_name, n_plots=1)
    plot.add_sub_plot(title=ex_instance + ", success rates vs training size.", x_label="Nb. tokens",
                      y_label="Success rates", ylim=[-10, 120])
    plot.plot(y=success_rate_avg, x=reservoir_sizes, yerr=success_rate_std, label="Success rate", subplot=1,
              marker='o', color='b')
    logging.save_plot(plot)

    # Open logging dir
    logging.open_dir()

# end if