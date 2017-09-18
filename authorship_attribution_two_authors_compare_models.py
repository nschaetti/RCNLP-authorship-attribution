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
from scipy import stats
from nsNLP.esn_models.converters.PosConverter import PosConverter
from nsNLP.esn_models.converters.TagConverter import TagConverter
from nsNLP.esn_models.converters.WVConverter import WVConverter
from nsNLP.esn_models.converters.FuncWordConverter import FuncWordConverter
from tools.Logging import Logging

#########################################################################
# Experience settings
#########################################################################

# Exp. info
ex_name = "Authorship Attribution"
ex_instance = "Two Authors compare models"

# Reservoir Properties
rc_leak_rate = 0.01  # Leak rate
rc_input_scaling = 0.25  # Input scaling
rc_size = 2000  # Reservoir size
rc_spectral_radius = 0.1  # Spectral radius
rc_w_sparsity = 0.1
rc_input_sparsity = 0.01
sl_smoothing_param = 0.5

####################################################
# Functions
####################################################


# Create model
def create_model(name):
    """
    Create classifier model
    :param name: Classifier's name
    :return:
    """
    if name == 'SLTextClassifier-DP':
        return SLTextClassifier(classes=[0, 1], smoothing="dp", smoothing_param=sl_smoothing_param)
    elif name == 'SLTextClassifier-JM':
        return SLTextClassifier(classes=[0, 1], smoothing="jm", smoothing_param=sl_smoothing_param)
    elif name == 'TFIDFTextClassifier':
        return TFIDFTextClassifier(classes=[0, 1])
    elif name == 'EchoWordClassifier':
        return EchoWordClassifier(classes=[0, 1], size=rc_size, input_scaling=rc_input_scaling,
                                  leak_rate=rc_leak_rate,
                                  input_sparsity=rc_input_sparsity, converter=converter,
                                  spectral_radius=rc_spectral_radius, w_sparsity=rc_w_sparsity)
    elif name == 'SL2GramTextClassifier-DP':
        return SL2GramTextClassifier(classes=[0, 1], smoothing="dp", smoothing_param=sl_smoothing_param)
    elif name == 'SL2GramTextClassifier-JM':
        return SL2GramTextClassifier(classes=[0, 1], smoothing="jm", smoothing_param=sl_smoothing_param)
    elif name == 'TFIDF2GramTextClassifier':
        return TFIDF2GramTextClassifier(classes=[0, 1])
    # end if
# end create_model

####################################################
# Main function
####################################################

if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(
        description="RCNLP - Compare the Echo Text Classifier to other models with two authors")

    # Argument
    parser.add_argument("--dataset", type=str, help="Dataset's directory")
    parser.add_argument("--author1", type=str, help="First author", default="1")
    parser.add_argument("--author2", type=str, help="Second author", default="2")
    parser.add_argument("--lang", type=str, help="Language (en_core_web_md, ar, en, es, pt)", default='en_core_web_md')
    parser.add_argument("--converter", type=str, help="The text converter to use (fw, pos, tag, wv)", default='pos')
    parser.add_argument("--pca-model", type=str, help="PCA model to load", default=None)
    parser.add_argument("--in-components", type=int, help="Number of principal component to reduce inputs to",
                        default=-1)
    parser.add_argument("--sentence", action='store_true', help="Test sentence classification rate?", default=False)
    parser.add_argument("--k", type=int, help="n-Fold Cross Validation", default=10)
    parser.add_argument("--samples", type=int, help="Number of reservoir to sample", default=50)
    parser.add_argument("--verbose", action='store_true', help="Verbose mode", default=False)
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

    # Choose a text to symbol converter
    if args.converter == "pos":
        converter = PosConverter(lang=args.lang, resize=args.in_components, pca_model=pca_model)
    elif args.converter == "tag":
        converter = TagConverter(lang=args.lang, resize=args.in_components, pca_model=pca_model)
    elif args.converter == "fw":
        converter = FuncWordConverter(lang=args.lang, resize=args.in_components, pca_model=pca_model)
    else:
        converter = WVConverter(lang=args.lang, resize=args.in_components, pca_model=pca_model)
    # end if

    # Prepare training and test set indexes.
    n_fold_samples = int(100 / args.k)
    indexes = np.arange(0, 100, 1)
    indexes.shape = (args.k, n_fold_samples)

    # Create Echo Word Classifier
    classifier = EchoWordClassifier(classes=[0, 1], size=rc_size, input_scaling=rc_input_scaling,
                                    leak_rate=rc_leak_rate,
                                    input_sparsity=rc_input_sparsity, converter=converter,
                                    spectral_radius=rc_spectral_radius, w_sparsity=rc_w_sparsity)

    # Models
    models = list()
    models.append({'name': "SLTextClassifier-DP", "samples": 1, "results": np.zeros(args.k), 'skip': True})
    models.append({'name': "SLTextClassifier-JM", "samples": 1, "results": np.zeros(args.k), 'skip': True})
    models.append({'name': "TFIDFTextClassifier", "samples": 1, "results": np.zeros(args.k), 'skip': True})
    models.append({'name': "EchoWordClassifier", "samples": 10, "results": np.zeros(args.k), 'skip': False})
    models.append({'name': "SL2GramTextClassifier-DP", "samples": 1, "results": np.zeros(args.k), 'skip': True})
    models.append({'name': "SL2GramTextClassifier-JM", "samples": 1, "results": np.zeros(args.k), 'skip': True})
    models.append({'name': "TFIDF2GramTextClassifier", "samples": 1, "results": np.zeros(args.k), 'skip': True})

    # For each model
    for model in models:
        if not model['skip']:
            # Log
            print(u"For model {}".format(model['name']))

            # Array for results
            success_rates = np.zeros((args.k, model['samples']))

            # For each sample
            for s in range(0, model['samples']):
                # Get model
                classifier = create_model(model['name'])

                # Log
                print(u"\tFor sample {}/{}".format(s+1, model['samples']))

                # k-Fold cross validation
                for k in range(0, args.k):
                    # Prepare training and test set.
                    test_set_indexes = indexes[k]
                    training_set_indexes = indexes
                    training_set_indexes = np.delete(training_set_indexes, k, axis=0)
                    training_set_indexes.shape = (100 - n_fold_samples)

                    # Add examples
                    for author_index, author_id in enumerate((args.author1, args.author2)):
                        author_path = os.path.join(args.dataset, "total", author_id)
                        for file_index in training_set_indexes:
                            file_path = os.path.join(author_path, str(file_index) + ".txt")
                            classifier.train(io.open(file_path, 'r').read(), author_index)
                        # end for
                    # end for

                    # Finalize model training
                    classifier.finalize(verbose=args.verbose)

                    # Init test epoch
                    test_set = list()

                    # Get text
                    for author_index, author_id in enumerate((args.author1, args.author2)):
                        author_path = os.path.join(args.dataset, "total", str(author_id))
                        for file_index in test_set_indexes:
                            file_path = os.path.join(author_path, str(file_index) + ".txt")
                            # Document success rate
                            if not args.sentence:
                                test_set.append((io.open(file_path, 'r').read(), author_index))
                            else:
                                # Sentence success rate
                                nlp = spacy.load(args.lang)
                                doc = nlp(io.open(file_path, 'r').read())
                                for sentence in doc.sents:
                                    test_set.append((sentence, author_index))
                                # end for
                            # end if
                        # end for
                    # end for

                    # Success rate
                    success_rate = Metrics.success_rate(classifier, test_set, verbose=args.verbose)
                    print(u"\t\t{} - Success rate : {}".format(k, success_rate))

                    # Save result
                    success_rates[k, s] = success_rate

                    # Reset classifier
                    classifier.reset()
                # end for
                print(u"\t\tAll - Success rate : {}".format(np.average(success_rates[:, s])))
            # end for

            # Average results
            model['results'] = np.average(success_rates, axis=1)

            # Log success
            logging.save_results(u"\tSuccess rate ", np.average(model['results']), display=True)
            logging.save_results(u"\tSuccess rate std ", np.std(model['results']), display=True)
        # end if
    # end for

    # Compare results
    for model1 in models:
        print(model1['name'])
        for model2 in models:
            if model1['name'] == model2['name']:
                logging.save_results(u"\t{} : {}".format(model2['name'], stats.ttest_rel(model1['results'], model2[
                    'results']).pvalue * 100))
            # end if
        # end for
    # end for

    # Open logging dir
    logging.open_dir()

# end if