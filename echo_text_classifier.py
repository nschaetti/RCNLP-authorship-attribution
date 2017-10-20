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
import numpy as np
from tools.ResultManager import ResultManager
from tools.functions import create_tokenizer
from parameters.ArgumentBuilder import ArgumentBuilder
from parameters.ParameterSpace import ParameterSpace
from corpus.CrossValidation import CrossValidation
from corpus.Corpus import Corpus

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

    # Params
    reservoir_params = args.get_reservoir_params()

    # Parameter space
    param_space = ParameterSpace(reservoir_params)

    # Experiment
    xp = ResultManager\
    (
        args.get_output(),
        args.get_value('name'),
        args.get_value('description'),
        reservoir_params,
        args.get_n_samples(),
        args.get_fold(),
        verbose=args.verbose()
    )

    # Tokenizer
    tokenizer = create_tokenizer(args.get_value("tokenizer"), args.get_input_params()[0][0])

    # Author list
    authors = reteursC50.get_authors()[:args.get_n_authors()]
    author_list = reteursC50.get_authors_list()[:args.get_n_authors()]

    # First params
    rc_size = int(args.get_reservoir_params()['reservoir_size'][0])
    rc_w_sparsity = args.get_reservoir_params()['w_sparsity'][0]

    # Create W matrix
    w = nsNLP.esn_models.ESNTextClassifier.w(rc_size=rc_size, rc_w_sparsity=rc_w_sparsity)

    # Save classifier
    if args.keep_W():
        xp.save_object(u"w", w)
    # end if

    # W index
    w_index = 0

    # Iterate
    for space in param_space:
        # Params
        reservoir_size = int(space['reservoir_size'])
        w_sparsity = space['w_sparsity']
        leak_rate = space['leak_rate']
        input_scaling = space['input_scaling']
        input_sparsity = space['input_sparsity']
        spectral_radius = space['spectral_radius']
        converter_desc = space['converters']

        # Choose the right tokenizer
        if converter_desc == "wv" or converter_desc == "pos" or converter_desc == "tag":
            tokenizer = create_tokenizer("spacy_wv", converter_desc)
        else:
            tokenizer = create_tokenizer("nltk", converter_desc)
        # end if

        # Set experience state
        xp.set_state(space)

        # Average sample
        average_sample = np.array([])

        # For each sample
        for n in range(args.get_n_samples()):
            # Set sample
            xp.set_sample_state(n)

            # Create ESN text classifier
            classifier = nsNLP.esn_models.ESNTextClassifier.create\
            (
                classes=author_list,
                rc_size=reservoir_size,
                rc_spectral_radius=spectral_radius,
                rc_leak_rate=leak_rate,
                rc_input_scaling=input_scaling,
                rc_input_sparsity=input_sparsity,
                rc_w_sparsity=w_sparsity,
                converter_desc=converter_desc,
                use_sparse_matrix=True if converter_desc == 'oh' else False,
                w=w if args.keep_W() else None
            )

            # Save w matrix
            if not args.keep_W():
                xp.save_object(u"w_{}".format(w_index), classifier.get_w(), info=u"{}".format(space))
            # end if

            # 10 fold cross validation
            cross_validation = CrossValidation(authors)

            # Average
            average_k_fold = np.array([])

            # For each fold
            for k, (train_set, test_set) in enumerate(cross_validation):
                # Set k
                xp.set_fold_state(k)

                # Add to examples
                for index, text in enumerate(train_set):
                    # Add
                    classifier.train(tokenizer(text.x()), text.y())
                # end for

                # Train
                classifier.finalize(verbose=False)

                # Counters
                successes = 0.0

                # Test the classifier
                for text in test_set:
                    # Predict
                    prediction, probs = classifier.predict(tokenizer(text.x()))

                    # Compare
                    if prediction == text.y():
                        successes += 1.0
                    # end if
                # end for

                # Print success rate
                xp.add_result(successes / float(len(test_set)))
                average_k_fold = np.append(average_k_fold, [successes / float(len(test_set))])

                # Reset classifier
                classifier.reset()
            # end for

            # Add
            average_sample = np.append(average_sample, [np.average(average_k_fold)])

            # Delete classifier
            del classifier
        # end for

        # W index
        w_index += 1
    # end for

    # Save experiment results
    xp.save()
# end if
