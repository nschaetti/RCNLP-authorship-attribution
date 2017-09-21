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

    # Parameter space
    param_space = ParameterSpace(args.get_reservoir_params())

    # Tokenizer
    if args.get_value("tokenizer") == "nltk":
        tokenizer = nsNLP.tokenization.NLTKTokenizer()
    else:
        tokenizer = nsNLP.tokenization.SpacyTokenizer()
    # end if

    # Iterate
    for space in param_space:
        # Params
        reservoir_size = space['reservoir_size']
        w_sparsity = space['w_sparsity']
        leak_rate = space['leak_rate']
        input_scaling = space['input_scaling']
        input_sparsity = space['input_sparsity']
        spectral_radius = space['spectral_radius']

        # Create ESN text classifier
        classifier = nsNLP.esn_models.ESNTextClassifier.create\
        (
            classes=range(15),
            rc_size=reservoir_size,
            rc_spectral_radius=spectral_radius,
            rc_leak_rate=leak_rate,
            rc_input_scaling=input_scaling,
            rc_input_sparsity=input_sparsity,
            rc_w_sparsity=w_sparsity,
            converter=args.get_input_params()
        )

        # 10 fold cross validation
        cross_validation = CrossValidation(reteursC50.get_authors()[:args.get_n_authors()])

        # For each fold
        for train_set, test_set in cross_validation:
            # Train the classifier
            for text in train_set:
                # Train
                classifier.train(tokenizer(text.get_text()), text.get_author().get_name())
            # end for
        # end for

        # Delete classifier
        del classifier
        # end for
# end if
