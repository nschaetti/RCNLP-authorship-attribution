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
from corpus.CrossValidation import CrossValidation
from corpus.Corpus import Corpus

####################################################
# Functions
####################################################


# Converter in
def converter_in(converters_desc, converter):
    """
    Is the converter in the desc
    :param converters_desc:
    :param converter:
    :return:
    """
    for converter_desc in converters_desc:
        if converter in converter_desc:
            return True
        # end if
    # end for
    return False
# end converter_in

####################################################
# Main function
####################################################

# Main function
if __name__ == "__main__":

    # Argument builder
    args = nsNLP.tools.ArgumentBuilder(desc=u"Argument test")

    # Dataset arguments
    args.add_argument(command="--dataset", name="dataset", type=str,
                      help="JSON file with the file description for each authors", required=True, extended=False)
    args.add_argument(command="--dataset-size", name="dataset_size", type=float,
                      help="Ratio of the data set to use (100 percent by default)", extended=False, default=100.0)
    args.add_argument(command="--k", name="k", type=int, help="K-Fold Cross Validation", extended=False, default=10)

    # Author parameters
    args.add_argument(command="--n-authors", name="n_authors", type=int,
                      help="Number of authors to include in the test", default=15, extended=False)
    for i in range(15):
        args.add_argument(command="--author{}".format(i), name="author{}".format(i), type=str,
                          help="{}th author to test".format(i), extended=False)
    # end for

    # ESN arguments
    args.add_argument(command="--reservoir-size", name="reservoir_size", type=float, help="Reservoir's size",
                      required=True, extended=True)
    args.add_argument(command="--spectral-radius", name="spectral_radius", type=float, help="Spectral radius",
                      default="1.0", extended=True)
    args.add_argument(command="--leak-rate", name="leak_rate", type=str, help="Reservoir's leak rate", extended=True,
                      default="1.0")
    args.add_argument(command="--input-scaling", name="input_scaling", type=str, help="Input scaling", extended=True,
                      default="0.5")
    args.add_argument(command="--input-sparsity", name="input_sparsity", type=str, help="Input sparsity", extended=True,
                      default="0.05")
    args.add_argument(command="--w-sparsity", name="w_sparsity", type=str, help="W sparsity", extended=True,
                      default="0.05")
    args.add_argument(command="--converters", name="converters", type=str,
                      help="The text converters to use (fw, pos, tag, wv, oh)", default='oh', extended=True)
    args.add_argument(command="--pca-path", name="pca_path", type=str, help="PCA model to load", default=None,
                      extended=False)
    args.add_argument(command="--keep-w", name="keep_w", action='store_true', help="Keep W matrix", default=False,
                      extended=False)
    args.add_argument(command="--aggregation", name="aggregation", type=str, help="Output aggregation method", extended=True,
                      default="average")
    args.add_argument(command="--state-gram", name="state_gram", type=str, help="State-gram value",
                      extended=True, default="1")

    # Tokenizer and word vector parameters
    args.add_argument(command="--tokenizer", name="tokenizer", type=str,
                      help="Which tokenizer to use (spacy, nltk, spacy-tokens)", default='nltk', extended=False)
    args.add_argument(command="--lang", name="lang", type=str, help="Tokenizer language parameters", default='en',
                      extended=False)

    # Experiment output parameters
    args.add_argument(command="--name", name="name", type=str, help="Experiment's name", extended=False, required=True)
    args.add_argument(command="--description", name="description", type=str, help="Experiment's description",
                      extended=False, required=True)
    args.add_argument(command="--output", name="output", type=str, help="Experiment's output directory", required=True,
                      extended=False)
    args.add_argument(command="--sentence", name="sentence", action='store_true',
                      help="Test sentence classification rate?", default=False, extended=False)
    args.add_argument(command="--n-samples", name="n_samples", type=int, help="Number of different reservoir to test",
                      default=1, extended=False)
    args.add_argument(command="--verbose", name="verbose", type=int, help="Verbose level", default=2, extended=False)

    # Parse arguments
    args.parse()

    # Corpus
    reteursC50 = Corpus(args.dataset)

    # Parameter space
    param_space = nsNLP.tools.ParameterSpace(args.get_space())

    # Experiment
    xp = nsNLP.tools.ResultManager\
    (
        args.output,
        args.name,
        args.description,
        args.get_space(),
        args.n_samples,
        args.k,
        verbose=args.verbose
    )

    # Author list
    authors = reteursC50.get_authors()[:args.n_authors]
    author_list = reteursC50.get_authors_list()[:args.n_authors]

    # Print authors
    xp.write(u"Authors : {}".format(author_list), log_level=0)

    # First params
    rc_size = int(args.get_space()['reservoir_size'][0])
    rc_w_sparsity = args.get_space()['w_sparsity'][0]

    # Create W matrix
    w = nsNLP.esn_models.ESNTextClassifier.w(rc_size=rc_size, rc_w_sparsity=rc_w_sparsity)

    # Save classifier
    if args.keep_w:
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
        aggregation = space['aggregation'][0][0]
        state_gram = space['state_gram']

        # Choose the right tokenizer
        if converter_in(converter_desc, "wv") or \
                converter_in(converter_desc, "pos") or \
                converter_in(converter_desc, "tag"):
            tokenizer = create_tokenizer("spacy_wv")
        else:
            tokenizer = create_tokenizer("nltk")
        # end if

        # Set experience state
        xp.set_state(space)

        # Average sample
        average_sample = np.array([])

        # For each sample
        for n in range(args.n_samples):
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
                use_sparse_matrix=True if converter_in(converter_desc, "oh") else False,
                w=w if args.keep_w else None,
                aggregation=aggregation,
                state_gram=state_gram
            )

            # Save w matrix
            if not args.keep_w:
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
