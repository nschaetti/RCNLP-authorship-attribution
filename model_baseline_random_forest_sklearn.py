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

import numpy as np
import torch.utils.data
from torch.autograd import Variable
import echotorch.nn as etnn
import echotorch.utils
from tools import argument_parsing, dataset, functions, features
import matplotlib.pyplot as plt
import nsNLP
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn import tree
from sklearn import ensemble

####################################################
# Main
####################################################


# Argument builder
args = nsNLP.tools.ArgumentBuilder(desc=u"Argument test")

# Dataset arguments
args.add_argument(command="--dataset-size", name="dataset_size", type=int,
                  help="Ratio of the data set to use (100 percent by default)", extended=False, default=100)
args.add_argument(command="--dataset-start", name="dataset_start", type=int,
                  help="Ratio of the data set to use (100 percent by default)", extended=False, default=100)
args.add_argument(command="--k", name="k", type=int, help="K-Fold Cross Validation", extended=False, default=10)
args.add_argument(command="--ngram", name="ngram", type=int, help="Ngram", extended=False, default=1)
args.add_argument(command="--analyzer", name="analyzer", type=str, help="word, char, char_wb", extended=False, default='word')
args.add_argument(command="--mfw", name="mfw", type=int, help="mfw", extended=False, default=None)
args.add_argument(command="--nestimators", name="nestimators", type=int, help="Nb. estimators", extended=False, default=10)
args.add_argument(command="--n-authors", name="n_authors", type=int,
                      help="Number of authors to include in the test", default=15, extended=False)

# Experiment output parameters
args.add_argument(command="--name", name="name", type=str, help="Experiment's name", extended=False, required=True)
args.add_argument(command="--description", name="description", type=str, help="Experiment's description",
                  extended=False, required=True)
args.add_argument(command="--output", name="output", type=str, help="Experiment's output directory", required=True,
                  extended=False)
args.add_argument(command="--n-samples", name="n_samples", type=int, help="Number of different reservoir to test",
                  default=1, extended=False)
args.add_argument(command="--verbose", name="verbose", type=int, help="Verbose level", default=2, extended=False)

# Parse arguments
args.parse()

# Load from directory
reutersc50_dataset, reuters_loader_train, reuters_loader_test = dataset.load_dataset(
    args.dataset_size,
    sentence_level=False,
    n_authors=args.n_authors
)

# Dataset start
reutersc50_dataset.set_start(0)

# Experiment
xp = nsNLP.tools.ResultManager\
(
    args.output,
    args.name,
    args.description,
    args.get_space(),
    1,
    args.k,
    verbose=args.verbose
)

# Average
average_k_fold = np.array([])

# Print authors
xp.write(u"Authors : {}".format(reutersc50_dataset.authors), log_level=0)

# For each batch
for k in range(10):
    # Set k
    xp.set_fold_state(k)

    # Choose fold
    reuters_loader_train.dataset.set_fold(k)
    reuters_loader_test.dataset.set_fold(k)

    # Samples and classes
    samples = list()
    classes = list()

    # Count vector
    count_vec = CountVectorizer(ngram_range=(1, args.ngram), analyzer=args.analyzer, max_features=args.mfw)

    # TF-IDF transformer
    tf_transformer = TfidfTransformer(use_idf=True)

    # Classifier
    classifier = ensemble.RandomForestClassifier()

    # Pipleline
    text_clf = Pipeline([('vec', count_vec),
                         ('tfidf', tf_transformer),
                         ('clf', classifier)])

    # Choose the right transformer
    reutersc50_dataset.transform = None

    # Get training data for this fold
    for i, data in enumerate(reuters_loader_train):
        # Sample
        inputs, label = data

        # Author
        label = int(label[0])

        # Add
        samples.append(inputs[0])
        classes.append(unicode(label))
    # end for

    # Train
    text_clf.fit(samples, classes)

    # Counters
    successes = 0.0
    count = 0.0

    # Get test data for this fold
    for i, data in enumerate(reuters_loader_test):
        # Sample
        inputs, label = data

        # Author
        label = unicode(int(label[0]))

        # Predict
        prediction = text_clf.predict(inputs)[0]

        # Check
        if label == prediction:
            successes += 1.0
        # end if
        count += 1.0
    # end for

    # Compute accuracy
    accuracy = successes / count

    # Print success rate
    xp.add_result(accuracy)
# end for

xp.save()