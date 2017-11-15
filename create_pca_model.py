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

import argparse
import numpy as np
import os
import nsNLP
import io
from sklearn.decomposition import PCA
import pickle
from tools.functions import create_tokenizer
from corpus.Corpus import Corpus

####################################################
# Main function
####################################################

if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description="Create PCA model of symbolic representations.")

    # Argument
    parser.add_argument("--dataset", type=str, help="JSON file with the file description for each authors")
    parser.add_argument("--components", type=int, help="Number of principal component to reduce inputs to", required=True)
    parser.add_argument("--converter", type=str, help="The text converter to use (fw, pos, tag, wv)", required=True)
    parser.add_argument("--output", type=str, help="Output model file", default='pca_output.p')
    args = parser.parse_args()

    # Corpus
    reteursC50 = Corpus(args.dataset)

    # Choose the right tokenizer
    if args.converter == "wv" or args.converter == "pos" or args.converter == "tag":
        tokenizer = create_tokenizer("spacy_wv")
    else:
        tokenizer = create_tokenizer("nltk")
    # end if

    # >> 1. Convert the text to symbolic or continuous representations
    if args.converter == "pos":
        converter = nsNLP.esn_models.converters.PosConverter()
    elif args.converter == "tag":
        converter = nsNLP.esn_models.converters.TagConverter()
    elif args.converter == "fw":
        converter = nsNLP.esn_models.converters.FuncWordConverter()
    elif args.converter == "letter":
        converter = nsNLP.esn_models.converters.LetterConverter()
    elif args.converter == "wv":
        converter = nsNLP.esn_models.converters.WVConverter()
    else:
        raise Exception(u"Unknown converter type: {}".format(args.converter))
    # end if

    # Index
    index = 0

    # For each author
    for author in reteursC50.get_authors():
        # For each author's text
        for text in author.get_texts():
            # Generate states for first author
            print(u"Transforming text {} from author {} to symbols".format(text.get_path(), author.get_name()))
            # Convert the text to Temporal Vector Representation
            doc_array = converter(tokenizer(text.get_text()))

            # Add
            if index == 0:
                symb_rep = doc_array
            else:
                symb_rep = np.vstack((symb_rep, doc_array))
            # end if
            index += 1
        # end for
    # end for

    # PCA
    pca = PCA(n_components=args.components)
    pca.fit(symb_rep)

    # Explained variance
    print("Explained variance : ")
    print(pca.explained_variance_)

    # Explained variance ratio
    print("Explained variance ratio : ")
    print(pca.explained_variance_ratio_)

    # Mean
    print("Mean : ")
    print(pca.mean_)

    # Noise variance
    print("Noise variance : ")
    print(pca.noise_variance_)

    # Save
    pickle.dump(pca, open(args.output, 'w'))

# end if