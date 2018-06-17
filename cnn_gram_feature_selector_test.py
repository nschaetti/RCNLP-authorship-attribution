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
import torch.utils.data
from torch.autograd import Variable
from torchlanguage import datasets
import torchlanguage.transforms as transforms
import echotorch.nn as etnn
import torchlanguage.models as models
import echotorch.utils
import matplotlib.pyplot as plt

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


# Load character embedding
def load_character_embedding(emb_path):
    """
    Load character embedding
    :param emb_path:
    :return:
    """
    token_to_ix, weights = torch.load(open(emb_path, 'rb'))
    return token_to_ix, weights
# end load_character_embedding

####################################################
# Main
####################################################


# Argument builder
args = nsNLP.tools.ArgumentBuilder(desc=u"Argument test")

# Dataset arguments
args.add_argument(command="--dataset", name="dataset", type=str, default="data/",
                  help="JSON file with the file description for each authors", required=False, extended=False)
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
args.add_argument(command="--transformer", name="transformer", type=str,
                  help="The text transformer to use (fw, pos, tag, wv, c1, c2, c3, cnn)", default='wv', extended=True)
args.add_argument(command="--pca-path", name="pca_path", type=str, help="PCA model to load", default=None,
                  extended=False)
args.add_argument(command="--keep-w", name="keep_w", action='store_true', help="Keep W matrix", default=False,
                  extended=False)
args.add_argument(command="--aggregation", name="aggregation", type=str, help="Output aggregation method", extended=True,
                  default="average")
args.add_argument(command="--state-gram", name="state_gram", type=str, help="State-gram value",
                  extended=True, default="1")
args.add_argument(command="--voc-size", name="voc_size", type=int, help="Voc. size",
                  default=30000, extended=False)
args.add_argument(command="--feedbacks", name="feedbacks", action='store_true', help="Use feedbacks?",
                  default=False, extended=False)
args.add_argument(command="--feedbacks-sparsity", name="feedbacks_sparsity", type=str, help="Feedbacks sparsity", extended=True,
                  default="0.05")

# Tokenizer and word vector parameters
args.add_argument(command="--tokenizer", name="tokenizer", type=str,
                  help="Which tokenizer to use (spacy, nltk, spacy-tokens)", default='nltk', extended=False)
args.add_argument(command="--lang", name="lang", type=str, help="Tokenizer language parameters",
                  default='en_vectors_web_lg', extended=True)
args.add_argument(command="--embedding", name="embedding", type=str, help="Which word embedding to use? (glove, word2vec, skipgram, pretrained)",
                  default='glove', extended=True)
args.add_argument(command="--embedding-path", name="embedding_path", type=str, help="Embedding directory",
                  default='~/Projets/TURING/Datasets/', extended=False)

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
args.add_argument(command="--cuda", name="cuda", action='store_true',
                  help="Use CUDA?", default=False, extended=False)

# Parse arguments
args.parse()

# CUDA
use_cuda = torch.cuda.is_available() if args.cuda else False

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

# CNN Glove Feature Selector
cgfs = models.cgfs(pretrained=True, n_gram=2, n_features=60)

# Remove last linear layer
cgfs.linear2 = echotorch.nn.Identity()

# Transformer
transformer = transforms.Compose([
    transforms.GloveVector(),
    transforms.ToNGram(n=2, overlapse=True),
    transforms.Reshape((-1, 1, 2, 300)),
    transforms.FeatureSelector(cgfs, 60, to_variable=True),
    transforms.Reshape((1, -1, 60)),
    transforms.Normalize(mean=-4.56512329954, std=0.911449706065)
])

# Reuters C50 dataset
reutersloader = torch.utils.data.DataLoader(
    datasets.ReutersC50Dataset(root=args.dataset, download=True, n_authors=args.n_authors, transform=transformer),
    batch_size=1, shuffle=False)

# Print authors
xp.write(u"Authors : {}".format(reutersloader.dataset.authors), log_level=0)

# First params
rc_size = int(args.get_space()['reservoir_size'][0])
rc_w_sparsity = args.get_space()['w_sparsity'][0]

# Create W matrix
w = etnn.ESNCell.generate_w(rc_size, rc_w_sparsity)

# Save classifier
if args.keep_w:
    xp.save_object(u"w", w)
# end if

# W index
w_index = 0

# Last space
last_space = dict()

# Iterate
for space in param_space:
    # Params
    reservoir_size = int(space['reservoir_size'])
    w_sparsity = space['w_sparsity']
    leak_rate = space['leak_rate']
    input_scaling = space['input_scaling']
    input_sparsity = space['input_sparsity']
    spectral_radius = space['spectral_radius']
    aggregation = space['aggregation'][0][0]
    state_gram = space['state_gram']
    lang = space['lang'][0][0]
    embedding = space['embedding'][0][0]

    # Set experience state
    xp.set_state(space)

    # Average sample
    average_sample = np.array([])

    # For each sample
    for n in range(args.n_samples):
        # Set sample
        xp.set_sample_state(n)
        
        # ESN cell
        esn = etnn.LiESN(
            input_dim=transformer.transforms[-1].input_dim,
            hidden_dim=reservoir_size,
            output_dim=reutersloader.dataset.n_authors,
            spectral_radius=spectral_radius,
            sparsity=input_sparsity,
            input_scaling=input_scaling,
            w_sparsity=w_sparsity,
            w=w if args.keep_w else None,
            learning_algo='inv',
            leaky_rate=leak_rate
        )
        if use_cuda:
            esn.cuda()
        # end if

        # Get training data for this fold
        for i, data in enumerate(reutersloader):
            # Inputs and labels
            inputs, labels, time_labels = data
            plt.imshow(inputs[0, 0].t().numpy(), cmap='Greys')
            plt.show()
        # end for

        # Finalize training
        esn.finalize()

        # Get test data for this fold
        for i, data in enumerate(reutersloader):
            # Inputs and labels
            inputs, labels, time_labels = data
        # end for

        # Reset learning
        esn.reset()
    # end for

    # Last space
    last_space = space
# end for

# Save experiment results
xp.save()
