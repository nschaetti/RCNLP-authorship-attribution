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


####################################################
# Main function
####################################################

if __name__ == "__main__":

    # Argument builder
    args = nsNLP.tools.ArgumentBuilder(desc=u"Argument test")

    # Add arguments
    args.add_argument(command="--dataset", name="dataset", type=str,
                      help="JSON file with the file description for each authors", required=True, extended=False)
    args.add_argument(command="--reservoir-size", name="reservoir_size", type=float, help="Reservoir's size",
                      required=True, extended=True)
    args.add_argument(command="--spectral-radius", name="spectral_radius", type=float, help="Spectral radius",
                      required=True, extended=True)

    # Parse arguments
    args.parse()

    # Print arguments
    print(args.reservoir_size)
    print(args.get_space())

    # Parameter space
    param_space = nsNLP.tools.ParameterSpace(args.get_space())

    # Go through space
    for param_point in param_space:
        print(param_point)
    # end for

# end if