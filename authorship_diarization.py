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
from core.tools.PAN16AuthorDiarizationLoader import PAN16AuthorDiarizationLoader


if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description="RCNLP - Authorship Diarization with Echo State Network")

    # Argument
    parser.add_argument("--file", type=str, help="Input text file")
    parser.add_argument("--truth", type=str, help="Truth JSON file")
    args = parser.parse_args()

    # Parse file
    loader = PAN16AuthorDiarizationLoader()
    data_set = loader(args.truth, args.file)

    # For each author
    for author in data_set:
        print("#################################################################")
        # For each text
        for text in author:
            print("[" + text + "]")
        # end for
    # end for

# end if