#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Auteur : Nils Schaetti <nils.schaetti@unine.ch>
# Date : 01.02.2017 17:59:05
# Lieu : Nyon, Suisse
#
# This file is part of the Reservoir Computing Memory Project.
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
#

import matplotlib.pyplot as plt
import numpy as np


# Manage and save results
class ResultManager(object):
    """
    Manage and save results
    """

    # Constructor
    def __init__(self, params_dict, n_samples, k=10):
        """
        Constructor
        :param params_dict:
        """
        self._params_dict = params_dict
        self._n_samples = n_samples
        self._k = k

        # Generate result matrix
        self._result_matrix = self._generate_matrix()
    # end __init__

    ###########################################
    # Public
    ###########################################

    # Change parameter state
    def set_state(self, param, value):
        """
        Change parameter state
        :param param:
        :param value:
        :return:
        """
        pass
    # end set_state

    # Change sample state
    def set_sample_state(self, n_sample):
        """
        Change sample state
        :param n_sample:
        :return:
        """
        pass
    # end set_sample_state

    # Change fold state
    def set_fold_state(self, k):
        """
        Change fold state
        :param k:
        :return:
        """
        pass
    # end set_fold_state

    # Save result
    def add_result(self, success_rate):
        """
        Save result
        :param success_rate:
        :return:
        """
        pass
    # end add_result

    # Save results
    def save(self, output_dir):
        """
        Save results
        :param output_dir:
        :return:
        """
        pass
    # end save

    # Add object
    def add_object(self, obj):
        """
        Add object
        :param obj:
        :return:
        """
        pass
    # end add_object

    ###########################################
    # Private
    ###########################################

    # Generate result matrix
    def _generate_matrix(self):
        """
        Generate matrix
        :return:
        """
        # Dim counters
        dims = list()

        # For each param
        for param in self._params_dict.keys():
            dims.append(len(self._params_dict[param]))
        # end for

        # Add samples
        dims.append(self._n_samples)

        # Add cross validation
        dims.append(self._k)

        return np.zeros(dims)
    # end _generate_matrix

# end
