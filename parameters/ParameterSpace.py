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

import argparse
import numpy as np


# Iterate through experience parameters
class ParameterSpace(object):
    """
    Iterate through experience parameters
    """

    # Constructor
    def __init__(self, parameters_dict):
        """
        Constructor
        :param paremeters_dict:
        """
        self._parameters_dict = parameters_dict
        self._parameters = parameters_dict.keys()
        self._pos = 0

        # Generate all parameters
        self._parameter_space = self._generate_parameters_space()
    # end __init__

    ##########################################
    # Private
    ##########################################

    # Iterate through parameter space
    def _iterate_through_parameter_space(self, space):
        """
        Iterate through parameter space
        :param space:
        :return:
        """
        if len(space) == 1:
            return space[0]
        # end if

        # New spaces
        new_spaces = list()

        # Lower lever
        subspaces = self._iterate_through_parameter_space(space[1:])

        # For each value
        for position in space[0]:
            # For each subspace
            for subspace in subspaces:
                s = position.copy()
                s.update(subspace)
                new_spaces.append(s)
            # end for
        # end for

        return new_spaces
    # end _iterate_through_parameter_space

    # Generate all parameters
    def _generate_parameters_space(self):
        """
        Generate paramters space
        :return:
        """
        params_space = list()

        # For each param
        for param in self._parameters:
            params_space.append(self._generate_param_space(param, self._parameters_dict[param]))
        # end for

        # Iterate through parameter space
        return self._iterate_through_parameter_space(params_space)
    # end _generate_paramters_space

    # Generate param space
    def _generate_param_space(self, param, values):
        """
        Generate param sapce
        :param param:
        :param values:
        :return:
        """
        param_space = list()

        # For each values
        for value in values:
            param_space.append({param: value})
        # end for

        return param_space
    # end _generate_param_space

    ##########################################
    # Override
    ##########################################

    # Iterate
    def __iter__(self):
        """
        Iterate
        :return:
        """
        return self
    # end __iter__

    # Next element
    def next(self):
        """
        Next element
        :return:
        """
        # Stop
        if self._pos >= len(self._parameter_space):
            raise StopIteration
        else:
            space = self._parameter_space[self._pos]
            self._pos += 1
            return space
        # end if
    # end next

# end ParameterSpace
