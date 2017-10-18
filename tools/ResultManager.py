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
import os
import codecs
import pickle
import datetime
import matplotlib.pyplot as plt
import numpy as np


# Manage and save results
class ResultManager(object):
    """
    Manage and save results
    """

    # Constructor
    def __init__(self, output_dir, name, description, params_dict, n_samples, k=10, verbose=False):
        """
        Constructor
        :param params_dict:
        """
        # Properties
        self._output_dir = output_dir
        self._name = name
        self._description = description
        self._params_dict = params_dict
        self._n_samples = n_samples
        self._k = k
        self._fold = 0
        self._sample = 0
        self._n_dim = len(params_dict.keys()) + 2
        self._verbose = verbose
        self._objects = list()

        # Param to dimension
        self._param2dim = dict()
        self._value2pos = dict()
        for index, param in enumerate(params_dict.keys()):
            self._param2dim[param] = index
            self._value2pos[param] = dict()
            for index2, value in enumerate(params_dict[param]):
                self._value2pos[param][value] = index2
            # end for
        # end for

        # Current parameter value
        self._pos = dict()
        for param in params_dict.keys():
            self._pos[param] = params_dict[param][0]
        # end for

        # Generate result matrix
        self._result_matrix = self._generate_matrix()

        # Create directory
        self._xp_dir = self._create_directory()

        # Open the output file
        self._output_file = self._open_log_file(os.path.join(self._xp_dir, u"output.log"))

        # Write log header
        self._write_log_header()

        # Log
        if self._verbose:
            self._write_log(u"Starting experiment {}".format(name))
            self._write_log(u"Result matrix is of dimension {}".format(self._n_dim))
        # end if
    # end __init__

    ###########################################
    # Public
    ###########################################

    # Change parameter state
    def set_state(self, pos):
        """
        Change parameter state
        :param param:
        :param value:
        :return:
        """
        if self._verbose:
            self._write_log(u"\tChanging param state to {}".format(pos))
        # end if

        # Params
        for param in pos.keys():
            self._pos[param] = pos[param]
        # end for
    # end set_state

    # Change sample state
    def set_sample_state(self, n_sample):
        """
        Change sample state
        :param n_sample:
        :return:
        """
        if self._verbose:
            self._write_log(u"\t\tChanging sample state to {}".format(n_sample))
        # end if

        self._sample = n_sample
    # end set_sample_state

    # Change fold state
    def set_fold_state(self, k):
        """
        Change fold state
        :param k:
        :return:
        """
        if self._verbose:
            self._write_log(u"\t\t\tChanging fold state to {}".format(k))
        # end if

        self._fold = k
    # end set_fold_state

    # Save result
    def add_result(self, success_rate):
        """
        Save result
        :param success_rate:
        :return:
        """
        # Element pos
        element_pos = [0] * self._n_dim

        # For each param
        for param in self._param2dim.keys():
            # Dim
            dim = self._param2dim[param]

            # Pos of value
            pos = self._value2pos[param][self._pos[param]]

            # Set
            element_pos[dim] = pos
        # end for

        # Sample
        element_pos[-2] = self._sample

        # Fold
        element_pos[-1] = self._fold

        # Verbose
        if self._verbose:
            self._write_log(u"\t\t\t\tSuccess rate {}".format(success_rate))
        # end if

        # Set
        self._result_matrix[tuple(element_pos)] = success_rate
    # end add_result

    # Save results
    def save(self):
        """
        Save results
        :return:
        """
        # Save overall success rate
        self._write_log(u"Overall success rate: {}".format(np.average(self._result_matrix)))

        # Save result matrix
        self.save_object(u"result_matrix", self._result_matrix)

        # Save global data
        self._save_global()

        # For each param
        for param in self._params_dict.keys():
            # If there is more than
            # one value.
            if len(self._params_dict[param]) > 1:
                self._save_param_data(param)
            # end if
        # end for
    # end save

    # Save object
    def save_object(self, name, obj):
        """
        Add object
        :param name: Object's name
        :param obj: Object
        :return:
        """
        # Write
        with open(os.path.join(self._xp_dir, name + u".p"), 'wb') as f:
            pickle.dump(obj=obj, file=f)
        # end with
    # end add_object

    ###########################################
    # Private
    ###########################################

    # Write log header
    def _write_log_header(self):
        """
        Write log header
        :return:
        """
        self._write_log(u"Experience name : {}".format(self._name))
        self._write_log(u"Description : {}".format(self._description))
        self._write_log(u"Date : {}".format(datetime.datetime.utcnow()))
    # end _write_log_header

    # Create directory
    def _create_directory(self):
        """
        Create the experience directory
        :return:
        """
        # XP directory
        self._xp_dir = os.path.join(self._output_dir, self._name)

        # Create if necessary
        if not os.path.exists(self._xp_dir):
            os.mkdir(self._xp_dir)
        # end if

        return self._xp_dir
    # end _create_directory

    # Write log
    def _write_log(self, text):
        """
        Write log
        :param text:
        :return:
        """
        print(text)
        self._output_file.write(text + u"\n")
    # end _write_log

    # Open the output log file
    def _open_log_file(self, filename):
        """
        Open the output log file
        :param filename:
        :return:
        """
        return codecs.open(filename, 'w', encoding='utf-8')
    # end _open_log_file

    # Save global data
    def _save_global(self):
        """
        Save global data
        :return:
        """
        # Save
        self._save_histogram(u"overall_results", self._result_matrix.flatten(), u"Overall results", u"Success rate", u"Proportion")
    # end _save_global

    # Save param data
    def _save_param_data(self, param):
        """
        Save param data
        :return:
        """
        # Create directory
        os.mkdir(os.path.join(self._xp_dir, param))

        # Param dimension
        dim = self._param2dim[param]

        # Possible values
        values = self._params_dict[param]

        # Number of values
        n_values = len(values)

        # All samples
        all_samples = np.array([])

        # Values samples
        value_samples = dict()

        # Sample per values
        for value in values:
            # All range
            position_vector = [slice(None)] * self._n_dim

            # Value position
            value_pos = self._value2pos[param][value]

            # Set index
            position_vector[dim] = value_pos

            # Samples
            samples = self._result_matrix[tuple(position_vector)]

            # Add to dict
            value_samples[value] = samples

            # Add to all samples
            all_samples = np.append(all_samples, samples)
        # end for
    # end _save_param_data

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

    # Save histogram
    def _save_histogram(self, filename, data, title, xlabel=u"", ylabel=u""):
        """
        Save histogram
        :param data:
        :return:
        """
        plt.hist(data)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(filename + u".png")
    # end _save_histogram

# end
