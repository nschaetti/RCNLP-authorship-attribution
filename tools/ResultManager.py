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
from sklearn.utils.extmath import cartesian
import csv
import scipy


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
        self._xp_dir, self._obj_dir = self._create_directory()

        # Open the output file
        self._output_file = self._open_log_file(os.path.join(self._xp_dir, u"output.log"))

        # Write log header
        self._write_log_header()

        # Open CSV result file
        self._csv_results = self._create_csv_results(os.path.join(self._xp_dir, u"output.csv"))

        # Log
        if self._verbose:
            self._write_log(u"Starting experiment {}".format(name))
            self._write_log(u"Result matrix is of dimension {}".format(self._n_dim))
            self._write_log(u"Result matrix is of shape {}".format(self._result_matrix.shape))
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

        # Set
        self._result_matrix[tuple(element_pos)] = success_rate

        # Verbose
        if self._verbose:
            # Write fold success rate
            self._write_log(u"\t\t\t\tSuccess rate {}".format(success_rate))

            # Last fold?
            if self._fold + 1 == self._k:
                k_pos = element_pos
                k_pos[-1] = slice(None)
                self._write_log(u"\t\t\t{}-fold success rate {}".format(self._k, np.average(self._result_matrix[tuple(k_pos)])))

                # Last sample?
                if self._sample + 1 == self._n_samples:
                    n_pos = k_pos
                    n_pos[-2] = slice(None)
                    # Folds perfs
                    folds_perfs = np.average(self._result_matrix[tuple(n_pos)], axis=-1).flatten()

                    # Print
                    self._write_log(u"\t\t{} samples success rate {} +- {}".format(self._n_samples, np.average(folds_perfs), np.std(folds_perfs)))
                    self._write_log(u"\t\tMax sample success rate {}".format(np.max(folds_perfs)))
                # end if
            # end if
        # end if

        # Write in CSV
        self._write_csv_result(success_rate)
    # end add_result

    # Save results
    def save(self):
        """
        Save results
        :return:
        """
        # Save overall success rate
        self._write_log(u"\tOverall success rate: {}".format(np.average(self._result_matrix)))

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
    def save_object(self, name, obj, info=""):
        """
        Add object
        :param name: Object's name
        :param obj: Object
        :return:
        """
        # Write
        with open(os.path.join(self._obj_dir, name + u".p"), 'wb') as f:
            pickle.dump(obj=obj, file=f)
        # end with

        # Infos?
        if info != "":
            with open(os.path.join(self._obj_dir, name + u".txt"), 'w') as f:
                f.write(info + u"\n")
            # end with
        # end if
    # end add_object

    ###########################################
    # Private
    ###########################################

    # Write result in CSV
    def _write_csv_result(self, success_rate):
        """
        Write result in CSV
        :param sucess_rate:
        :return:
        """
        # Row list
        row_list = list()

        # First to last param
        for param_index in range(len(self._params_dict.keys())):
            # For each param
            for param in self._params_dict.keys():
                if param_index == self._param2dim[param]:
                    row_list.append(self._pos[param])
                # end if
            # end for
        # end for

        # Add sample and fold
        row_list.append(self._sample)
        row_list.append(self._fold)

        # Add result
        row_list.append(success_rate)

        # Write
        self._csv_results.writerow(row_list)
    # end _write_csv_result

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

        # Object directory
        self._obj_dir = os.path.join(self._xp_dir, u"objects")

        # Create if necessary
        if not os.path.exists(self._xp_dir):
            os.mkdir(self._xp_dir)
        # end if

        # Create if necessary
        if not os.path.exists(self._obj_dir):
            os.mkdir(self._obj_dir)
        # end if

        return self._xp_dir, self._obj_dir
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
        # Result by samples
        sample_results = np.average(self._result_matrix, axis=-1)

        # Save
        self._save_histogram(os.path.join(self._xp_dir, u"overall_results.png"), self._result_matrix.flatten(), u"Overall results", u"Result", u"Proportion")

        # Save result by samples
        self._save_histogram(os.path.join(self._xp_dir, u"samples_results.png"), sample_results.flatten(), u"Samples results", u"Result", u"Proportion")

        # Show max samples results
        max_result, max_std, max_params = self._get_max_parameters()
        self._write_log(u"\tBest perf with {} +- {} : {}".format(max_result, max_std, max_params))

        # Show overall samples results
        max_result, max_std, max_params = self._get_max_parameters(samples=False)
        self._write_log(u"\tBest perf with {} +- {} : {}".format(max_result, max_std, max_params))
    # end _save_global

    # Get max parameters
    def _get_max_parameters(self, samples=True):
        """
        Get parameters with maximum results
        :return:
        """
        # Max values
        max_result = 0
        max_pos = ()
        max_std = 0

        # Remove
        if samples:
            n_remove = 1
        else:
            n_remove = 2
        # end if

        # Array of positions
        pos_array = list()

        # For each dimension
        for dim in range(self._n_dim - n_remove):
            # Size of this dim
            pos_array.append(np.arange(0, self._result_matrix.shape[dim]))
        # end for

        # Cartesian product
        cart_product = cartesian(np.array(pos_array))

        # For each pos
        for pos in cart_product:
            # Position
            pos_tuple = pos.tolist()

            # Get result
            pos_result = np.average(self._result_matrix[tuple(pos_tuple)])

            # Max?
            if pos_result > max_result:
                max_result = pos_result
                max_pos = pos_tuple
                max_std = np.std(self._result_matrix[tuple(pos_tuple)])
            # end if
        # end for

        # Return max, std, and parameters
        return max_result, max_std, self._pos_to_dict(max_pos)
    # end _get_max_parameters

    # Position to dictionary
    def _pos_to_dict(self, position):
        """
        Position to dictionary
        :return:
        """
        # Max. pos
        max_pos = {}

        # For each dimension
        for index, pos in enumerate(position):
            # For each parameters
            for param in self._params_dict.keys():
                if index == self._param2dim[param]:
                    for param_value in self._params_dict[param]:
                        if pos == self._value2pos[param][param_value]:
                            max_pos[param] = param_value
                        # end if
                    # end for
                # end if
            # end for
        # end for

        # Sample
        if len(position) == self._n_dim-1:
            max_pos['samples'] = position[-1]
        # end if

        return max_pos
    # end _pos_to_dict

    # Save param data
    def _save_param_data(self, param):
        """
        Save param data
        :return:
        """
        # Create directory
        param_path = os.path.join(self._xp_dir, param)
        if not os.path.exists(param_path):
            os.mkdir(param_path)
        # end if

        # Param dimension
        dim = self._param2dim[param]

        # Possible values
        values = self._params_dict[param]

        # Number of values
        n_values = len(values)

        # Values samples
        value_samples = dict()
        value2sample = dict()

        # Plot value
        plot_results = np.array([])
        plot_std = np.array([])

        # Perf per values
        value2perf = dict()
        value2std = dict()

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

            # Samples perfs
            samples_results = np.average(samples, axis=-1).flatten()

            # Add to dict
            value_samples[value] = samples
            value_samples[value].shape = (self._n_samples, self._k)
            value2sample[value] = samples_results

            # Add to plot
            plot_results = np.append(plot_results, np.average(samples_results))
            plot_std = np.append(plot_std, np.std(samples_results))

            # Value to perf
            value2perf[value] = np.average(samples_results)
            value2std[value] = np.std(samples_results)
        # end for

        # Save the plot
        self._save_plot(os.path.join(param_path, u"plot.png"), values, plot_results, plot_std, u"Results vs {}".format(param), param, u"Results")

        # Write param CSV
        self._write_param_csv(os.path.join(param_path, u"samples.csv"), value2sample)

        # Write param tests
        self._write_param_tests(os.path.join(param_path, u"t-tests.csv"), value_samples)
    # end _save_param_data

    # Write param tests
    def _write_param_tests(self, filename, value2samples):
        """
        Write param tests
        :param filename:
        :param value2samples:
        :return:
        """
        # Values
        values = value2samples.keys()

        # Create CSV
        c = csv.writer(open(filename, 'wb'))

        # Write header
        c.writerow([u""] + values)

        # T-tests values
        t_tests = dict()

        # For each value
        for value1 in values:
            t_tests[value1] = dict()
            for value2 in values:
                if value1 != value2:
                    #print(value2samples[value1].flatten())
                    #print(value2samples[value2].flatten())
                    t_tests[value1][value2] = scipy.stats.ttest_rel(value2samples[value1].flatten(), value2samples[value2].flatten()).pvalue
                else:
                    t_tests[value1][value2] = 0.0
                # end if
            # end for
        # end for

        # For each value
        for value1 in values:
            ttest_row = [value1]
            for value2 in values:
                ttest_row.append(t_tests[value1][value2])
            # end for
            c.writerow(ttest_row)
        # end for
    # end _write_param_tests

    # Write param CSV
    def _write_param_csv(self, filename, value2samples):
        """
        Write param CSV
        :param filename:
        :param value2samples:
        :return:
        """
        # Values
        values = value2samples.keys()

        # Create CSV
        c = csv.writer(open(filename, 'wb'))

        # Write header
        c.writerow(values)

        # For each sample
        for index in range(len(value2samples[values[0]])):
            sample_row = list()
            for value in values:
                sample_row.append(value2samples[value][index])
            # end for
            c.writerow(sample_row)
        # end for
    # end _write_param_csv

    # Save plot
    def _save_plot(self, filename, x, y, data_std, title, xlabel, ylabel):
        """
        Save plot
        :param filename:
        :param data:
        :param title:
        :param x_label:
        :param y_label:
        :return:
        """
        plt.figure()
        plt.errorbar(x, y, yerr=data_std)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(filename)
        plt.close()
    # end _save_plot

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
        plt.savefig(filename)
        plt.close()
    # end _save_histogram

    # Create CSV file
    def _create_csv_results(self, filename):
        """
        Create CSV file
        :param filename:
        :return:
        """
        # Writer
        c = csv.writer(open(filename, 'wb'))

        # Row list
        row_list = list()

        # First to last param
        for param_index in range(len(self._params_dict.keys())):
            # For each param
            for param in self._params_dict.keys():
                if param_index == self._param2dim[param]:
                    row_list.append(param)
                # end if
            # end for
        # end for

        # Append samples and fold
        row_list.append(u"samples")
        row_list.append(u"fold")
        row_list.append(u"result")

        # Write header
        c.writerow(row_list)

        return c
    # end _create_csv_results

# end
