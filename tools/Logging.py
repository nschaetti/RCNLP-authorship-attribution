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

import os
import time
import subprocess


# Log the result of an experiment
class Logging(object):

    # Constructor
    def __init__(self, results_directoy="results", exp_name="noname", exp_inst="noname", exp_value=""):

        # Variables
        self._saved_variables = {}
        self._results_directory = results_directoy
        self._exp_name = exp_name
        self._exp_inst = exp_inst
        self._exp_value = exp_value
        if exp_value != "":
            self._path = os.path.join(".", self._results_directory, self._exp_name, self._exp_inst,
                                      self._exp_value + "_" + time.strftime("%d-%m-%Y_%H:%M:%S"))
        else:
            self._path = os.path.join(".", self._results_directory, self._exp_name, self._exp_inst,
                                      time.strftime("%d-%m-%Y_%H:%M:%S"))
        self._global_count = 0
        self._plot_count = 0

        # Create directories
        self._create_directories()
    # end __init__

    # Create the directories
    def _create_directories(self):

        # Results directory
        if not os.path.exists(os.path.join(".", self._results_directory)):
            os.makedirs(self._results_directory)

        # Experience directory
        if not os.path.exists(os.path.join(".", self._results_directory, self._exp_name)):
            os.makedirs(os.path.join(".", self._results_directory, self._exp_name))

        # Instance directory
        if not os.path.exists(os.path.join(".", self._results_directory, self._exp_name, self._exp_inst)):
            os.makedirs(os.path.join(".", self._results_directory, self._exp_name, self._exp_inst))

        # Date directory
        os.makedirs(os.path.join(".", self._path))

    # end create_directories

    # Save globals
    def save_globals(self):

        # Open file
        f = open(os.path.join(self._path, "globals." + str(self._global_count)), 'w')

        # Write globals
        var_globals = globals()
        for var in var_globals:
            f.write(var + " : " + str(var_globals[var]) + "\n\n")

        # Increment
        self._global_count += 1

    # end save_globals

    # Save globals
    def save_variables(self, variables):

        # Open file
        f = open(os.path.join(self._path, "vars"), 'a')

        # Write variables
        for var in variables:
            f.write(var + " : " + str(variables[var]) + "\n\n")

        # Close
        f.close()

    # end save_globals

    # Generate experience name
    @staticmethod
    def generate_experience_name(variables):
        name = ""
        # For each variables
        for var in variables:
            if var[0:3] == "rc_":
                name += var[3:] + "=" + str(variables[var]) + "_"
            # end if
        # end for
        return name
    # end generate_experience_name

    # Save a variable
    def save_variable(self, name, value):

        # Open file
        f = open(os.path.join(self._path, "vars"), 'a')

        # Write globals
        f.write(str({'name': name, 'value': value}) + "\n\n")

        # Close
        f.close()

    # end save_variable

    # Save a graph
    def save_plot(self, plot):

        # Save plot
        plot.save_image(os.path.join(self._path, "plot." + str(self._plot_count) + ".png"))

        # Next plot
        self._plot_count += 1

    # end save_graph

    # Save results
    def save_results(self, name, value, display=False):

        # Open file
        f = open(os.path.join(self._path, "results"), 'a')

        # Write globals
        f.write(name + " : " + str(value) + "\n\n")

        # Display if need
        if display:
            print('\033[91m' + name + " : " + str(value) + '\033[0m')
        # endif

        # Close
        f.close()

    # end save_results

    # Open loggin directory
    def open_dir(self):

        # Launch nautilus
        subprocess.Popen(['xdg-open', self._path])

    # end open_dir

# Logging
