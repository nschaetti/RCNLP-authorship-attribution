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


class PlotGenerator(object):

    # Constructor
    def __init__(self, title="", n_plots=1):

        # Variables
        self._sub_fig = []
        self._plot_count = 1
        self._n_plots = n_plots
        self._axes = []

        # Figure
        self._fig = plt.figure(figsize=(16, 13.5))

        # Subtitle
        self._fig.suptitle(title, fontsize=14, fontweight='bold')

    # end __init__

    # Add sub plot
    def add_sub_plot(self, title="", x_label="", y_label="", xlim=None, ylim=None):

        # Add subplot
        ax = self._fig.add_subplot((self._n_plots * 100) + 10 + self._plot_count)
        self._axes += [ax]

        # Title
        ax.set_title(title)

        # Axis labels
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        # Range X
        if xlim is not None:
            ax.set_xlim(xlim)

        # Range Y
        if ylim is not None:
            ax.set_ylim(ylim)

        # Next subplot
        self._plot_count += 1

    # end add_sub_plot

    # Plot
    def plot(self, y, x=[], label="", yerr=None, subplot=1, marker='', linestyle='-', color='b'):

        # Select subplot
        plt.subplot((self._n_plots * 100) + 10 + subplot)

        # Add plot
        if yerr is None:
            if len(x) == 0:
                plt.plot(y, label=label, marker=marker, linestyle=linestyle, color=color)
            else:
                plt.plot(x, y, label=label, marker=marker, linestyle=linestyle, color=color)
        else:
            if len(x) == 0:
                plt.errorbar(y, yerr=yerr, label=label, linestyle=linestyle, color=color)
            else:
                plt.errorbar(x, y, yerr=yerr, label=label, linestyle=linestyle, color=color)

        # Legend
        plt.legend(loc="best", ncol=2, shadow=False, fancybox=False)

    # end plot

    # Image
    def imshow(self, data, cmap=None):
        plt.imshow(data, cmap=cmap)
    # end imshow

    # Add horizontal line
    def add_hline(self, value, length, label="", subplot=1):

        # Select subplot
        plt.subplot((self._n_plots * 100) + 10 + subplot)

        # Repeat
        ar = np.repeat(value, length)

        # Plot
        plt.plot(ar, label=label, marker='', linestyle='--', color='g')

    # end add_hline

    # Save the figure in a PNG image
    def save_image(self, filename):

        # Adjust space
        plt.tight_layout()
        self._fig.subplots_adjust(top=0.93)

        # Save plot
        self._fig.savefig(filename)

        # Close plot
        plt.close()
    # end save_image

    # Show the current plot
    def show_plot(self):

        # Show
        plt.show()
    # end show_plot

# end PlotGenerator
