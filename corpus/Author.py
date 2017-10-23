# -*- coding: utf-8 -*-
#
# File : corpus/IQLACorpus.py
# Description : .
# Date : 16/08/2017
#
# Copyright Nils Schaetti, University of Neuchâtel <nils.schaetti@unine.ch>

# Imports
import json
import os
from Text import Text


# Class to access to an author
class Author(object):
    """
    Class to access to an author
    """

    # Constructor
    def __init__(self, dataset_path, name):
        """
        Constructor
        """
        # Properties
        self._dataset_path = dataset_path
        self._name = name
        self._texts = list()

        # Load
        self._load()
    # end __init__

    ###########################################
    # Public
    ###########################################

    # Get name
    def get_name(self):
        """
        Get name
        :return:
        """
        return self._name
    # end get_name

    # Get texts
    def get_texts(self):
        """
        Get texts
        :return:
        """
        return self._texts
    # end get_texts

    # Get number of texts
    def get_n_texts(self):
        """
        Get number of texts
        :return:
        """
        return len(self._texts)
    # end get_n_texts

    ############################################
    # Override
    ############################################

    # To string
    def __unicode__(self):
        """
        To string
        :return:
        """
        return u"Author(name:{}, n_texts:{})".format(self._name, len(self._texts))
    # end __unicode__

    # To string
    def __str__(self):
        """
        To string
        :return:
        """
        return "Author(name:{}, n_texts:{})".format(self._name, len(self._texts))
    # end __unicode__

    ############################################
    # Override
    ############################################

    # Load author's texts
    def _load(self):
        """
        Load author's texts
        :return:
        """
        # Author info
        author_texts = json.load(open(os.path.join(self._dataset_path, "authors.json"), 'r'))[self._name]

        # For each texts
        for author_text in author_texts:
            self._texts.append(Text(text_path=os.path.join(self._dataset_path, author_text + u".txt"), author=self))
        # end for
    # end _load

# end Author
