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
from Author import Author


# Class to access to the corpus
class Corpus(object):
    """
    Class to access to the corpus
    """

    # Constructor
    def __init__(self, dataset_path):
        """
        Constructor
        :param dataset_path: Path to dataset
        """
        # Properties
        self._dataset_path = dataset_path
        self._authors = list()
        self._texts = list()

        # Load dataset
        self._load()
    # end __init__

    ########################################
    # Public
    ########################################

    # Get list of authors
    def get_authors(self):
        """
        Get list of authors
        :return:
        """
        return self._authors
    # end authors

    # Get author
    def get_author(self, author_name):
        """
        Get author
        :param author_name:
        :return:
        """
        for author in self._authors:
            if author.get_name() == author_name:
                return author
            # end if
        # end for
        return None
    # end get_author

    # Get authors list
    def get_authors_list(self):
        """
        Get authors list
        :return:
        """
        result = list()
        for author in self._authors:
            result.append(author.get_name())
        # end for
        return result
    # end get_authors_list

    # Get the number of authors
    def get_n_authors(self):
        """
        Get the number of authors
        :return:
        """
        return len(self._authors)
    # end get_n_authors

    # Get the number of texts
    def get_n_texts(self):
        """
        Get the number of texts
        :return:
        """
        return len(self._texts)
    # end get_n_texts

    # Get texts
    def get_texts(self):
        """
        Get texts
        :return:
        """
        return self._texts
    # end get_texts

    # Get author's texts
    def get_author_texts(self, author_name):
        """
        Get author's texts
        :param author_name:
        :return:
        """
        return self.get_author(author_name).get_texts()
    # end get_author_texts

    ########################################
    # Override
    ########################################

    # Iterator
    def __iter__(self):
        """
        Iterator
        :return:
        """
        return self
    # end __iter__

    # Next
    def next(self):
        """
        Next
        :return:
        """
        pass
    # end next

    ########################################
    # Private
    ########################################

    # Load
    def _load(self):
        """
        Load
        :return
        """
        authors_infos = json.load(open(os.path.join(self._dataset_path), 'r'))
        for author_name in authors_infos.keys():
            # New author
            author = Author(name=author_name, dataset_path=self._dataset_path)

            # Add texts
            for text in author.get_texts():
                self._texts.append(text)
            # end for

            # Add
            self._authors.append(author)
        # end for
    # end _load

# end Corpus
