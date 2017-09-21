# -*- coding: utf-8 -*-
#
# File : corpus/IQLACorpus.py
# Description : .
# Date : 16/08/2017
#
# Copyright Nils Schaetti, University of Neuchâtel <nils.schaetti@unine.ch>

# Imports
import math


# Iterate through a train and test dataset
# for a k-fold cross validation
class CrossValidation(object):
    """
    Iterate through a train and test dataset
    for a k-fold cross validation
    """

    # Constructor
    def __init__(self, authors, k=10):
        """
        Constructor
        """
        # Properties
        self._authors = authors
        self._k = k
        self._pos = 0
        self._n_texts = len(authors[0].get_texts())
        self._fold_size = math.floor(float(self._n_texts) / float(k))
    # end __init__

    #################################################
    # Override
    #################################################

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
        if self._pos >= self._k:
            raise StopIteration
        # end if

        # Sets
        train_set = list()
        test_set = list()

        # Total text indexes
        total_text_ids = range(self._n_texts)

        # Test indexes
        test_text_ids = total_text_ids[self._pos*self._fold_size:self._pos*self._fold_size+self._fold_size]

        # Remove test indexes
        train_text_ids = total_text_ids
        for idx in test_text_ids:
            train_text_ids.remove(idx)
        # end for

        # Get train set
        # Each authors
        for author in self._authors:
            # Texts
            author_texts = author.get_texts()

            # Each indexes
            for idx in train_set:
                train_set.append(author_texts[idx])
            # end for
        # end for

        # Get test set
        # Each authors
        for author in self._authors:
            # Texts
            author_texts = author.get_texts()

            # Each indexes
            for idx in test_set:
                test_set.append(author_texts[idx])
            # end for
        # end for

        # Next fold
        self._k += 1

        # Result
        return train_set, test_set
    # end next

# end CrossValidation
