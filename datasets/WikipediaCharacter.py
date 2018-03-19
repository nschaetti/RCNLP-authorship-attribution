#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

# Imports
import os
from torch.utils.data import Dataset


# Wikipedia Character prediction
class WikipediaCharacter(Dataset):
    """
    Wikipedia Character prediction
    """

    # Constructor
    def __init__(self, context_size, token_to_ix, root='./data'):
        """
        Constructor
        :param context_size:
        :param token_to_ix:
        """
        # Properties
        self.root = root
        self.context_size = context_size
        self.token_to_ix = token_to_ix

        # Load file list
        self.files = self._load()
    # end __init__

    ############################################
    # OVERRIDE
    ############################################

    # Length
    def __len__(self):
        """
        Length
        :return:
        """
        return len(self.files)
    # end __len__

    # Get item
    def __getitem__(self, item):
        """
        Get item
        :param item:
        :return:
        """
        pass
    # end __getitem__

    ############################################
    # PRIVATE
    ############################################

    # Load file list
    def _load(self):
        """
        Load file list
        :return:
        """
        return os.listdir(self.root)
    # end _load