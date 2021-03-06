# -*- coding: utf-8 -*-
#

# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F


# CNN Feature Selector
class CNN2DFeatureSelector(nn.Module):
    """
    CNN Feature Selector
    """

    # Constructor
    def __init__(self, n_gram, n_authors=15, embedding_dim=300, out_channels=(20, 20, 20), kernel_sizes=(3, 5, 10), n_features=30):
        """
        Constructor
        :param n_authors:
        :param embedding_dim: Embedding layer dimension
        :param out_channels: Number of output channels
        :param kernel_sizes: Different kernel sizes
        """
        super(CNN2DFeatureSelector, self).__init__()
        self.n_features = n_features
        self.n_authors = n_authors

        # Conv window 1
        self.conv_w1 = nn.Conv2d(in_channels=1, out_channels=out_channels[0], kernel_size=(n_gram, kernel_sizes[0]))

        # Conv window 2
        self.conv_w2 = nn.Conv2d(in_channels=1, out_channels=out_channels[1], kernel_size=(n_gram, kernel_sizes[1]))

        # Conv window 3
        self.conv_w3 = nn.Conv2d(in_channels=1, out_channels=out_channels[2], kernel_size=(n_gram, kernel_sizes[2]))

        # Max pooling layer
        self.max_pool_w1 = nn.MaxPool1d(kernel_size=embedding_dim - out_channels[0] + 1, stride=0)
        self.max_pool_w2 = nn.MaxPool1d(kernel_size=embedding_dim - out_channels[1] + 1, stride=0)
        self.max_pool_w3 = nn.MaxPool1d(kernel_size=embedding_dim - out_channels[2] + 1, stride=0)

        # Linear layer 1
        self.linear_size = out_channels[0] + out_channels[1] + out_channels[2]
        self.linear = nn.Linear(self.linear_size, n_features)

        # Linear layer 2
        self.linear2 = nn.Linear(self.n_features, n_authors)
    # end __init__

    # Forward
    def forward(self, x):
        """
        Forward
        :param x:
        :return:
        """
        # Conv window
        out_win1 = F.relu(self.conv_w1(x))
        out_win2 = F.relu(self.conv_w2(x))
        out_win3 = F.relu(self.conv_w3(x))

        # Max pooling
        max_win1 = self.max_pool_w1(out_win1)
        max_win2 = self.max_pool_w2(out_win2)
        max_win3 = self.max_pool_w3(out_win3)

        # Concatenate
        out = torch.cat((max_win1, max_win2, max_win3), dim=1)

        # Flatten
        out = out.view(-1, self.linear_size)

        # Linear 1
        out = F.relu(self.linear(out))

        # Linear 2
        out = F.relu(self.linear2(out))

        # Log softmax
        log_prob = F.log_softmax(out, dim=1)

        # Log Softmax
        return log_prob
    # end forward

# end CNN2DFeatureSelector