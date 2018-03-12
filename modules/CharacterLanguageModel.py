#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CharacterLanguageModel(nn.Module):

    # Constructor
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(CharacterLanguageModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)
    # end __init__

    # Forward
    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs
    # end forward

# end CharacterLanguageModel
