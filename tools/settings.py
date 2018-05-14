#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

# Global settings
n_authors = 15
k = 10

# Glove settings
glove_embedding_dim = 300

# CGFS settings
cgfs_epoch = 400
cgfs_input_dim = glove_embedding_dim
cgfs_mean = -4.56512329954
cgfs_std = 0.911449706065
cgfs_output_dim = {1: 30, 2: 60, 3: 90}
cgfs_lr = 0.001
cgfs_momentum = 0.9

# CCSAA Settings
ccsaa_epoch = 150
ccsaa_text_length = 20
ccsaa_voc_size = 84
ccsaa_output_dim = 150
ccsaa_mean = -4.56512329954
ccsaa_std = 0.911449706065
ccsaa_embedding_dim = 50
ccsaa_lr = 0.001
ccsaa_momentum = 0.9

