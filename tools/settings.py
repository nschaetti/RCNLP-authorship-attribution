#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

# Global settings
n_authors = 15
n_pretrain_authors = 50 - n_authors
k = 10

# Authors
authors = [u'JanLopatka', u'WilliamKazer', u'MarcelMichelson', u'KirstinRidley', u'GrahamEarnshaw', u'MichaelConnor',
           u'MartinWolk', u'ToddNissen', u'PatriciaCommins', u'KevinMorrison', u'HeatherScoffield', u'BradDorfman',
           u'DavidLawder', u'KevinDrawbaugh', u'LynnleyBrowning', u'ScottHillis', u'FumikoFujisaki', u'TimFarrand',
           u'SarahDavison', u'AaronPressman', u'JohnMastrini', u'NickLouth', u'PierreTran', u'AlexanderSmith',
           u'MatthewBunce', u'KouroshKarimkhany', u'JimGilchrist', u'DarrenSchuettler', u'TanEeLyn', u'JoeOrtiz',
           u'MureDickie', u'EdnaFernandes', u'JoWinterbottom', u'RogerFillion', u'BenjaminKangLim', u"LynneO'Donnell",
           u'JonathanBirt', u'BernardHickey', u'RobinSidel', u'AlanCrosby', u'LydiaZajc', u'PeterHumphrey',
           u'KeithWeir', u'EricAuchard', u'TheresePoletti', u'KarlPenhaul', u'SimonCowell', u'JaneMacartney',
           u'SamuelPerry', u'MarkBendeich']
train_authors = [u'JanLopatka', u'WilliamKazer', u'MarcelMichelson', u'KirstinRidley', u'GrahamEarnshaw',
                 u'MichaelConnor', u'MartinWolk', u'ToddNissen', u'PatriciaCommins', u'KevinMorrison',
                 u'HeatherScoffield', u'BradDorfman', u'DavidLawder', u'KevinDrawbaugh', u'LynnleyBrowning']
pretrain_authors = [a for a in authors if a not in train_authors]

# Glove settings
glove_embedding_dim = 300

# CGFS settings
cgfs_epoch = 230
cgfs_input_dim = glove_embedding_dim
cgfs_mean = -4.56512329954
cgfs_std = 0.911449706065
# cgfs_output_dim = {'c1': 30, 'c2': 60, 'c3': 90}
# cgfs_output_dim_num = {1: 30, 2: 60, 3: 90}
cgfs_output_dim = {'c1': 30, 'c2': 40, 'c3': 50}
cgfs_output_dim_num = {1: 30, 2: 40, 3: 50}
cgfs_lr = 0.001
cgfs_momentum = 0.9
cgfs_files = 5
cgfs_batch_size = 64

# Sense2Vec settings
s2v_path = "/home/schaetti/Projets/TURING/Datasets/sense2vec"
s2v_embedding_dim = 128

# CCSAA Settings
ccsaa_epoch = 150
ccsaa_text_length = 20
ccsaa_voc_size = 84
ccsaa_pretrain_voc_size = 90
ccsaa_output_dim = 50
ccsaa_mean = -4.56512329954
ccsaa_std = 0.911449706065
ccsaa_embedding_dim = 50
ccsaa_lr = 0.001
ccsaa_momentum = 0.9
ccsaa_files = 5
ccsaa_batch_size = 128

# wEnc settings
wenc_epoch = 130
wenc_input_dim = glove_embedding_dim
wenc_output_dim = 450
wenc_output_dim_num = 450
wenc_lr = 0.001
wenc_momentum = 0.9
wenc_files = 5
wenc_batch_size = 512

# cEnc Settings
cenc_epoch = 150
cenc_text_length = 20
cenc_voc_size = 84
cenc_pretrain_voc_size = 90
cenc_output_dim = 300
cenc_embedding_dim = 300
cenc_lr = 0.001
cenc_momentum = 0.9
cenc_files = 5
cenc_batch_size = 128

# Word encoder settings
wordencoder_epoch = 400
wordencoder_input_dim = glove_embedding_dim
wordencoder_output_dim = 900
wordencoder_output_dim_num = 900
wordencoder_lr = 0.05
wordencoder_momentum = 0.9
wordencoder_batch_size = 512

# character encoder settings
charencoder_epoch = 150
charencoder_text_length = 20
charencoder_voc_size = 84
charencoder_pretrain_voc_size = 90
charencoder_lr = 0.05
charencoder_momentum = 0.9
charencoder_files = 5
charencoder_batch_size = 128

