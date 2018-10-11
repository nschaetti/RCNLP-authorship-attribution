#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

# Imports
import nsNLP
import sys
import torchlanguage.transforms
import os
import torch
from ccsaa_selector import load_ccsaa, load_ccsaa35
from cgfs_selector import load_cgfs, load_cgfs35
from wenc_selector import load_wenc
from cenc_selector import load_cenc
from wae_encoder import load_WAE
from chae_encoder import load_CHAE
import settings


# Create tokenizer
def create_tokenizer(tokenizer_type, lang="en_core_web_lg"):
    """
    Create tokenizer
    :param tokenizer_type: Tokenizer
    :return:
    """
    # Tokenizer
    if tokenizer_type == "nltk":
        tokenizer = nsNLP.tokenization.NLTKTokenizer()
    elif tokenizer_type == "nltk-twitter":
        tokenizer = nsNLP.tokenization.NLTKTweetTokenizer()
    elif tokenizer_type == "spacy":
        tokenizer = nsNLP.tokenization.SpacyTokenizer(lang=lang)
    elif tokenizer_type == "spacy_wv":
        tokenizer = nsNLP.tokenization.SpacyTokenizer(lang=lang, original=True)
    else:
        sys.stderr.write(u"Unknown tokenizer type!\n")
        exit()
    # end if

    # Return tokenizer object
    return tokenizer
# end create_tokenizer


# Create transformer
def create_transformer(feature, embedding="", path="", lang="en_vectors_web_lg", fold=0, use_cuda=False, dataset_size=100, dataset_start=0):
    """
    Create the transformer
    :param feature:
    :param embedding:
    :param path:
    :param lang:
    :param n_gram:
    :return:
    """
    # ## Part-Of-Speech
    if "pos" in feature:
        return torchlanguage.transforms.Compose([
            torchlanguage.transforms.PartOfSpeech(model=lang),
            torchlanguage.transforms.ToIndex(),
            torchlanguage.transforms.ToOneHot(voc_size=16)
        ])
    # ## Function Words Embedding
    elif "fwv" in feature:
        return torchlanguage.transforms.Compose([
            torchlanguage.transforms.FunctionWord(model=lang, join=True),
            torchlanguage.transforms.GloveVector(model=lang),
            torchlanguage.transforms.Reshape((-1, 300))
        ])
    # ## Function words
    elif "fw" in feature:
        return torchlanguage.transforms.Compose([
            torchlanguage.transforms.FunctionWord(model=lang),
            torchlanguage.transforms.ToIndex(),
            torchlanguage.transforms.ToOneHot(voc_size=300)
        ])
    # ## Tag
    elif "tag" in feature:
        return torchlanguage.transforms.Compose([
            torchlanguage.transforms.Tag(model=lang),
            torchlanguage.transforms.ToIndex(),
            torchlanguage.transforms.ToOneHot(voc_size=45)
        ])
    # ## Word Vector bigram
    elif "wv2" in feature:
        return torchlanguage.transforms.Compose([
            torchlanguage.transforms.GloveVector(model=lang),
            torchlanguage.transforms.ToNGram(n=2, overlapse=False),
            torchlanguage.transforms.Reshape((-1, 600))
        ])
    # ## Word Vector
    elif "wv" in feature:
        if embedding == 'glove':
            return torchlanguage.transforms.Compose([
                torchlanguage.transforms.GloveVector(model=lang),
                torchlanguage.transforms.Reshape((-1, 300))
            ])
        elif embedding == 'word2vec':
            return torchlanguage.transforms.Compose([
                torchlanguage.transforms.Token(model=lang),
                torchlanguage.transforms.GensimModel(
                    model_path=os.path.join(path, embedding, "embedding.en.bin")
                )
            ])
        elif embedding == 'fasttext':
            return torchlanguage.transforms.Compose([
                torchlanguage.transforms.Token(model=lang),
                torchlanguage.transforms.GensimModel(
                    model_path=os.path.join(path, embedding, "embedding.en.vec")
                )
            ])
        # end if
    # Sense2Vec
    elif "s2v" in feature:
        return torchlanguage.transforms.Compose([
            torchlanguage.transforms.Sense2Vec(path=settings.s2v_path),
            torchlanguage.transforms.Reshape((-1, settings.s2v_embedding_dim))
        ])
    # ## Character embedding
    elif "c1" in feature:
        token_to_ix, embedding_weights = load_character_embedding(path)
        embedding_dim = embedding_weights.size(1)
        return torchlanguage.transforms.Compose([
            torchlanguage.transforms.Character(),
            torchlanguage.transforms.ToIndex(token_to_ix=token_to_ix),
            torchlanguage.transforms.Embedding(weights=embedding_weights, voc_size=len(token_to_ix)),
            torchlanguage.transforms.Reshape((-1, embedding_dim))
        ])
    # ## Character 2-gram embedding
    elif "c2" in feature:
        token_to_ix, embedding_weights = load_character_embedding(path)
        embedding_dim = embedding_weights.size(1)
        return torchlanguage.transforms.Compose([
            torchlanguage.transforms.Character2Gram(overlapse=True),
            torchlanguage.transforms.ToIndex(token_to_ix=token_to_ix),
            torchlanguage.transforms.Embedding(weights=embedding_weights, voc_size=len(token_to_ix)),
            torchlanguage.transforms.Reshape((-1, embedding_dim))
        ])
    # ## Character 3-gram embedding
    elif "c3" in feature:
        token_to_ix, embedding_weights = load_character_embedding(path)
        embedding_dim = embedding_weights.size(1)
        return torchlanguage.transforms.Compose([
            torchlanguage.transforms.Character3Gram(overlapse=True),
            torchlanguage.transforms.ToIndex(token_to_ix=token_to_ix),
            torchlanguage.transforms.Embedding(weights=embedding_weights, voc_size=len(token_to_ix)),
            torchlanguage.transforms.Reshape((-1, embedding_dim))
        ])
    # ## CNN Glove Feature Selector
    elif "cgfs35" in feature:
        _, transformer = load_cgfs35(use_cuda=use_cuda)
        return transformer
    elif "cgfs" in feature:
        _, transformer = load_cgfs(fold, dataset_size=100, dataset_start=0, use_cuda=use_cuda)
        return transformer
    # ## CNN Character Selector for Authorship Attribution
    elif "ccsaa35" in feature:
        _, transformer = load_ccsaa35(use_cuda=use_cuda)
        return transformer
    elif "ccsaa" in feature:
        _, transformer = load_ccsaa(fold, dataset_size=100, dataset_start=0, use_cuda=use_cuda)
        return transformer
    # Wenc
    elif "wenc" in feature:
        _, transformer = load_wenc(fold, dataset_size=100, dataset_start=0, use_cuda=use_cuda)
        return transformer
    # Cenc
    elif "cenc" in feature:
        _, transformer = load_cenc(fold, dataset_size=dataset_size, dataset_start=dataset_start, use_cuda=use_cuda)
        return transformer
    # CHAE
    elif "CHAE" in feature:
        _, transformer = load_CHAE(use_cuda=use_cuda)
        return transformer
    # CHAE
    elif "CEAE" in feature:
        _, transformer = load_CHAE(path, sub_dir="CEAE", use_cuda=use_cuda)
        return transformer
    # CHAE
    elif "WAE450" in feature:
        _, transformer = load_WAE(sub_dir="WAE450", use_cuda=use_cuda)
        return transformer
    # WAE
    elif "WAE" in feature:
        _, transformer = load_WAE(use_cuda=use_cuda)
        return transformer
    else:
        raise NotImplementedError(u"Feature type {} not implemented".format(feature))
    # end if
# end create_transformer


# Load character embedding
def load_character_embedding(emb_path):
    """
    Load character embedding
    :param emb_path:
    :return:
    """
    token_to_ix, weights = torch.load(open(emb_path, 'rb'))
    return token_to_ix, weights
# end load_character_embedding
