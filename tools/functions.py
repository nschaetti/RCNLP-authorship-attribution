#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

# Imports
import nsNLP
import sys


# Create tokenizer
def create_tokenizer(tokenizer_type, converter_type):
    """
    Create tokenizer
    :param tokenizer: Tokenizer
    :return:
    """
    # Check converter
    if (converter_type == "wv" or converter_type == "pos" or converter_type == "tag") and tokenizer_type != "spacy_wv":
        sys.stderr.write(u"Only Spacy tokenizer is possible for word vector, tag or POS!\n")
        exit()
    # end if

    # Tokenizer
    if tokenizer_type == "nltk":
        tokenizer = nsNLP.tokenization.NLTKTokenizer()
    elif tokenizer_type == "spacy":
        tokenizer = nsNLP.tokenization.SpacyTokenizer()
    elif tokenizer_type == "spacy_wv":
        tokenizer = nsNLP.tokenization.SpacyTokenizer(original=True)
    else:
        sys.stderr.write(u"Unknown tokenizer type!\n")
        exit()
    # end if

    # Return tokenizer object
    return tokenizer
# end create_tokenizer
