#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

# Imports
import nsNLP
import sys


# Create tokenizer
def create_tokenizer(tokenizer_type):
    """
    Create tokenizer
    :param tokenizer_type: Tokenizer
    :return:
    """
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
