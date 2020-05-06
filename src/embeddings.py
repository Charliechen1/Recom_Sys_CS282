import numpy as np
import gensim
from gensim.scripts.glove2word2vec import glove2word2vec
from transformers import *
import torch
import torch.nn as nn
from torch.autograd import Variable

MAX_SEQUENCE_LENGTH = 100

class Tokenizer:
    def __init__(self, vocab_table):
        self.vocab_table = vocab_table

    def tokenize(self, text):
        tokens = gensim.utils.tokenize(text, lowercase=True)
        res = ['pad'] * MAX_SEQUENCE_LENGTH
        res[:min(MAX_SEQUENCE_LENGTH, len(tokens))] = tokens[:min(MAX_SEQUENCE_LENGTH, len(tokens))]
        return res

    def convert_tokens_to_ids(self, tokens):
        idxs = [self.vocab_table[token].index for token in tokens]
        return idxs


def get_embed_layer(method = 'word2vec'):
    if method == 'glove':
        glove_input_file = '../data/glove.6B/glove.6B.100d.txt'
        word2vec_output_file = '../data/glove.6B/glove.6B.100d.txt.word2vec'
        glove2word2vec(glove_input_file, word2vec_output_file)
        glove_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
        embed_dim = 100
        embedding = nn.Embedding.from_pretrained(torch.FloatTensor(glove_model.vectors))
        tokenizer = Tokenizer(glove_model.vocab)
        return tokenizer, embedding, embed_dim

    elif method == 'word2vec':
        word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('../data/GoogleNews-vectors-negative300.bin',
                                                                              binary=True)
        embed_dim = 300
        embedding = nn.Embedding.from_pretrained(torch.FloatTensor(word2vec_model.vectors))
        tokenizer = Tokenizer(word2vec_model.vocab)
        return tokenizer, embedding, embed_dim


    elif method == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        embed_dim = 768
        model.eval()
        return tokenizer, model.embeddings.word_embeddings, embed_dim




