import numpy as np
import gensim
from gensim.scripts.glove2word2vec import glove2word2vec
from transformers import *
import torch
import torch.nn as nn
from torch.autograd import Variable
from conf import *


class Tokenizer:
    def __init__(self, vocab_table, method):
        self.vocab_table = vocab_table
        self.unk_map = {
            'glove': 'unk',
            'word2vec': 'unk',
        }
        self.unk_label = self.unk_map[method]

    def tokenize(self, text, max_len):
        tokens = list(gensim.utils.tokenize(text, lowercase=True))
        res = ['pad'] * max_len
        res[:min(max_len, len(tokens))] = tokens[:min(max_len, len(tokens))]
        return res

    def convert_tokens_to_ids(self, tokens):
        idxs = [self.vocab_table.get(token, self.vocab_table[self.unk_label]).index \
                for token in tokens]
        return idxs

    def encode(self, text, max_length, pad_to_max_length=True):
        tokens = self.tokenize(text, max_length)
        idxs = self.convert_tokens_to_ids(tokens)
        return idxs

def get_embed_layer(method = 'word2vec'):
    if method == 'glove':
        glove_input_file = 'data/glove_6B/glove.6B.100d.txt'
        word2vec_output_file = 'data/glove_6B/glove.6B.100d.txt.word2vec'
        glove2word2vec(glove_input_file, word2vec_output_file)
        glove_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
        embed_dim = 100
        lookup = torch.FloatTensor(glove_model.vectors)
        if to_gpu:
            lookup = lookup.cuda()
        embedding = nn.Embedding.from_pretrained(lookup, freeze=pretrain_freeze)
        tokenizer = Tokenizer(glove_model.vocab, method)
        return tokenizer, embedding, embed_dim

    elif method == 'word2vec':
        word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(
            'data/GoogleNews-vectors-negative300.bin',
            binary=True
        )
        embed_dim = 300
        lookup = torch.FloatTensor(word2vec_model.vectors)
        if to_gpu:
            lookup = lookup.cuda()
        embedding = nn.Embedding.from_pretrained(lookup, freeze=pretrain_freeze)
        tokenizer = Tokenizer(word2vec_model.vocab, method)
        return tokenizer, embedding, embed_dim

    elif method == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        embed_dim = 768

        if pretrain_freeze:
            model.eval()
        else:
            model.train()
        return tokenizer, model.embeddings.word_embeddings, embed_dim
