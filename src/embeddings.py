import numpy as np
from keras.preprocessing.sequence import pad_sequences
from nltk import word_tokenize
import gensim

import torch
import torch.nn as nn
from torch.autograd import Variable

MAX_SEQUENCE_LENGTH = 100

# Instead of return embedded vectors, return weights for embedding
class Embedding:
    
    def __init__(self, method='glove'):
        self.method = method
        if method == 'glove':
            self.glove_embeddings = {}
            with open('../data/glove_6B/glove.6B.50d.txt') as f:
                for line in f:
                    values = line.split(' ')
                    self.glove_embeddings[values[0]] = np.array(values[1:], dtype = np.float32)
            self.embed_dim = 50

        elif method == 'word2vec':
            self.word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
            self.embed_dim = 300

    def embed(self, text):
        tokens = word_tokenize(text.lower())
        tokens = pad_sequences(tokens, maxlen=MAX_SEQUENCE_LENGTH, padding="pre", truncating="post")
        result = np.array([])
        if self.method == 'glove':
            for token in tokens:
                result = np.vstack((result, self.glove_embeddings[token]))

        elif self.method == 'word2vec':
            for token in tokens:
                result = np.vstack((result, self.word2vec_model[token]))

        return torch.from_numpy(result)


class ReviewNN(nn.Module):
    def __init__(self, weights_matrix, hidden_size, num_layers):
        super(self).__init__()
        self.embedding = Embedding('word2vec')
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(self.embedding.embed_dim, hidden_size, num_layers, batch_first=True)

    def forward(self, reviews, hidden):
        return self.gru(self.embedding.embed(reviews), hidden)

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))
