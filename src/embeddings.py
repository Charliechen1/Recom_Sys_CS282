import numpy as np
from nltk import word_tokenize
import gensim


class Embedding:
    
    def __init__(self):
        
        self.glove_embeddings = {}
        with open('../data/glove_6B/glove.6B.50d.txt') as f:
            for line in f:
                values = line.split(' ')
                self.glove_embeddings[values[0]] = np.array(values[1:], dtype = np.float32)

        self.word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

    def embed(self, text, method = 'glove'):

        tokens = word_tokenize(text.lower())
        if method == 'glove':
            return np.sum(self.glove_embeddings[token] for token in tokens)
    
        elif method == 'word2vec':
            return np.sum(self.word2vec_model[token] for token in tokens)
            
        
