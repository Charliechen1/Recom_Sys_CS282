import numpy as np
from nltk import word_tokenize

class Embedding:
    
    def __init__(self):
        
        self.glove_embeddings = {}
        with open('../data/glove_6B/glove.6B.50d.txt') as f:
            for line in f:
                values = line.split(' ')
                self.glove_embeddings[values[0]] = np.array(values[1:], dtype = np.float32)
    
    def embed(self, text, method = 'glove'):
    
        if method == 'glove':
        
            tokens = word_tokenize(text.lower())
            return np.sum(self.glove_embeddings[token] for token in tokens)
    
    
            
        
