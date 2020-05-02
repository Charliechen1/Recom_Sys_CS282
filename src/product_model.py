import torch
import torch.nn as nn
from product_utils import process_prod_batch


class ProductTower(nn.Module):
    def __init__(self, embed_model, embed_tokenizer, embed_dim, seq_len, lstm_hidden_dim, n_lstm,
                 prod2idx):
        super(ProductTower, self).__init__()
        self.lstm = nn.LSTM(embed_dim, lstm_hidden_dim, batch_first=True, num_layers=n_lstm)
        self.embed_model = embed_model
        self.embed_tokenizer = embed_tokenizer
        self.seq_len = seq_len

        self.prod2idx = prod2idx

    def forward(self, X):
        embedding, also_view, also_buy = process_prod_batch(X, self.embed_model, self.embed_tokenizer, self.seq_len, self.prod2idx)
        lstm_out, _ = self.lstm(embedding)

        return lstm_out
