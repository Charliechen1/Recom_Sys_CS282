import torch
import torch.nn as nn


class FM(nn.Module):  # taken from https://www.kaggle.com/gennadylaptev/factorization-machine-implemented-in-pytorch
    def __init__(self, n=None, k=None):
        super().__init__()
        # Initially we fill V with random values sampled from Gaussian distribution
        # NB: use nn.Parameter to compute gradients
        self.V = nn.Parameter(torch.randn(n, k), requires_grad=True)
        self.lin = nn.Linear(n, 1)

    def forward(self, x):
        out_1 = torch.matmul(x, self.V).pow(2).sum(1, keepdim=True)  # S_1^2
        out_2 = torch.matmul(x.pow(2), self.V.pow(2)).sum(1, keepdim=True)  # S_2

        out_inter = 0.5 * (out_1 - out_2)
        out_lin = self.lin(x)
        out = out_inter + out_lin

        return out


class ProductTower(nn.Module):
    def __init__(self, embed_model, embed_dim, rnn_hidden_dim, rnn_num_layers,
                 fm_n, fm_k,
                 rnn_type='GRU'):
        super(ProductTower, self).__init__()
        self.embed_model = embed_model
        if rnn_type == 'GRU':
            self.rnn = nn.GRU(embed_dim, rnn_hidden_dim, rnn_num_layers, batch_first=True)
        else:
            self.rnn = nn.LSTM(embed_dim, rnn_hidden_dim, batch_first=True, num_layers=rnn_num_layers)
        self.fm = FM(fm_n, fm_k)

    def forward(self, text, bop):
        embedding = self.embed_model(text)
        rnn_out, _ = self.rnn(embedding)
        fm_out = self.fm(bop)

        return rnn_out, fm_out
