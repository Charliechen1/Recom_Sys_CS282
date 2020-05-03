import torch
import torch.nn as nn
import torchfm.model.fm as fm


class ProductTower(nn.Module):
    def __init__(self, embed_model, embed_dim, rnn_hidden_dim, rnn_num_layers, rnn_type='GRU',
                 fm_type='fm', **kwargs):
        super(ProductTower, self).__init__()
        self.embed_model = embed_model
        if rnn_type == 'GRU':
            self.rnn = nn.GRU(embed_dim, rnn_hidden_dim, rnn_num_layers, batch_first=True)
        else:
            self.rnn = nn.LSTM(embed_dim, rnn_hidden_dim, batch_first=True, num_layers=rnn_num_layers)
        self.fm = fm.FactorizationMachineModel(kwargs['fm_field_dims'], kwargs['fm_embed_dim'])

    def forward(self, text, bop):
        embedding = self.embed_model(text)
        rnn_out, _ = self.rnn(embedding)
        fm_out = self.fm(bop)

        return rnn_out, fm_out
