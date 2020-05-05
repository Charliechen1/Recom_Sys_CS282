import torch
import torch.nn as nn
from src.Layers import SelfAttnDSSM, ProductTower

class RecomModel(nn.Module):
    def __init__(self, rnn_hid_dim, d_inner,
                 n_head, pad_len, n_sen,
                 d_k, d_v,
                 embed_model,
                 lm_embed_dim,
                 n_rnn,
                 fm_field_dims,
                 fm_embed_dim,
                 rnn_type='GRU',
                 fm_type='fm',
                 dropout=0.1):

        super(RecomModel, self).__init__()

        self.pad_len = pad_len
        self.n_sen = n_sen
        self.rnn_hid_dim = rnn_hid_dim

        self.product_tower = ProductTower(embed_model, lm_embed_dim,
                                          rnn_hid_dim, n_rnn,
                                          fm_field_dims,
                                          fm_embed_dim,
                                          )
        #TODO add model for review side here


        self.dssm = SelfAttnDSSM(rnn_hid_dim, rnn_hid_dim,
                                 n_head, n_sen,
                                 d_k, d_v,
                                 dropout)


    def forward(self, review, rev_bop):
        bc_size = review.shape[0]
        query, fm_out = self.product_tower(review, rev_bop)

        # TODO: trivial document: for test purpose, replace by true review side model later
        document = [torch.zeros(bc_size, self.pad_len, self.rnn_hid_dim)\
                    for _ in range(self.n_sen)]

        dssm_out = self.dssm(query, document)

        return dssm_out

