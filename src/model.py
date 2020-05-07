import torch
import torch.nn as nn
import torch.nn.functional as F
from Layers import SelfAttnDSSM, DNNDSSM, ProductTower, ReviewTower
from conf import *

class RecomModel(nn.Module):
    def __init__(self, rnn_hid_dim, d_inner,
                 n_head, seq_len, n_sen,
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

        self.seq_len = seq_len
        self.n_sen = n_sen
        self.rnn_hid_dim = rnn_hid_dim
        self.act = nn.ReLU()

        self.product_tower = ProductTower(embed_model, lm_embed_dim,
                                          rnn_hid_dim, n_rnn,
                                          fm_field_dims,
                                          fm_embed_dim,
                                          )
        self.review_tower = ReviewTower(embed_model, lm_embed_dim,
                                        rnn_hid_dim, n_rnn, n_sen)

        self.dssm = SelfAttnDSSM(rnn_hid_dim * 2, rnn_hid_dim,
                                 n_head, n_sen, seq_len,
                                 d_k, d_v,
                                 dropout)
        #self.dssm = DNNDSSM(rnn_hid_dim * 2, seq_len, n_sen, dropout)
        self.cls_linear1 = nn.Linear(1 + rnn_hid_dim * 4, 1 + rnn_hid_dim * 4)
        self.cls_linear2 = nn.Linear(1 + rnn_hid_dim * 4, 1 + rnn_hid_dim * 4)
        self.cls_linear3 = nn.Linear(1 + rnn_hid_dim * 4, 2)
        

    def forward(self, product, prod_bop, reviews):
        bc_size = product.shape[0]
        query, fm_out = self.product_tower(product, prod_bop)
        
        document = self.review_tower(reviews)

        dssm_out = self.dssm(query, document)
        
        cat_tensor = torch.cat((dssm_out, fm_out.unsqueeze(1)), dim=1)
        res = self.act(self.cls_linear1(cat_tensor))
        res = self.act(self.cls_linear2(res))
        res = self.cls_linear3(res)
        return res

