import torch
import torch.nn as nn
import torch.nn.functional as F
from Layers import SelfAttnDSSM, DNNDSSM, DocumentNet, QueryNet
from conf import *

class RecomModel(nn.Module):
    def __init__(self, rnn_hid_dim, d_inner,
                 n_head, seq_len, 
                 doc_n_sen, que_n_sen,
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
        self.n_rnn = n_rnn
        self.rnn_hid_dim = rnn_hid_dim
        self.act = nn.ReLU()
        
        self.doc_net = DocumentNet(embed_model, lm_embed_dim,
                                        rnn_hid_dim, n_rnn, doc_n_sen)
        
        self.que_net = DocumentNet(embed_model, lm_embed_dim,
                                        rnn_hid_dim, n_rnn, que_n_sen)
        
        if self.n_rnn:
            d_model = rnn_hid_dim * 2
        else:
            d_model = lm_embed_dim
            
        if dssm_type == 'self_attn_dssm':
            self.dssm = SelfAttnDSSM(d_model, rnn_hid_dim,
                                     n_head, 
                                     doc_n_sen, que_n_sen, 
                                     seq_len,
                                     d_k, d_v,
                                     dropout)
        elif dssm_type == 'dnn_dssm':
            self.dssm = DNNDSSM(d_model, seq_len, doc_n_sen, que_n_sen, dropout)
        self.cls_linear1 = nn.Linear(d_model, d_model)
        self.cls_linear2 = nn.Linear(d_model, d_model)
        self.cls_linear3 = nn.Linear(d_model, 1)
        

    def forward(self, doc, que):
        doc = self.doc_net(doc)
        que = self.que_net(que)
        
        dssm_out = self.dssm(doc, que)
        
        # feedforward part
        res = dssm_out + self.act(self.cls_linear1(dssm_out))
        res = res + self.act(self.cls_linear2(res))
        res = self.cls_linear3(res).squeeze()
        return res

