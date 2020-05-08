import torch
import torch.nn as nn
import torch.nn.functional as F
from Layers import DotProdAttnDSSM, SimpleFC, CNNDSSM, DocumentNet, QueryNet, ConcatFF, ConcatFM
from conf import *

class RecomModel(nn.Module):
    def __init__(self, rnn_hid_dim, d_inner,
                 n_head, n_attn, 
                 seq_len, doc_n_sen, que_n_sen,
                 d_k, d_v,
                 embed_model,
                 lm_embed_dim,
                 n_rnn,
                 fm_embed_dim,
                 rnn_type='GRU',
                 ds_type='cff',
                 dropout=0.1):

        super(RecomModel, self).__init__()

        self.seq_len = seq_len
        self.n_rnn = n_rnn
        self.rnn_hid_dim = rnn_hid_dim
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        self.doc_net = DocumentNet(embed_model, lm_embed_dim,
                                        rnn_hid_dim, n_rnn, doc_n_sen)
        
        self.que_net = DocumentNet(embed_model, lm_embed_dim,
                                        rnn_hid_dim, n_rnn, que_n_sen)
        
        if self.n_rnn:
            d_model = rnn_hid_dim * 2
        else:
            d_model = lm_embed_dim
            
        if dssm_type == 'dot_prod_attn_dssm':
            self.dssm = DotProdAttnDSSM(d_model, rnn_hid_dim,
                                     n_head, n_attn,
                                     seq_len, doc_n_sen, que_n_sen, 
                                     d_k, d_v,
                                     dropout)
        elif dssm_type == 'simple_fc':
            self.dssm = SimpleFC(d_model, seq_len, doc_n_sen, que_n_sen, dropout)
        elif dssm_type == 'cnn_dssm':
            self.dssm = CNNDSSM(d_model, seq_len, doc_n_sen, que_n_sen, dropout=dropout)
        
        if ds_type == 'cff':
            self.downstream = ConcatFF(d_model, dropout)
        elif ds_type == 'fm':
            self.downstream = ConcatFM(d_model, fm_embed_dim, dropout)
        

    def forward(self, doc, que):
        doc = self.doc_net(doc)
        que = self.que_net(que)
        
        doc_emb, que_emb = self.dssm(doc, que)
        res = self.downstream(doc_emb, que_emb)
        return res

