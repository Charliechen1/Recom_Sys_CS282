import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Sublayers import EncoderLayer, DecoderLayer, ReviewGRU
from embeddings import get_embed_layer
import torchfm.model.fm as fm

class DNNDSSM(nn.Module):
    def __init__(self, d_model, seq_len, n_sen, dropout=0.1):
        super(DNNDSSM, self).__init__()
        self.dropout = dropout
        self.act = nn.ReLU()
        self.doc_sen_linear = nn.Linear(n_sen, 1)
        self.doclinear = nn.Linear(seq_len, 1)
        self.quelinear = nn.Linear(seq_len, 1)
        self.catlinear1 = nn.Linear(d_model * 2, d_model * 2)
        self.catlinear2 = nn.Linear(d_model * 2, d_model * 2)
        
    def forward(self, query, document):
        query = query.squeeze().transpose(1, 2)
        document = torch.stack(document).transpose(0, 3)
        document = self.doc_sen_linear(document).squeeze()
        document = document.transpose(0, 1)
        
        query = self.act(self.quelinear(query)).squeeze()
        document = self.act(self.doclinear(document)).squeeze()
        
        res = torch.cat((query, document), dim=1)
        res = res + self.act(self.catlinear1(res))
        res = res + self.act(self.catlinear2(res))
        
        return res
    
class SelfAttnDSSM(nn.Module):
    """
    Idea from paper: "A Hierarchical Attention Retrieval Model for Healthcare Question Answering"
    """
    def __init__(self, d_model, d_inner, n_head, n_sen, seq_len, d_k, d_v, dropout=0.1):
        super(SelfAttnDSSM, self).__init__()
        self.query_self_attn = EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout)

        self.cros_attn_blocks = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_sen)
        ])

        self.doc_self_attn_blocks = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_sen)
        ])
        
        self.doc_linear_blocks = nn.ModuleList([
            nn.Linear(seq_len, 1) for _ in range(n_sen)
        ])

        self.doc_high_self_attn = EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout)
        
        self.doc_linear = nn.Linear(n_sen, 1)
        self.que_linear = nn.Linear(seq_len, 1)

    def forward(self, query, document):

        # query is the embedding for product title, with size (batch_size, seq_len, emb_size)
        # document should be a list of tensor with size (batch_size, seq_len, emb_size)
        query_self_attn_res, _ = self.query_self_attn(query)

        doc_cros_attn_res = [cros_attn_layer(query, document[idx])[0] \
                             for idx, cros_attn_layer in enumerate(self.cros_attn_blocks)]

        # sen_no * (batch_size, seq_len, emb_size) -> sen_no * (batch_size, emb_size)
        doc_cros_attn_res = [F.relu(doc_linear_block(attn_res.transpose(1, 2))).squeeze() \
                            for attn_res, doc_linear_block in zip(doc_cros_attn_res, self.doc_linear_blocks)]

        # sen_no * (batch_size, emb_size) -> (sen_no, batch_size, emb_size)
        doc_cros_attn_res = torch.stack(doc_cros_attn_res)
        # (sen_no, batch_size, emb_size) -> (batch_size, sen_no, emb_size)
        doc_cros_attn_res = doc_cros_attn_res.transpose(0, 1)
        doc_self_attn_res, _ = self.doc_high_self_attn(doc_cros_attn_res)

        # (batch_size, sen_no, emb_size) -> (batch_size, emb_size)
        doc_self_attn_res = F.relu(self.doc_linear(doc_self_attn_res.transpose(1, 2))).squeeze()
        query_self_attn_res = F.relu(self.que_linear(query_self_attn_res.transpose(1, 2))).squeeze()

        # element wize
        #dssm_out = doc_self_attn_res * query_self_attn_res
        dssm_out = torch.cat((doc_self_attn_res, query_self_attn_res), dim=1)
        return dssm_out


class ReviewTower(nn.Module):

    def __init__(self, embedding, embed_dim, rnn_hid_dim, rnn_num_layers, n_reviews):
        super(ReviewTower, self).__init__()

        self.n_reviews = n_reviews
        self.embedding = embedding
        self.embed_dim = embed_dim

        self.review_blocks = nn.ModuleList([
            ReviewGRU(embed_dim, rnn_hid_dim, rnn_num_layers, embedding)
            for _ in range(n_reviews)
        ])

    def forward(self, reviews):
        reviews = [review_block(review) \
                   for review, review_block in zip(reviews, self.review_blocks)]
        return reviews


class ProductTower(nn.Module):
    def __init__(self, embed_model, embed_dim,
                 rnn_hidden_dim, rnn_num_layers,
                 fm_field_dims,
                 fm_embed_dim,
                 rnn_type='GRU',
                 fm_type='fm'):
        super(ProductTower, self).__init__()
        self.embed_model = embed_model
        if rnn_type == 'GRU':
            self.rnn = nn.GRU(embed_dim, rnn_hidden_dim,
                              rnn_num_layers, batch_first=True, 
                              bidirectional=True)
        else:
            self.rnn = nn.LSTM(embed_dim, rnn_hidden_dim,
                            batch_first=True, num_layers=rnn_num_layers, 
                            bidirectional=True)
        self.fm = fm.FactorizationMachineModel(fm_field_dims, fm_embed_dim)

    def forward(self, text, bop):
        embedding = self.embed_model(text)
        rnn_out, _ = self.rnn(embedding)
        fm_out = self.fm(bop)

        return rnn_out, fm_out
