import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Sublayers import EncoderLayer, DecoderLayer, ReviewGRU, TorchFM, LAttn, GAttn, FCLayer
#from embeddings import get_embed_layer
import torchfm.model.fm as fm

# Based on the model proposed in "Interpretable Convolutional Neural Networks with Dual Local
# and Global Aention for Review Rating Prediction"


class CNNDSSM(nn.Module):
    def __init__(self, emb_size, seq_len, doc_n_sen, que_n_sen,
                 win_size=5, n_channels=200, filter_lengths=[2, 3, 4], n_channels_list=[100, 100, 100],
                 hidden_dim=500, out_dim=50, dropout=0.5):
        super().__init__()
        T_user = seq_len * que_n_sen
        T_item = seq_len * doc_n_sen
        self.local_attention_user = LAttn(T_user, emb_size, win_size, n_channels)
        self.global_attention_user = GAttn(T_user, emb_size, filter_lengths, n_channels_list)
        self.local_attention_item = LAttn(T_item, emb_size, win_size, n_channels)
        self.global_attention_item = GAttn(T_item, emb_size, filter_lengths, n_channels_list)
        self.fc_user = FCLayer(n_channels + np.sum(n_channels_list), hidden_dim, out_dim, dropout)
        self.fc_item = FCLayer(n_channels + np.sum(n_channels_list), hidden_dim, out_dim, dropout)

    def forward(self, document, query):
        #user side (query)
        x_u = torch.cat(query, dim=1)
        l_out_u = self.local_attention_user(x_u)
        g_out_u = self.global_attention_user(x_u)
        out_u = torch.cat([l_out_u] + g_out_u, dim=1)
        out_u = self.fc_user(out_u)

        #item side (document)
        x_i = torch.cat(document, dim=1)
        l_out_i = self.local_attention_item(x_i)
        g_out_i = self.global_attention_item(x_i)
        out_i = torch.cat([l_out_i] + g_out_i, dim=1)
        out_i = self.fc_item(out_i)

        return out_i, out_u


    
class SimpleFC(nn.Module):
    def __init__(self, d_model, seq_len, doc_n_sen, que_n_sen, dropout=0.1):
        super(SimpleFC, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.doc_linear1 = nn.Linear(seq_len * doc_n_sen, seq_len * doc_n_sen)
        self.que_linear1 = nn.Linear(seq_len * que_n_sen, seq_len * que_n_sen)
        self.doc_linear2 = nn.Linear(seq_len * doc_n_sen, 1)
        self.que_linear2 = nn.Linear(seq_len * que_n_sen, 1)
    
    def forward(self, document, query):
        # sen_no * (batch_size, seq_len, emb_size) -> (batch_size, seq_len * sen_no, emb_size)
        document = torch.cat(document, dim=1).transpose(1, 2)
        query = torch.cat(query, dim=1).transpose(1, 2)
        
        # (batch_size, seq_len * sen_no, emb_size) -> (batch_size, emb_size)
        document = self.act(self.doc_linear1(document))
        query = self.act(self.que_linear1(query))
        document = self.act(self.doc_linear2(document).squeeze(-1))
        query = self.act(self.que_linear2(query).squeeze(-1))

        return document, query

class DeepCoNN(nn.Module):
    def __init__(self, emb_size, seq_len, doc_n_sen, que_n_sen, win_size=5, depth=1):
        super(DeepCoNN, self).__init__()
        self.pool_dim = 2 * (emb_size - 1)//2 + 1
        self.conv = nn.Sequential(
            nn.Conv2d(1, 1, (win_size, emb_size), padding=((win_size-1)//2, (emb_size - 1)//2)),
            nn.ReLU(),
            nn.MaxPool2d((1, self.pool_dim))
        ) 
        self.que_fc = nn.Linear(seq_len * que_n_sen, emb_size)
        self.doc_fc = nn.Linear(seq_len * doc_n_sen, emb_size)

    def forward(self, document, query):
        # sen_no * (batch_size, seq_len, emb_size) -> (batch_size, seq_len * sen_no, emb_size)
        document = torch.cat(document, dim=1)
        query = torch.cat(query, dim=1)

        # (batch_size, seq_len * sen_no, emb_size) -> (batch_size, emb_size)
        document = self.conv(document)
        query = self.conv(query)
        document = self.doc_fc(document)
        query = self.que_fc(query)

        return document, query


class DotProdAttnDSSM(nn.Module):
    """
    Idea from paper: "A Hierarchical Attention Retrieval Model for Healthcare Question Answering"
    """
    def __init__(self, d_model, d_inner, 
                 n_head, n_attn,
                 seq_len, doc_n_sen, que_n_sen, 
                 d_k, d_v, 
                 dropout=0.1):
        super(DotProdAttnDSSM, self).__init__()
        
        self.n_attn = n_attn
        
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        # both query and document are with size:
        # sen_no * (batch_size, seq_len, emb_size)
        
        # Just for cross attention:
        # query: sen_no * (batch_size, seq_len, emb_size) -> (batch_size, seq_len, emb_size)
        self.que_sen_linear = nn.Linear(que_n_sen, 1)
        
        # apply cross attention of query to document
        # document: sen_no * (batch_size, seq_len, emb_size) unchanged
        self.cros_attn_blocks = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(doc_n_sen)
        ])
        
        self.doc_self_attn_blocks = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(doc_n_sen)
        ])
        
        self.que_self_attn_blocks = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(que_n_sen)
        ])
        
        self.que_ff_linear = nn.Linear(seq_len * que_n_sen, seq_len * que_n_sen)
        self.doc_ff_linear = nn.Linear(seq_len * doc_n_sen, seq_len * doc_n_sen)
        self.que_emb_linear = nn.Linear(seq_len * que_n_sen, 1)
        self.doc_emb_linear = nn.Linear(seq_len * doc_n_sen, 1)

        
    def forward(self, document, query):
        # both query and document are with size:
        # sen_no * (batch_size, seq_len, emb_size)
        
        # from sen_no * (batch_size, seq_len, emb_size)
        # to (sen_no, batch_size, seq_len, emb_size)
        # to (emb_size, batch_size, seq_len, sen_no)
        query_enc = torch.stack(query).transpose(0, 3)
        
        # (emb_size, batch_size, seq_len, sen_no) -> (emb_size, batch_size, seq_len)
        # (emb_size, batch_size, seq_len) -> (batch_size, emb_size, seq_len)
        # (batch_size, emb_size, seq_len) -> (batch_size, seq_len, emb_size)
        query_enc = self.que_sen_linear(query_enc).squeeze(-1).transpose(0, 1).transpose(1, 2)
        
        # sen_no * (batch_size, seq_len, emb_size) unchanged
        doc_attn_res = [cros_attn_layer(query_enc, document[idx])[0] \
                            for idx, cros_attn_layer in enumerate(self.cros_attn_blocks)]
        
        del query_enc
        
        que_attn_res = [self_attn_layer(query[idx])[0] \
                            for idx, self_attn_layer in enumerate(self.que_self_attn_blocks)]
        
        doc_attn_res = [self_attn_layer(doc_attn_res[idx])[0] \
                            for idx, self_attn_layer in enumerate(self.doc_self_attn_blocks)]
        
        # sen_no * (batch_size, seq_len, emb_size) -> (batch_size, seq_len * sen_no, emb_size) 
        que_attn_res = torch.cat(query, dim=1)
        # sen_no * (batch_size, seq_len, emb_size) -> (batch_size, seq_len * sen_no, emb_size) 
        doc_attn_res = torch.cat(doc_attn_res, dim=1)
        
        # (batch_size, seq_len * sen_no, emb_size) -> (batch_size, emb_size, seq_len * sen_no) 
        que_attn_res = que_attn_res.transpose(1, 2)
        doc_attn_res = doc_attn_res.transpose(1, 2)
        
        # (batch_size, emb_size, seq_len * sen_no) unchanged
        que_attn_res = self.act(self.que_ff_linear(que_attn_res))
        doc_attn_res = self.act(self.doc_ff_linear(doc_attn_res))
        
        # (batch_size, emb_size, seq_len * sen_no) -> (batch_size, emb_size)
        que_attn_res = self.act(self.que_emb_linear(que_attn_res)).squeeze(-1)
        doc_attn_res = self.act(self.doc_emb_linear(doc_attn_res)).squeeze(-1)
        
        return doc_attn_res, que_attn_res


class DocumentNet(nn.Module):

    def __init__(self, embedding, embed_dim, rnn_hid_dim, rnn_num_layers, n_reviews):
        super(DocumentNet, self).__init__()

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


class QueryNet(nn.Module):
    def __init__(self, embed_model, embed_dim,
                 rnn_hidden_dim, rnn_num_layers,
                 fm_field_dims,
                 fm_embed_dim,
                 rnn_type='GRU',
                 fm_type='fm'):
        super(QueryNet, self).__init__()
        self.embed_model = embed_model
        self.rnn_num_layers = rnn_num_layers
        if rnn_num_layers:
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
        if self.rnn_num_layers:
            rnn_out, _ = self.rnn(embedding)
        else:
            rnn_out = embedding
        fm_out = self.fm(bop)

        return rnn_out, fm_out

    
class ConcatFF(nn.Module):
    def __init__(self, d_model, dropout):
        super(ConcatFF, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.linear1 = nn.Linear(d_model * 2, d_model * 2)
        self.linear2 = nn.Linear(d_model * 2, d_model * 2)
        self.linear3 = nn.Linear(d_model * 2, 1)
        
    def forward(self, doc_emb, que_emb):
        concat = torch.cat((doc_emb, que_emb), dim=1)
        # feedforward part
        concat = concat + self.dropout(self.act(self.linear1(concat)))
        concat = concat + self.dropout(self.act(self.linear2(concat)))
        concat = self.linear3(concat).squeeze(-1)
        
        return concat
    
    
class ConcatFM(nn.Module):
    def __init__(self, d_model, fm_embed_dim, dropout):
        super(ConcatFM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        
        self.fm = TorchFM(d_model * 2, fm_embed_dim)
        
    def forward(self, doc_emb, que_emb):
        
        concat = torch.cat((doc_emb, que_emb), dim=1)
        # fm part
        res = self.fm(concat)
        return res
    
class DotProd(nn.Module):
    def __init__(self, d_model, dropout):
        super(DotProd, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, 1)
        
    def forward(self, doc_emb, que_emb):
        
        mul = doc_emv * que_emb
        res = self.dropout(self.act(self.linear(mul)))
        res = self.linear2(res).squeeze()
        
        return res
    
