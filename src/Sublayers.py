import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from conf import *

"""
modified from https://github.com/jadore801120/attention-is-all-you-need-pytorch.git
"""

class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn

class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(
            self, query, doc_sen,
            slf_attn_mask=None, dec_enc_attn_mask=None):

        cro_output, cro_enc_attn = self.enc_attn(
            query, query, doc_sen, mask=dec_enc_attn_mask)

        cro_output, cro_slf_attn = self.slf_attn(
            cro_output, cro_output, cro_output, mask=slf_attn_mask)

        cro_output = self.pos_ffn(cro_output)
        return cro_output, cro_enc_attn, cro_slf_attn

######################################
# Following are the sublayers needed #
######################################
class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        q = self.layer_norm(q)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        return q, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x
        x = self.layer_norm(x)

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        return x


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class ReviewGRU(nn.Module):

    def __init__(self, embed_dim, rnn_hid_dim, rnn_num_layers, embedding):
        super().__init__()

        self.embedding = embedding
        self.rnn_num_layers = rnn_num_layers
        if rnn_num_layers:
            if rnn_type=='GRU':
                self.rnn = nn.GRU(embed_dim, rnn_hid_dim,
                              num_layers=rnn_num_layers,
                              batch_first = True, bidirectional=True)
            else:
                self.rnn = nn.LSTM(embed_dim, rnn_hidden_dim,
                                   num_layers=rnn_num_layers,
                                   batch_first=True, bidirectional=True)


    def forward(self, x):

        x = self.embedding(x)
        
        if self.rnn_num_layers:
            x, _ = self.rnn(x)
        
        return x

class TorchFM(nn.Module):
    def __init__(self, n=None, k=None):
        super().__init__()
        # Initially we fill V with random values sampled from Gaussian distribution
        # NB: use nn.Parameter to compute gradients
        self.V = nn.Parameter(torch.randn(n, k),requires_grad=True)
        self.lin = nn.Linear(n, 1)
        
    def forward(self, x):
        out_1 = torch.matmul(x, self.V).pow(2).sum(1, keepdim=True) #S_1^2
        out_2 = torch.matmul(x.pow(2), self.V.pow(2)).sum(1, keepdim=True) # S_2
        
        out_inter = 0.5 * (out_1 - out_2)
        out_lin = self.lin(x)
        out = out_inter + out_lin
        
        return out


# sublayers for CNNDSSM
class LAttn(nn.Module):
    def __init__(self, T, emb_size, win_size, n_channels):
        super().__init__()
        assert win_size % 2 == 1
        self.attention = nn.Sequential(
            nn.Conv2d(1, 1, (win_size, emb_size), padding=((win_size-1)//2, 0)),
            nn.Sigmoid()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(1, n_channels, (1, emb_size)),
            nn.MaxPool2d((T, 1))
        )

    def forward(self, x):
        scores = self.attention(torch.unsqueeze(x, dim=1))
        out = torch.mul(x, torch.squeeze(scores, dim=1))
        out = torch.squeeze(self.conv(torch.unsqueeze(out, 1)))
        return out


class GAttn(nn.Module):
    def __init__(self, T, emb_size, filter_lengths, n_channels_list):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(1, 1, (T*2 - 1, emb_size), padding=(T - 1, 0)),
            nn.Sigmoid()
        )
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, n_channels, (filter_length, emb_size)),
                nn.Tanh(),
                nn.MaxPool2d((T - filter_length + 1, 1))
            ) for filter_length, n_channels in zip(filter_lengths, n_channels_list)
        ])

    def forward(self, x):
        scores = self.attention(torch.unsqueeze(x, dim=1))
        print(x.shape)
        print(scores.shape)
        out = torch.mul(x, torch.squeeze(scores, dim=1))
        outs = [torch.squeeze(conv(torch.unsqueeze(out, 1))) for conv in self.convs]
        return outs


class FCLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, dropout):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        out = self.linear1(x)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = F.relu(out)
        return out