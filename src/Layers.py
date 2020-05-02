import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Layers import EncoderLayer, DecoderLayer

class SelfAttnDSSM(nn.Module):
    """
    Idea from paper: "A Hierarchical Attention Retrieval Model for Healthcare Question Answering"
    """
    def __init__(self, d_model, d_inner, n_head, n_sen, d_k, d_v, dropout=0.1):
        self.query_self_attn = EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout)

        self.cros_attn_blocks = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_sen)
        ])

        self.doc_self_attn_blocks = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_sen)
        ])

        self.doc_high_self_attn = EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout)

    def forward(self, query, document):
        # query is the embedding for product title, with size (batch_size, seq_len, emb_size)
        # document should be a list of tensor with size (batch_size, seq_len, emb_size)
        query_self_attn_res = self.query_self_attn(query)

        doc_cros_attn_res = [cros_attn_layer(query, document[idx]) \
                             for idx, cros_attn_layer in enumerate(self.cros_attn_blocks)]

        # sen_no * (batch_size, seq_len, emb_size) -> sen_no * (batch_size, emb_size)
        doc_cros_attn_res = [attn_res.mean(1) for attn_res in doc_attn_res]

        # sen_no * (batch_size, emb_size) -> (sen_no, batch_size, emb_size)
        doc_cros_attn_res = torch.stack(doc_cros_attn_res)
        # (sen_no, batch_size, emb_size) -> (batch_size, sen_no, emb_size)
        doc_cros_attn_res = doc_cros_attn_res.transpose(0, 1)

        doc_self_attn_res = self.doc_high_self_attn(doc_cros_attn_res)

        # (batch_size, sen_no, emb_size) -> (batch_size, emb_size)
        doc_self_attn_res = doc_self_attn_res.mean(1)
        query_self_attn_res = query_self_attn_res.mean(1)

        # element wize
        dssm_out = doc_self_attn_res * query_self_attn_res
        return dssm_out



