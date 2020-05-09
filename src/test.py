import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import logging
import sys
import matplotlib.pyplot as plt

from Products import Products, prepare_prod_batch
from Reviews import Reviews, prepare_rev_batch
from embeddings import get_embed_layer
from model import RecomModel
from conf import *

import sys

# initialize logger
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        
def parse_logger(string=''):
    if not string:
        ret = logging.getLogger('stdout')
        hdlr = logging.StreamHandler(sys.stdout)
    else:
        ret = logging.getLogger(string)
        hdlr = logging.FileHandler(string)
    ret.setLevel(logging.INFO)
    ret.addHandler(hdlr)
    hdlr.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    return ret
   
def prepare_batch_data(idx_batch):

    rev_idx_batch = [p[0] for p in idx_batch]
    prod_idx_batch = [p[1] for p in idx_batch]

    rev_batch = [r.get_reviews(rev_idx=rev_idx) for rev_idx in rev_idx_batch]
    prod_batch = [r.get_reviews(pro_idx=prod_idx) for prod_idx in prod_idx_batch]
    score_batch = [r.get_rating(rev_idx, pro_idx, True)[0] \
                   for rev_idx, pro_idx in zip(rev_idx_batch, prod_idx_batch)]

    target = torch.tensor(score_batch).float()

    prod = prepare_rev_batch(prod_batch, tokenizer, pro_n_sen, seq_len)
    rev = prepare_rev_batch(rev_batch, tokenizer, rev_n_sen, seq_len)
    
    if to_gpu:
        target = target.cuda()
        rev = [r.cuda() for r in rev]
        prod = [r.cuda() for r in prod]
    
    #return text, bop, rev, target
    return rev, prod, target

logger = parse_logger()
logger.setLevel(logging.INFO)

# get data
logger.info("start loading data")
p = Products(domain)
r = Reviews(domain)
# how to split train and test data
# and how to fetch data by batch
r.train_test_split(test_ratio, valid_ratio)
print(f"Size train: {len(r.idx_train)} valid: {len(r.idx_valid)} test: {len(r.idx_test)}")
logger.info("ended loading data")

# get tokenizer and embedding setting
tokenizer, embedding, embed_dim = get_embed_layer(embedding_type)

model = RecomModel(rnn_hidden_dim, rnn_hidden_dim,
                   n_head, n_attn, 
                   seq_len, pro_n_sen, rev_n_sen, 
                   d_k, d_v,
                   embedding,
                   embed_dim,
                   n_rnn,
                   fm_embed_dim=fm_embed_dim,
                   ds_type=ds_type,
                   rnn_type=rnn_type,
                   dropout=dropout,)
                   
#model.apply(init_weights)

if to_gpu:
    to_index = cuda_index
    model = nn.DataParallel(model, device_ids=[cuda_index])
    device = torch.device(f"cuda:{to_index}")
    logger.info(f'sending whole model data to CUDA device {str(device)}')
    model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

model_path = sys.argv[1]

model_data = torch.load(model_path)
model.load_state_dict(model_data['model_state_dict'])
optimizer.load_state_dict(model_data['optimizer_state_dict'])
model.eval()

test_size = len(r.idx_test)
test_loss_list = []
num = 0
loss = 0
test_batch_size = 4
with torch.no_grad():
    test_idx = r.get_batch_bikey(test_size, src='test')
    for fold_no in range(test_size // test_batch_size + 1):
        test_idx_batch = test_idx[fold_no * batch_size:fold_no * batch_size + batch_size]
        if not len(test_idx_batch):
            continue
        num += 1
        rev, prod, target = prepare_batch_data(test_idx_batch)
        res = model(prod, rev)
        fold_loss = criterion(res, target)
        loss += fold_loss
        if fold_no % 1 == 0:
            logger.info(f'testing progress: {fold_no}/{test_size // batch_size + 1}')
test_loss = loss / num
logger.info(f'testing loss: {test_loss:.4}')