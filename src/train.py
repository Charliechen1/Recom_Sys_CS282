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

    #corres_rev_batch = [r.get_reviews(idx) for idx in rev_idx_batch]
    #rev_batch = [corres_rev_batch[0] + r.get_reviews(rev_idx, prod_idx) 
    #    for rev_idx, prod_idx, corees_rev \
    #        in zip(rev_idx_batch, prod_idx_batch, corres_rev_batch)]
    rev_batch = [r.get_reviews(rev_idx, prod_idx) 
                for rev_idx, prod_idx in zip(rev_idx_batch, prod_idx_batch)]
    prod_batch = [p.get_product(idx, reduce=True) for idx in prod_idx_batch]
    score_batch = [r.get_rating(rev_idx, pro_idx, True)[0] > cls_thre\
                   for rev_idx, pro_idx in zip(rev_idx_batch, prod_idx_batch)]

    target = torch.tensor(score_batch).long()

    text, bop = prepare_prod_batch(prod_batch, tokenizer, seq_len)
    rev = prepare_rev_batch(rev_batch, tokenizer, n_reviews, seq_len)
    
    if to_gpu:
        target = target.cuda()
        text = text.cuda()
        bop = bop.cuda()
        rev = [r.cuda() for r in rev]
    
    return text, bop, rev, target

def triple_loss(a, p, n, margin=0.2) : 
    d = nn.PairwiseDistance(p=2)
    distance = d(a, p) - d(a, n) + margin 
    loss = torch.mean(torch.max(distance, torch.zeros_like(distance))) 
    return loss

logger = parse_logger()
logger.setLevel(logging.INFO)

# get data
logger.info("start loading data")
p = Products(domain)
r = Reviews(domain)
# how to split train and test data
# and how to fetch data by batch
r.train_test_split(test_ratio)
logger.info("ended loading data")

# get tokenizer and embedding setting
tokenizer, embedding, embed_dim = get_embed_layer(embedding_type)

model = RecomModel(rnn_hidden_dim, rnn_hidden_dim,
                   n_head, seq_len, n_reviews,
                   d_k, d_v,
                   embedding,
                   embed_dim,
                   n_rnn,
                   fm_field_dims=[2] * fm_n,
                   fm_embed_dim=fm_embed_dim,
                   rnn_type=rnn_type,
                   fm_type=fm_type,
                   dropout=0.1,)
                   
model.apply(init_weights)

if to_gpu:
    to_index = cuda_index
    model = nn.DataParallel(model, device_ids=[cuda_index])
    device = torch.device(f"cuda:{to_index}")
    logger.info(f'sending whole model data to CUDA device {str(device)}')
    model.to(device)


logger.info(f"Model is on gpu: {next(model.parameters()).is_cuda}")

#criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

loss_track, acc_track = [], []

logger.info('start training')
for no in range(no_of_iter):
    model.zero_grad()
    
    train_idx_batch = r.get_batch_bikey(batch_size, from_train=False)
    text, bop, rev, target = prepare_batch_data(train_idx_batch)

    res = model(text, bop, rev)
    loss = criterion(res, target)
    
    pred = torch.max(res, 1)[1]
    
    train_acc = np.sum(np.array(pred.tolist()) == np.array(target.tolist())) / batch_size
    
    logger.info(f'{no}/{no_of_iter} of iterations, current loss: {loss:.4}, current acc: {train_acc:.2%}')
    loss.backward()
    optimizer.step()
    
    loss_track.append(loss)
    acc_track.append(train_acc)

x = list(range(no_of_iter))
fig, axs = plt.subplots(2)
axs[0].plot(x, acc_track)
axs[0].set_title('Training acc')
axs[1].plot(x, loss_track)
axs[1].set_title('Loss value')
plt.savefig('training_record.jpg')

# start testing
test_idx_batch = r.get_batch_bikey(test_size, from_train=False)
text, bop, rev, target = prepare_batch_data(test_idx_batch)
model.eval()
res = model(text, bop, rev)
loss = criterion(res, target)

logger.info(f'testing loss: {loss:.4}')

pred = torch.max(res, 1)[1]
test_acc = np.sum(np.array(pred.tolist()) == np.array(target.tolist())) / test_size
logger.info(f'train acc: {test_acc:.2%}')
logger.info(pred)
logger.info(target)

