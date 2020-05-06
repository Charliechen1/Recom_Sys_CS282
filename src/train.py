import torch
import torch.nn as nn
import torch.optim as optim

import logging
import sys
import matplotlib.pyplot as plt

from Products import Products, prepare_prod_batch
from Reviews import Reviews, prepare_rev_batch
from embeddings import get_embed_layer
from model import RecomModel
from conf import *

# initialize logger
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

    rev_batch = [r.get_reviews(idx) for idx in rev_idx_batch]
    prod_batch = [p.get_product(idx, reduce=True) for idx in prod_idx_batch]
    score_batch = [r.get_rating(rev_idx, pro_idx, True)[0]\
                   for rev_idx, pro_idx in zip(rev_idx_batch, prod_idx_batch)]

    target = torch.tensor(score_batch).float()

    text, bop = prepare_prod_batch(prod_batch, tokenizer, seq_len)
    rev = prepare_rev_batch(rev_batch, tokenizer, n_reviews, seq_len)
    
    if to_gpu:
        target = target.cuda()
        text = text.cuda()
        bop = bop.cuda()
        rev = [r.cuda() for r in rev]
    
    return text, bop, rev, target

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

if to_gpu:
    to_index = cuda_index
    model = nn.DataParallel(model, device_ids=[cuda_index])
    device = torch.device(f"cuda:{to_index}")
    logger.info(f'sending whole model data to CUDA device {str(device)}')
    model.to(device)


logger.info(f"Model is on gpu: {next(model.parameters()).is_cuda}")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
loss_track = []


logger.info('start training')
for no in range(no_of_iter):
    model.zero_grad()
    
    train_idx_batch = r.get_batch_bikey(batch_size, from_train=False)
    text, bop, rev, target = prepare_batch_data(train_idx_batch)

    res = model(text, bop, rev)
    loss = criterion(res, target)

    loss_track.append(loss)
    logger.info(f'{no}/{no_of_iter} of iterations, current loss: {loss:.4}')

    loss.backward()
    optimizer.step()

x = list(range(no_of_iter))
plt.plot(x, loss_track)
plt.savefig('loss.jpg')

# start testing
test_idx_batch = r.get_batch_bikey(test_size, from_train=False)
text, bop, rev, target = prepare_batch_data(test_idx_batch)
model.eval()
res = model(text, bop, rev)
loss = criterion(res, target)

logger.info(f'testing loss: {loss:.4}')

print(res)
print(target)

