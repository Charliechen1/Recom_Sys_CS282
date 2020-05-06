# data setting
#domain = 'AMAZON_FASHION'
#domain = 'Gift_Cards'
domain = 'Video_Games'

# model setting
embedding_type = 'bert'
rnn_type = 'GRU'
fm_type = 'fm'
batch_size = 128
n_reviews = 1
seq_len = 100
fm_n = 2000
fm_embed_dim = 16
n_head = 16
d_k = d_v = 64
n_rnn = 1
rnn_hidden_dim = 200

# training setting
to_gpu=False
test_ratio = 0.2
no_of_iter = 300
lr=1e-3
weight_decay=1e-2
cuda_index=0
pretrain_freeze=True
cls_thre=3

# test setting
test_size = 128


