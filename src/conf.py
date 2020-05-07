# data setting
#domain = 'AMAZON_FASHION'
#domain = 'Gift_Cards'
domain = 'Video_Games'
#domain = 'Movies_and_TV'

# model setting
embedding_type = 'bert'
rnn_type = 'GRU'
fm_type = 'fm'
batch_size = 128
n_reviews = 1
seq_len = 100
fm_n = 2000
fm_embed_dim = 16
n_head = 12
d_k = d_v = 64
n_rnn = 1
rnn_hidden_dim = 200

# training setting
to_gpu=True
test_ratio = 0.2
no_of_iter = 500
lr=1e-6
weight_decay=1e-6
cuda_index=0
pretrain_freeze=False
cls_thre=3.5

# test setting
test_size = 128