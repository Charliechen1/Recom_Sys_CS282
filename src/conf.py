# data setting
domain = 'AMAZON_FASHION'

# model setting
embedding_type = 'bert'
rnn_type = 'GRU'
fm_type = 'fm'
batch_size = 4
n_reviews = 40
seq_len = 100
fm_n = 2000
fm_embed_dim = 16
n_head = 16
d_k = d_v = 64
n_rnn = 1
rnn_hidden_dim = 200

# training setting
to_gpu=True
test_ratio = 0.2
no_of_iter = 100
lr=1e-4
weight_decay=0.01
cuda_index=0
pretrain_freeze=True


