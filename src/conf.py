# data setting
#domain = 'AMAZON_FASHION'
#domain = 'Gift_Cards'
domain = 'Video_Games'
#domain = 'Movies_and_TV'

# model setting
embedding_type = 'glove'
rnn_type = 'GRU'
fm_type = 'fm'
dssm_type = 'dnn_dssm'
batch_size = 16
pro_n_sen = 30
rev_n_sen = 15
seq_len = 150
fm_n = 2000
fm_embed_dim = 16
n_head = 12
d_k = d_v = 64
n_rnn = 1
rnn_hidden_dim = 200

# training setting
to_gpu = True
test_ratio = 0.1
valid_ratio = 0.1
no_of_iter = 2000
lr = 2e-6
weight_decay = 1e-6
cuda_index = 0
pretrain_freeze = False

# test setting
valid_size = 50