import json
import os
import gzip
from field_parser import parser_register
from field_reducer import reducer_register
from collections import defaultdict
from random import sample
from sklearn.model_selection import train_test_split
import torch
import numpy as np

class Reviews:
    def __init__(self, domain_name):
        self.domain_name = domain_name
        self.idx2rev = defaultdict(list)
        self.biidx2rat = defaultdict(list)
        self.rev_idx_field = 'reviewerID'
        self.pro_idx_field = 'asin'
        self.rev2pro = defaultdict(list)
        self.pro2rev = defaultdict(list)
        self.data_storage = []
        self.bikey_storage = []

        self.interested_fields = ['reviewerName', 'reviewText',
                                  'summary', 'unixReviewTime', 'overall']

        self.idx_train = []
        self.idx_test = []
        self._load_data(f'data/{domain_name}.json.gz')

    def _load_data(self, data_file_name):
        if not os.path.isfile(data_file_name):
            print ("File not exist")
            return

        with gzip.open(data_file_name, 'rb') as f:
            for line in f:
                line_data = json.loads(line)
                rev_idx = line_data[self.rev_idx_field]
                pro_idx = line_data[self.pro_idx_field]
                # load review information
                data = {}
                for field in self.interested_fields:
                    parser = parser_register.get(field, None)
                    if parser:
                        data[field] = parser(line_data.get(field, None))

                num_idx = len(self.data_storage)
                self.data_storage.append(data)

                # save indexing
                self.idx2rev[rev_idx].append(num_idx)
                self.biidx2rat[f'{rev_idx}_{pro_idx}'].append(num_idx)
                self.bikey_storage.append((rev_idx, pro_idx))
                self.rev2pro[rev_idx].append(pro_idx)
                self.pro2rev[pro_idx].append(rev_idx)

    def get_rating(self, rev_idx, pro_idx, filter_none=False):
        whole_idx = f'{rev_idx}_{pro_idx}'
        # don't need to worry about non-existing key for a defaultdict
        data_idx_list = self.biidx2rat[whole_idx]
        scores = [self.data_storage[idx].get('overall', -1)
                  for idx in data_idx_list]

        if filter_none:
            scores = list(filter(lambda x: x >= 0, scores))
        return scores

    def get_reviews(self, rev_idx, pro_idx=None):
        if not pro_idx:
            data_idx_list = self.idx2rev[rev_idx]
        else:
            data_idx_list = self.biidx2rat[f'{rev_idx}_{pro_idx}']
        reviews= [self.data_storage[idx]
                  for idx in data_idx_list]
        # reviews are sorted by their time
        reviews.sort(key=lambda x: x['unixReviewTime'])
        return reviews

    def get_all_rev_idx(self):
        return list(self.idx2rev.keys())

    def get_all_rev(self):
        return self.data_storage

    def get_batch_bikey(self, batch_size, from_train=True):
        if from_train:
            return sample(self.idx_train, batch_size)
        else:
            return sample(self.idx_test, batch_size)

    def get_all_bi_idx(self):
        return self.bikey_storage

    def get_pro_by_rev(self, rev_idx):
        return self.rev2pro.get(rev_idx, [])

    def train_test_split(self, test_size, random_state=42):
        idx_train, idx_test = train_test_split(
            self.bikey_storage, test_size=test_size, random_state=random_state
        )
        self.idx_train = idx_train
        self.idx_test = idx_test
        return idx_train, idx_test


def pad_reviews(reviews, n_reviews):
    padded_reviews = ['' for _ in range(n_reviews)]
    padded_reviews[:min(len(reviews), n_reviews)] = reviews[:min(len(reviews), n_reviews)]
    return padded_reviews

def prepare_rev_batch(rev_batch, tokenizer, n_reviews, max_len):
    reviews_batch = [[rev['reviewText'] for rev in rev_list] for rev_list in rev_batch]
    reviews_pad = [pad_reviews(reviews, n_reviews) for reviews in reviews_batch]
    enc_reviews = [[tokenizer.encode(rev, max_length=max_len, pad_to_max_length=True) \
                    for rev in rev_list] for rev_list in reviews_pad]

    enc_reviews = np.array(enc_reviews)
    enc_reviews = np.transpose(enc_reviews, (1, 0, 2))
    enc_reviews = [torch.tensor(rev_list).long() for rev_list in enc_reviews]
    return enc_reviews

if __name__ == '__main__':
    #r = Reviews('AMAZON_FASHION')
    r = Reviews('Gift_Cards')
    rev_idx_list = r.get_all_rev_idx()
    print(len(rev_idx_list))
    for rev_idx in rev_idx_list:
        pro_idx_list = r.get_pro_by_rev(rev_idx)
        for pro_idx in pro_idx_list:
            reviews = r.get_reviews(rev_idx, pro_idx)
            ratings = r.get_rating(rev_idx, pro_idx)

