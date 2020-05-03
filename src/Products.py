import numpy as np
import gzip
import json
import os.path
from src.field_parser import parser_register
from src.field_reducer import reducer_register
from collections import defaultdict
from random import sample
import torch


class Products:
    def __init__(self, domain_name):
        self.domain_name = domain_name
        self.fields_interested = ['title', 'rank', 'also_view', 'also_buy', 'description']
        self.reducer_fields = {'also_view': ('bag_of_prod', 'freq_also_view'),
                               'also_buy': ('bag_of_prod', 'freq_also_buy')}
        self.idx2ftr = defaultdict(int)
        self.index_field = 'asin'
        self.data_storage = []
        self._load_meta(f'data/meta_{domain_name}.json.gz')
        self._load_attr('freq_also_view', 'dict/high_freq_1000_also_view.txt')
        self._load_attr('freq_also_buy', 'dict/high_freq_1000_also_buy.txt')

    def _load_meta(self, meta_file_name):
        """
        Function to load meta information. This is the major source of features
        """
        # asin is the index for product
        # currently we fetch these part of data
        if not os.path.isfile(meta_file_name):
            print ("File not exist")
            return

        with gzip.open(meta_file_name, 'rb') as f:
            for line in f:
                line_data = json.loads(line)
                index = line_data[self.index_field]
                if not index:
                    continue
                data = {}
                for field in self.fields_interested:
                    ftr_parser = parser_register.get(field, None)
                    ftr_val = None
                    if ftr_parser:
                        ftr_val = ftr_parser(line_data.get(field, None))
                    # I have checked the data, there might be multiple records for
                    # a single index, but seems they are simply duplicated
                    data[field] = ftr_val
                num_idx = len(self.data_storage)
                self.data_storage.append(data)
                self.idx2ftr[index] = num_idx

    def _load_attr(self, attr_name, file_name):
        with open(file_name, 'rb') as f:
            data = []
            idx = {}
            for line in f:
                piece = line.strip()
                data.append(piece)
                idx[piece] = len(data) - 1
            setattr(self, attr_name, data)
            setattr(self, f'{attr_name}_idx', idx)

    def _reduce_data(self, num_idx):
        prod = self.data_storage[num_idx]
        for field, conf in self.reducer_fields.items():
            reducer_name, dict_name = conf
            reducer = reducer_register.get(reducer_name, None)
            if not reducer:
                continue
            field_val = prod.get(field, None)
            prod[f'{field}_{reducer_name}'] = reducer(field_val,
                                                      idx2ftr=getattr(self, f'{dict_name}_idx'),
                                                      data_storage=getattr(self, f'{dict_name}'))


    def get_product(self, index, reduce=True):
        num_idx = self.idx2ftr.get(index, -1)
        if num_idx == -1:
            return None
        if reduce:
            self._reduce_data(num_idx)
        return self.data_storage[num_idx]

    def get_product_list(self, index_list):
        num_index_list = [self.idx2ftr[idx] for idx in index_list]

    def get_attributes_column(self):
        return self.fields_interested

    def get_all_product_idx(self):
        return list(self.idx2ftr.keys())

    def get_all_product(self):
        return self.data_storage

    def get_batch(self, batch_size):
        return sample(self.data_storage, batch_size)


def prepare_prod_batch(prod_batch, tokenizer):
    encoded_texts = []
    bops = []

    for prod in prod_batch:
        # text part
        text = f"title: {prod['title']}, description: {prod['description']}"
        encoded_text = tokenizer.encode(text, max_length=100, pad_to_max_length=True)
        encoded_texts.append(encoded_text)

        # BoP part
        bop = np.concatenate((prod['also_view_bag_of_prod'], prod['also_view_bag_of_prod']), axis=0)
        bops.append(bop)

    text_tensor = torch.tensor(encoded_texts)
    bop_tensor = torch.from_numpy(np.stack(bops, axis=0)).long()

    return text_tensor, bop_tensor


if __name__ == '__main__':
    products = Products("AMAZON_FASHION")
    # products = Products("Gift_Cards")
    test_idx = "B01GKWEJTO"
    res = products.get_product(test_idx)

    print(f"detailed info for product {test_idx} is: \n{json.dumps(res)}")

    # all_rec = products.get_all_record()
    num_iter, batch_size = 100000, 128
    for _ in range(num_iter):
        batch_sample = products.get_batch(batch_size)

