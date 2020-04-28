import numpy as np
import gzip
import json
import os.path
from field_parser import parser_register
from field_reducer import reducer_register
from collections import defaultdict

class Products:
    def __init__(self, domain_name):
        self.domain_name = domain_name
        self.fields_interested = ['title', 'rank', 'also_view', 'also_buy', 'description']
        self.idx2ftr = defaultdict(dict)
        self._load_meta(f'data/meta_{domain_name}.json.gz')
        self.index_field = 'asin'

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
                for field in self.fields_interested:
                    ftr_parser = parser_register.get(field, None)
                    ftr_val = None
                    if ftr_parser:
                        ftr_val = ftr_parser(line_data.get(field, None))
                    # I have checked the data, there might be multiple records for
                    # a single index, but seems they are simply duplicated
                    self.idx2ftr[index][field] = ftr_val

    def get_record(self, index):
        return self.idx2ftr.get(index, {})

    def get_attributes_column(self):
        return self.fields_interested

if __name__ == '__main__':
    products = Products("AMAZON_FASHION")
    #products = Products("Gift_Cards")
    test_idx = "B00004T3SN"
    res = products.get_record(test_idx)

    print(f"detailed info for product {test_idx} is: \n{json.dumps(res)}")

