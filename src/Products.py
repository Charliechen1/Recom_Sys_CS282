import os
import json
import gzip
import pandas as pd
from urllib.request import urlopen

def num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)

class Products:
    def __init__(self, filename):
        self.df = pd.read_pickle(filename)
        if len(self.df) < 1:
            print("Invalid Filename!")
        else:
            self.attribute_types = list(self.df.columns)
            print("Products in total: ", len(self.df))
        return

    def get_attribute_types(self):
        return self.attribute_types

    def get_record(self, product_code):
        return self.df.loc[product_code]

    def get_attributes(self, product_code, types: []):
        if not types:
            types = self.attribute_types
        data = {}
        record = self.df.loc[product_code]
        for type in types:
            data[type] = record[type]

        return data

    def get_attribute_with_aggregation(self, product_code, types: [], agg_types: []):
        if not types:
            types = self.attribute_types
        data = {}
        record = self.df.loc[product_code]
        for type, agg_type in zip(types, agg_types):
            # data[type] = record[type].values[0]
            info = record[type]
            if agg_type == 'COUNT':
                data[type + '_count'] = len(info)
            elif agg_type == "AVERAGE":
                # might exist NaN element
                sums = 0
                for i in info:
                    if pd.notna(i) :
                        sums += num(i)
                data[type + '_avg'] = sums / len(info)

        return data


if __name__ == "__main__":
    # Expected output:
    # Products in total: 535
    # print the number of products in total
    products = Products('test.pkl')

    # Expected output:
    # ['Gift Amount:', 'Format:', 'Size:']
    # print a list of atttributes' names
    print("Style contains these attributes: ", products.get_attribute_types())

    # Retrieve product infos using its asin
    print("---------------Access the product B001GXRQW0-----------------")
    print("Raw record: ", products.get_record('B001GXRQW0'))
    print("Gift Amount: ", products.get_attributes('B001GXRQW0', ['Gift Amount:']))
    print("Gift Amount Count: ", products.get_attribute_with_aggregation('B001GXRQW0', ['Gift Amount:'], ['COUNT']))
    print("Gift Amount Average: ", products.get_attribute_with_aggregation('B001GXRQW0', ['Gift Amount:'], ['AVERAGE']))