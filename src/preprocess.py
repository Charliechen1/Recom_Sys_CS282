import pandas as pd
import gzip
import json
import os
import numpy as np


root = "../data/"
os.chdir(root)


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield json.loads(l)


def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


def reviews(df):
    df['review-item-tuple'] = df[['asin', 'reviewText']].apply(tuple, axis=1)
    return df.groupby('reviewerID')['review-item-tuple'].agg(list)


def descriptions(df):
    df['description'] = df['description'].apply(lambda x: [] if x is np.nan else x)
    return df.groupby('asin')['description'].agg(lambda s: sum(s, []))


def ratings(df):
    df["both"] = df[["reviewerID", "asin"]].apply('-'.join, axis=1)
    return df.groupby('both')[['overall', 'reviewText']].agg(list)


class DataLoader:
    def __init__(self, name='Gift_Cards'):
        data_path = name + '.json.gz'
        meta_path = 'meta_' + data_path
        self.data_table = getDF(data_path)
        meta_table = getDF(meta_path)
        self.reviews = reviews(self.data_table)
        self.descriptions = descriptions(meta_table)
        self.ratings = ratings(self.data_table)

    def get_descriptions(self, item_id):
        return self.descriptions[item_id]

    def get_rating(self, user_id, item_id):
        key = '-'.join([user_id, item_id])
        if key in self.ratings.index:
            ret_list = list(self.ratings.loc[key])
        else:
            ret_list = [[], []]

        ret_list.append(list(self.descriptions[item_id]))

        return ret_list

    def get_reviews(self, user_id):
        return self.reviews[user_id]

    def user_keys(self):
        return list(self.reviews.index)

    def item_keys(self):
        return list(self.descriptions.index)
