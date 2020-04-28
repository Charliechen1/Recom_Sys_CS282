
import json
import gzip
import pandas as pd
from pandas.io.json import json_normalize

categories = ['Gift_Cards', 'Magazine_Subscriptions']
df_total = pd.DataFrame(columns=['asin'])
for category in categories:
    data = []
    with gzip.open(category + '.json.gz') as f:
        for l in f:
            data.append(json.loads(l.strip()))
    df = pd.DataFrame.from_dict(data)
    df_clean = df[['asin', 'style']]
    df_clean = df_clean[df_clean['style'].notnull()]
    A = df_clean[['asin']].join([json_normalize(df_clean['style'].tolist())])
    style = list(A.columns)[1:]
    B = A.drop_duplicates()
    temp = B.groupby('asin')[style].agg(lambda x: list(x))
    df_total = pd.concat([df_total, temp], sort=False)

df_total.to_pickle('test.pkl')