import torch
import re


def bag_of_products(prod_list, prod2idx):
    bop = [0]*len(prod2idx)
    for prod in prod_list:
        if prod in prod2idx:
            bop[prod2idx[prod]] += 1
    return bop


def embed_title_batch(titles, model, tokenizer, seq_len):
    encoded_batch = tokenizer.batch_encode_plus(titles, max_length=seq_len, pad_to_max_length=True)
    input_ids = torch.tensor(encoded_batch['input_ids'])
    attention_mask = torch.tensor(encoded_batch['attention_mask'])
    out = model(input_ids, attention_mask)
    return out[0]


def process_prod_batch(prod_batch, model, tokenizer, seq_len, prod2idx):
    texts = []
    also_views = []
    also_buys = []

    for prod in prod_batch:
        if 'getTime' in prod['title']:
            title = re.findall(r'\nAmazon.com: ([^\n]+)\n', prod['title'])[0]
        else:
            title = prod['title']
        description = ' '.join(prod['description'])
        text = f"title: {title}, description: {description}"
        texts.append(text)
        also_views.append(bag_of_products(prod['also_view'], prod2idx))
        also_buys.append(bag_of_products(prod['also_buy'], prod2idx))

    return embed_title_batch(texts, model, tokenizer, seq_len), torch.tensor(also_views), torch.tensor(also_buys)