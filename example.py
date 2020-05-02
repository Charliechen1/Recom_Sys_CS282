from src.Products import Products
from src.Reviews import Reviews
import time

domain = 'Gift_Cards'
p = Products(domain)
r = Reviews(domain)

# how to split train and test data
# and how to fetch data by batch
r.train_test_split(0.2)
print(len(r.bikey_storage))
print(len(r.idx_train), len(r.idx_test))

batch_size = 128

no_iter = 10000

s = time.time()
for _ in range(no_iter):
    train_idx_batch = r.get_batch_bikey(batch_size, from_train=True)

    train_rev_idx_batch = [p[0] for p in train_idx_batch]
    train_prod_idx_batch = [p[1] for p in train_idx_batch]

    train_rev_batch = [r.get_reviews(idx) for idx in train_rev_idx_batch]
    train_prod_batch = [p.get_product(idx) for idx in train_prod_idx_batch]

    train_prod_batch = p.get_product_list(train_prod_idx_batch)

e = time.time()
print(f'time spent: {e - s}')


