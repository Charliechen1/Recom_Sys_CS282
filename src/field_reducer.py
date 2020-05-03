import numpy as np

reducer_register = dict()

def register(func):
    """
    register the functions as plugins
    """
    func_key = '_'.join(func.__name__.split('_')[:-1])
    reducer_register[func_key] = func
    return func

@register
def count_reducer(ftr, **kwargs):
    if type(ftr) != list:
        return 0
    try:
        return len(ftr)
    except:
        return 0

@register
def max_reducer(ftr, **kwargs):
    try:
        return max(ftr)
    except:
        return None

@register
def min_reducer(ftr, **kwargs):
    try:
        return min(ftr)
    except:
        return None

@register
def avg_reducer(ftr, **kwargs):
    try:
        return sum(ftr) / len(ftr)
    except:
        return None

@register
def bag_of_prod_reducer(ftr, **kwargs):
    idx2ftr = kwargs['idx2ftr']
    ds = kwargs['data_storage']
    res = np.zeros(len(ds))

    for prod_id in ftr:
        if prod_id not in idx2ftr:
            continue
        res[idx2ftr[prod_id]] = 1

    return res
