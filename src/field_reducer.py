import numpy as np

reducer_register = dict()

def register(func):
    """
    register the functions as plugins
    """
    func_key = '_'.join(func.__name__.split('_')[1:])
    reducer_register[func_key] = func
    return func

@register
def count_reducer(ftr):
    if type(ftr) != list:
        return 0
    try:
        return len(ftr)
    except:
        return 0

@register
def max_reducer(ftr):
    try:
        return max(ftr)
    except:
        return None

@register
def min_reducer(ftr):
    try:
        return min(ftr)
    except:
        return None

@register
def avg_reducer(ftr):
    try:
        return sum(ftr) / len(ftr)
    except:
        return None
