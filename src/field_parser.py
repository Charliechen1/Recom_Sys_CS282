import pandas as pd

parser_register = dict()

def register(func):
    """
    register the functions as plugins
    """
    func_key = '_'.join(func.__name__.split('_')[1:])
    parser_register[func_key] = func
    return func

@register
def parse_title(ftr):
    if not ftr: return ""
    return ftr

@register
def parse_rank(ftr):
    if not ftr: return -1
    try:
        rank_str = ftr.split('in')[0].replace(',', '')
        return int(rank_str)
    except:
        return -1

@register
def parse_also_view(ftr):
    if not ftr: return []
    return ftr

@register
def parse_also_buy(ftr):
    if not ftr: return []
    return ftr

@register
def parse_description(ftr):
    if not ftr: return []
    return ftr
