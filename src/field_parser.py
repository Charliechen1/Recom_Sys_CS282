import re

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
    if 'getTime' in ftr:
        titles = re.findall(r'\nAmazon.com: ([^\n]+)\n', ftr)
        if len(titles) > 0:
            return titles[0]
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
    if not ftr: return ""
    return ' '.join(ftr)

@register
def parse_reviewerName(ftr):
    if not ftr: return ""
    return ftr

@register
def parse_reviewText(ftr):
    if not ftr: return ""
    return ftr

@register
def parse_summary(ftr):
    if not ftr: return ""
    return ftr

@register
def parse_unixReviewTime(ftr):
    if not ftr: return -1
    return int(ftr)

@register
def parse_overall(ftr):
    if not ftr: return -1
    return int(ftr)

@register
def parse_style(ftr):
    if not ftr: return {}
    size_dict = {
        "x-small": 1,
        "xs": 1,
        "small": 2,
        "s": 2,
        "medium": 3,
        "m": 3,
        "large": 4,
        "l": 4,
        "x-large": 5,
        "xl": 5,
        "xx-large": 6,
        "xxl": 6,
        "xxx-large": 7,
        "xxxl": 8
    }
    res = {}
    res['metal_type'] = ftr.get('Metal Type:', '').strip()
    res['gem_type'] = ftr.get('Gem Type:', '').strip()
    res['length'] = ftr.get('Length', 0.0)
    res['size'] = size_dict.get(ftr.get('Size:', '').strip().split(' ')[0].lower(), 0)
    res['style'] = ftr.get('Style:', '').strip()

    return res
