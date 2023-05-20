"""
@Desc:
@Reference:
@Notes:

"""
from typing import List

def are_same_strings(string1: str, string2: str):
    if not isinstance(string1, str) or not isinstance(string2, str):
        raise ValueError("input should be strings")
    return string1.strip().lower() == string2.strip().lower()


def rm_extra_spaces(string: str):
    return " ".join(string.strip().split())

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        import argparse
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def flatten_key_words(key_words:List[str], max_len:int, word_sep: str):
    kw_string = ""
    for one in key_words:
        if kw_string == "" and len(one) <= max_len:
            kw_string = one
            continue
        next_string = f"{kw_string} {word_sep} {one}"
        if len(next_string) > max_len:
            break
        kw_string = next_string
    return kw_string