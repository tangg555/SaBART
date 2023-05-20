"""
@Desc:
@Reference:
"""

import os
import pickle
import shutil
import json
from pathlib import Path


def reverse_dict_key_val(dict_obj: dict, warning_allowed=True):
    new_dict = {}
    for key, val in dict_obj.items():
        if warning_allowed and val in new_dict:
            print(f"Warning: key ({val}) already inside the new dict, original: {new_dict[val]}; replaced: {key}")
        new_dict[val] = key
    return new_dict

