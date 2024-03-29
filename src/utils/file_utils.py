"""
@Desc:
@Reference:
"""

import os
import pickle
import shutil
import json
import linecache
from pathlib import Path


def save_json(content, file_path: Path, indent=4, **json_dump_kwargs):
    with file_path.open("w", encoding="utf-8") as fw:
        json.dump(content, fw, indent=indent, ensure_ascii=False, **json_dump_kwargs)


def load_json(file_path: Path):
    with open(file_path, "r", encoding="utf-8") as fr:
        return json.load(fr)

def save_list(content: list, file_path: Path):
    with file_path.open("w", encoding="utf-8") as fw:
        fw.writelines(content)

def copy_file(source_path, target_path):
    shutil.copy(source_path, target_path)

def copy_dir(source_path, target_path):
    shutil.copytree(source_path, target_path)

def rm_dir(dest_path):
    shutil.rmtree(dest_path)

def output_obj_to_file(obj, file_path):
    with open(file_path, 'w', encoding='utf-8') as fw:
        fw.write(str(obj))


def output_long_text_to_file(long_text, file_path, delimiters=None):
    with open(file_path, 'w', encoding='utf-8') as fw:
        long_text = str(long_text)
        if delimiters is None:
            delimiters = [',', '.', ';', '!', '?']
        elif isinstance(delimiters, list):
            pass
        elif isinstance(delimiters, str):
            delimiters = [delimiters]
        else:
            raise ValueError

        for punc in delimiters:
            long_text.replace(punc, punc + '\n')
        fw.write(long_text)


def file_to_lines(data_file: str):
    if not os.path.exists(data_file):
        print(f'Error: file_path {data_file} does not existes')
    with open(data_file, 'r', encoding='utf-8') as fr:
        return fr.readlines()


def lines_to_file(lines: list, data_file: str):
    _dir = os.path.dirname(data_file)
    if not os.path.exists(_dir):
        print(f'Error: file directory {_dir} does not existes')
    with open(data_file, 'w', encoding='utf-8') as fw:
        return fw.writelines(lines)


def pickle_load(path):
    """pickle.load(path)"""
    with open(path, "rb") as f:
        return pickle.load(f)


def pickle_save(obj, path):
    """pickle.dump(obj, path)"""
    with open(path, "wb") as f:
        return pickle.dump(obj, f)


def get_line_from_file(file_path, index):
    file_path = Path(file_path)
    if file_path.exists():
        raise FileNotFoundError(f"{file_path} not existing.")
    return linecache.getline(file_path, index)
