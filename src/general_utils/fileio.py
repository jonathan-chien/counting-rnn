import json
import os
from pathlib import Path
from typing import Union

import torch


def make_dir(*path_parts: Union[str, Path], chdir=False):
    dir = Path(*path_parts) 
    dir.mkdir(parents=True, exist_ok=True)
    if chdir: os.chdir(dir)
    return dir

def make_file_dir_and_id(base_dir, sub_dir_1, sub_dir_2, file_ind, chdir=False):
    """ 
    Utility for creating directory and filename for the 
    configs/base_dir/sub_dir_1/sub_dir_2/id1_id2_...idn.json format.
    """
    dir = make_dir(base_dir, sub_dir_1, sub_dir_2, chdir=chdir) 
    filename = '_'.join(file_ind)
    return dir, filename
    
def save_to_json(x, filepath, indent=2):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(x, f, indent=indent)

def load_from_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)
    
def torch_save(x, filepath):
    with open(filepath, 'wb') as f:
        torch.save(x, f)

def torch_load(filepath):
    with open(filepath, 'rb') as f:
        return torch.load(f)




        

