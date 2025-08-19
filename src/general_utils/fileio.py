import json
import os
from pathlib import Path
from typing import Literal, Union

import torch


def get_dir(*path_parts):
    path = Path(*path_parts)
    if not path.exists():
            raise FileNotFoundError(
                f"Path {path} does not exist."
            )
    elif not path.is_dir():
        raise NotADirectoryError(
            f"Path {path} exists but is not a directory."
        )
    return path

def make_dir(*path_parts: Union[str, Path], chdir=False, parents=True, exist_ok=True):
    dir = Path(*path_parts) 
    dir.mkdir(parents=parents, exist_ok=exist_ok)
    if chdir: 
        os.chdir(dir)
    return dir

def make_filename(*file_ind, joiner='_'):
    return joiner.join(file_ind)
    
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
    
def get_filepath_with_suffix(dir, suffix, return_as='Path'):
    """ 
    """
    path = Path(dir)
    matches = list(path.glob(f'*{suffix}'))

    if not matches:
        raise FileNotFoundError(
            f"No file ending with '{suffix}' found in '{dir}'."
        )
    elif len(matches) > 1:
        raise RuntimeError(
            f"Multiple files ending with '{suffix}' found in '{dir}': ."
            f"{[str(m.name) for m in matches]}"
        )
    
    if return_as == 'str':
        out = str(matches[0])
    elif return_as != 'Path':
        raise ValueError(
            f"Unrecognized value {return_as} for return_as. Must be 'str' or 'Path'."
        )
    else:
        out = matches[0]
    
    return out
    
def get_filepaths(dir: Union[Path, str], return_as: Literal['Path', 'str'] = Path):

    dir = Path(dir)
    matches = [m for m in dir.glob('*') if m.is_file()]
    if return_as == 'str':
        return [str(m) for m in matches]
    elif return_as == 'Path':
        return matches
    else:
        raise ValueError(f"Unrecognized value '{return_as}' for `return_as`. Must be 'Path' or 'str'.")


        

