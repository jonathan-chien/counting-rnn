from datetime import datetime
import getpass
from pathlib import Path
import platform
import socket
import subprocess
import textwrap
from typing import Union
import warnings

from . import fileio as fileio_utils


def collect_metadata(additional_info: dict, short=False, enforce_clean_git_tree=True):
    """ 
    """
    metadata = {
        'datetime': datetime.now().isoformat(),
        'hostname': socket.gethostname(),
        'user': getpass.getuser(),
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'git_version': get_commit_hash(short=short, enforce_clean_git_tree=enforce_clean_git_tree),
        'git_branch': get_git_branch(),
        'git_clean': is_git_tree_clean(),
    }
    metadata.update(additional_info)

    return metadata

def collect_and_save_metadata(
    additional_info: dict, 
    filepath: Union[str, Path], 
    short=False, 
    enforce_clean_git_tree=True, 
    overwrite=False
):
    """
    """
    metadata = collect_metadata(
        additional_info=additional_info, 
        short=short, 
        enforce_clean_git_tree=enforce_clean_git_tree
    )

    filepath = Path(filepath)
    if filepath.exists() and not overwrite:
        raise RuntimeError(
            f"{filepath} already exists (`overwrite`=False)!"
        )
    
    fileio_utils.save_to_json(metadata, filepath)

    return metadata

def is_git_tree_clean():
    """ 
    """
    result = subprocess.run(
        ['git', 'status', '--porcelain'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return result.stdout.strip() == ''

def get_git_branch():
    """ 
    """
    result = subprocess.run(
        ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
        stdout=subprocess.PIPE,
        text=True
    )
    return result.stdout.strip()

def get_commit_hash(short=False, enforce_clean_git_tree=True):
    """ 
    """
    if not is_git_tree_clean():
        if enforce_clean_git_tree:
            raise RuntimeError(
                "Git tree is dirty (enforce_clean_git_tree=True)."
            )
        else:
            warnings.warn("Git Tree is dirty!")
        
    result = subprocess.run(
        ['git', 'rev-parse', '--short' if short else 'HEAD'],
        stdout=subprocess.PIPE, 
        text=True
    )
     
    return result.stdout.strip()

def create_textfile(content: str, filepath: Union[str, Path] = 'README.md', dedent=False, overwrite=False):
    """ 
    """
    filepath = Path(filepath)
    if filepath.exists() and not overwrite:
        raise FileExistsError(
            f"The file {filepath} already exists (`overwrite` = False)."
        )

    if dedent:
        content = textwrap.dedent(content)

    filepath.write_text(content, encoding='utf-8')
    
    return filepath
