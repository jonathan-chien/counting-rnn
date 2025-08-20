from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd


# ------------------------------ Boolean check ------------------------------- #
def is_int(x):
    return isinstance(x, int) 

def is_pos_int(x):
    return (is_int(x) and x > 0) 

def is_nonneg_int(x):
    return (is_int(x) and x >= 0)

def is_float(x):
    return isinstance(x, float)

def is_pos_float(x):
    return (is_float(x) and x > 0)

def is_nonneg_float(x):
    return (is_float(x) and x >= 0)

def is_str(x):
    return isinstance(x, str)

def is_pathlib_path(x):
    return isinstance(x, Path)

def is_pathlib_path_or_str(x):
    return is_pathlib_path(x) or is_str(x)

def is_dict(x):
    return isinstance(x, dict)

def is_list(x):
    return isinstance(x, list)

def is_tuple(x):
    return isinstance(x, tuple)

def is_dataframe(x):
    return isinstance(x, pd.DataFrame)

def is_iterable(x):
    return (isinstance(x, Iterable) and not isinstance(x, (str, bytes)))


# --------------------------- Validation (raises) ---------------------------- #
def error_message(expected: str, actual: Any):
    return f"Expected {expected}, but got type {type(actual).__name__}."

def validate_int(x):
    if not is_int(x):
        raise TypeError(error_message("an int", x))
    
def validate_pos_int(x):
    if not is_pos_int(x):
        raise ValueError(error_message("a positive int", x))
    
def validate_nonneg_int(x):
    if not is_nonneg_int(x):
        raise ValueError(error_message("a non-negative int", x))
    
def validate_float(x):
    if not is_float(x):
        return TypeError(error_message("a float", x))
    
def validate_pos_float(x):
    if not is_pos_float(x):
        return ValueError(error_message("a positive float", x))
    
def validate_nonneg_float(x):
    if not is_nonneg_float(x):
        return ValueError(error_message("a non-negative float", x))
    
def validate_str(x):
    if not is_str(x):
        raise TypeError(error_message("a string", x))
    
def validate_pathlib_path(x):
    if not is_pathlib_path(x):
        raise TypeError(error_message("an object of type pathlib.Path", x))
    
def validate_pathlib_path_or_str(x):
    if not (is_pathlib_path(x) or is_str(x)):
        raise TypeError(error_message("an object of type pathlib.Path, or a str", x))

def validate_iterable(x):
    """ 
    """
    if not is_iterable(x):
        raise TypeError(error_message("an iterable", x))
    
def validate_iterable_contents(x, predicate, expected_description: str):
    validate_iterable(x)
    for i_elem, elem in enumerate(x):
        if not predicate(elem):
            raise ValueError(
                f"Element at index {i_elem} failed validation: expected "
                f"{expected_description}, but got {elem}."
            )