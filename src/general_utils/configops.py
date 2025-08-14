import argparse
import ast
import copy
from dataclasses import is_dataclass


def traverse_dotted_path(root, dotted_path: str):

    if dotted_path == '':
        raise ValueError("Empty dotted path.")
    
    dotted_path_parts = dotted_path.split('.')

    # # Validate.
    
    # if '' in dotted_path_parts:
    #     raise ValueError(
    #         f"'' detected in dotted path: '{dotted_path}'. Check for double dot typos, e.g. 'a..b.c'."
    #     )
    # if any(part.strip() != part for part in dotted_path_parts):
    #     raise ValueError(
    #         f"Empty space detected in one of the path parts: {dotted_path}."
    #     )
    # if any(part is None for part in dotted_path_parts):
    #     raise ValueError(
    #         f"None "
    #     )

    branch = root 
    for i_part, part in enumerate(dotted_path_parts):
        # Validate during iteration so exact failure point can be output.
        if part == '':
            raise ValueError(
                f"'' detected for {part} at position {i_part} in dotted path: "
                f"'{dotted_path}'. Check for double dot typos, e.g. 'a..b.c'."
            )
        if part.strip() != part:
            raise ValueError(
                f"Empty space detected in for {part} at position {i_part} in " 
                f"dotted_path: '{dotted_path}'."
            )
        if part is None:
            raise ValueError(
                f"None value encounted for {part} at position {i_part} in " 
                f"dotted_path: '{dotted_path}'."
            )

        if is_dataclass(branch):
            branch = getattr(branch, part)
        elif isinstance(branch, dict):
            branch = branch[part]
        else:
            incorrect = 'root' if i_part == 0 else f'branch {i_part-1}'
            raise RuntimeError(
                "Unexpected condition reached during attempted traversal of " 
                f"root object according to dotted path: '{dotted_path}'. " 
                "Expected a dataclass or dict instance for root and all branches "
                f"but got type {type(branch)} for {incorrect}. This resulted in "
                f"this exception being raised at position {i_part} in the dotted path."
            )
        
        return branch
        

def get_parser():
    """ 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--set', action='append', default=[])
    return parser

def apply_cli_override(cfg, override_list):
    """ 
    """
    cfg_copy = copy.deepcopy(cfg)

    for override in override_list:
        if override.count('=') != 1:
            raise ValueError(
                "Overrides passed in with the '--set' flag must contain " 
                f"exactly one '=', as in e.g. a=2; however, got {override}." 
            )
        
        dotted_cfg_path_string, value_string = override.split('=')
        dotted_cfg_path_string_parts = dotted_cfg_path_string.split('.', -1)
        try:
            value = ast.literal_eval(value_string) 
        except (ValueError, SyntaxError):
            # Gracefully fall back to using string for actual string inputs.
            value = value_string

        # Iteratively work to leaf. branch is referenced in error message so it must be defined first.
        branch = cfg_copy
        retrieval_error_message = (
            "Unexpected condition reached during attempted traversal of cfg " 
            "tree. Expected a dataclass or dict instance for all branches, " 
            f"but got type {type(branch)} for {branch} in override string '{override}'."
        )
        for part in dotted_cfg_path_string_parts[:-1]:
            if is_dataclass(branch):
                branch = getattr(branch, part)
            elif isinstance(branch, dict):
                branch = branch[part]
            else:
                raise RuntimeError(retrieval_error_message)

        # Set leaf value.
        leaf = dotted_cfg_path_string_parts[-1]
        if is_dataclass(branch):
            setattr(branch, leaf, value)
        elif isinstance(branch, dict):
            branch[leaf] = value
        else:
            raise RuntimeError(retrieval_error_message)

    return cfg_copy

def parse_and_apply_cli_overrides(cfg):
    """ 
    """
    parser = get_parser()
    args = parser.parse_args()
    override_list = args.set or []
    return apply_cli_override(cfg, override_list), override_list

