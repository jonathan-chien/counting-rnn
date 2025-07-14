import argparse
import ast
import copy
from dataclasses import is_dataclass


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

        retrieval_error_message = (
            "Unexpected condition reached during attempted traversal of cfg " 
            "tree. Expected a dataclass or dict instance for all branches, " 
            f"but got type {type(branch)} for {branch} in override string '{override}'."
        )

        # Iteratively work to leaf. 
        branch = cfg_copy
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
