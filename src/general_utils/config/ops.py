import argparse
import ast
import copy
from dataclasses import is_dataclass

from .. import validation as validation_utils


def traverse_dotted_path(root, dotted_path: str):

    validation_utils.validate_str(dotted_path)
    if dotted_path == '':
        return root
    
    dotted_path_parts = dotted_path.split('.')

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
        if branch is None:
            # Raise exception if None encountered on a non-leaf node.
            raise ValueError(
                "None value encountered for branch when attempting to access with " 
                f"{part} at position {i_part} in dotted_path: '{dotted_path}'."
            )

        if is_dataclass(branch):
            branch = getattr(branch, part)
        elif isinstance(branch, dict):
            branch = branch[part]
        elif isinstance(branch, (list, tuple)):
            # try:
            #     idx = int(part)
            #     validation_utils.validate_nonneg_int(idx)
            # except ValueError:
            #     raise ValueError(
            #         "Expected a str representation of a non-negative integer at "
            #         f"position {i_part} in dotted path: '{dotted_path}', but got {part}."
            #     )
            # branch = branch[part]
            if part == '*':
                # Wildcard, traverse for all items in iterable.
                remainder = '.'.join(dotted_path_parts[i_part+1:])
                if remainder == '':
                    # If no more parts, return the whole iterable.
                    return list(branch) if isinstance(branch, tuple) else branch
                else:
                    # Otherwise, recursively traverse the rest of the path.
                    return [traverse_dotted_path(item, remainder) for item in branch]
            else:
                # If part is not wildcard, assume it is an index.
                try:
                    idx = int(part)
                except ValueError:
                    raise ValueError(
                        "Expected a str representation of an int at position "
                        f"{i_part} in dotted path: '{dotted_path}', but got {part}."
                    )
                try:
                    branch = branch[idx]
                except IndexError:
                    raise IndexError(
                        f"Index {idx} out of bounds for branch at position {i_part} in "
                        f"dotted path: '{dotted_path}'. Branch has length {len(branch)}."
                    )
                # try:
                #     part = ast.literal_eval(part)
                # except (ValueError, SyntaxError):
                #     raise ValueError(
                #         f"Expected a str representation of a list of non-negative integers at "
                #         f"position {i_part} in dotted path: '{dotted_path}', but got {part}."   
                #     )
                # branch = [traverse_dotted_path(item, '.'.join(dotted_path_parts[i_part+1:])) for item in branch[part]]
        else:
            incorrect = 'root' if i_part == 0 else f'branch {i_part-1}'
            raise RuntimeError(
                "Unexpected condition reached during attempted traversal of " 
                f"root object according to dotted path: '{dotted_path}'. " 
                "Expected a dataclass, dict, or iterable for root and all branches "
                f"but got type {type(branch)} for {incorrect}. This resulted in "
                f"this exception being raised at position {i_part} in the dotted path."
            )
        
    return branch
        
def parse_override_kv_pairs(override_kv_pair_list):
    """
    Parse command line overrides in the form of a list/tuple of [KEY, VALUE] pairs, with graceful fallback for true string values. 
    TODO: Use this in apply_cli_override function.
    # """
    # d = {}
    # for override in override_list:
    #     validation_utils.validate_str(override)
    #     if override.count('=') != 1:
    #         raise ValueError(
    #             "Overrides must contain exactly one '=', as in e.g. a=2; " 
    #             f"however, got {override}." 
    #         )
    #     key, val_str = override.split('=') 
    #     try:
    #         d[key] = ast.literal_eval(val_str)
    #     except (ValueError, SyntaxError):
    #         d[key] = val_str
    # print(list(d.items()))
    # return d
    d = {}

    if not override_kv_pair_list:
        return d
    
    for kv_pair in override_kv_pair_list:
        if not (isinstance(kv_pair, (list, tuple)) and len(kv_pair) == 2):
            raise ValueError(
                "Override must take the form of a KEY, VALUE pair "
                f"(list/tuple of length 2) but got {kv_pair!r}."
            )
        
        key, val_str = kv_pair
        try:
            d[key] = ast.literal_eval(val_str)
        except (ValueError, SyntaxError):
            d[key] = val_str

    return d

def select(pre_map, dotted_key, default):
    return pre_map.get(dotted_key, default)

def make_parent_parser():
    """ 
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--pre', nargs=2, action='append', default=[], help="Pre config construction overrides.")
    parser.add_argument('--set', nargs=2, action='append', default=[], help="Post config construction overrides.")
    parser.add_argument('--idx', type=int, required=True, help="Zero-based integer to use for filename.")
    parser.add_argument('--zfill', type=int, default=4, help="Zero-pad width for filename (default=4).")
    # parser.add_argument('--sweep_pre', action='append', default=[])
    # parser.add_argument('--sweep_set', action='append', default=[])
    return parser

def apply_cli_override(cfg, override_kv_pair_list, raise_if_not_exist=True):
    """ 
    """
    def format_path(path_string: str):
        """ 
        If we want e.g. data_cfg.field_1 = 2, we'd pass cfg=data_cfg,
        override_list='--set field_1 = 2', resulting in an empty string for
        dotted_path_to_final_branch. This is intended, as the
        traverse_dotted_path function will then return the root object, here
        data_cfg. For clearer error messages, the name '<root>' will be returned
        instead of the empty string.
        """
        validation_utils.validate_str(path_string)
        return '<root>' if path_string == '' else path_string

    cfg_copy = copy.deepcopy(cfg)

    kv_dict = parse_override_kv_pairs(override_kv_pair_list)
    for key, value in kv_dict.items():
        # Get dotted path to final branch and leaf name. 
        validation_utils.validate_str(key)
        dotted_path_to_leaf, leaf_value = key, value
        dotted_path_to_leaf_parts = dotted_path_to_leaf.split('.') 
        dotted_path_to_final_branch = '.'.join(dotted_path_to_leaf_parts[:-1]) 
        leaf_name = dotted_path_to_leaf_parts[-1]
        
        # Disallow wildcard *, support for which hasn't been implemented yet.
        if '*' in dotted_path_to_leaf_parts:
            raise ValueError(
                "Wildcard '*' detected in dotted path to final branch: "
                f"'{dotted_path_to_leaf}'. This is not currently supported, "
                "as it would require broadcast setting logic, which hasn't been implemented yet."
            )
        
        # # Get leaf value.
        # try:
        #     leaf_value = ast.literal_eval(value_string) 
        # except (ValueError, SyntaxError):
        #     # Gracefully fall back to using string for actual string inputs.
        #     leaf_value = value_string

        # Traverse to final branch.
        final_branch = traverse_dotted_path(cfg_copy, dotted_path_to_final_branch)

        # Set leaf value.
        if is_dataclass(final_branch):
            if not hasattr(final_branch, leaf_name) and raise_if_not_exist:
                raise AttributeError(
                    f"Dataclass at '{format_path(dotted_path_to_final_branch)}' has no field '{leaf_name}'."
                )
            setattr(final_branch, leaf_name, leaf_value)
        elif isinstance(final_branch, dict):
            if leaf_name not in final_branch and raise_if_not_exist:
                raise KeyError(
                    f"Key '{leaf_name}' not in dict at '{format_path(dotted_path_to_final_branch)}'."
                )
            final_branch[leaf_name] = leaf_value
        elif isinstance(final_branch, list):
            try:
                idx = int(leaf_name)
            except ValueError:
                raise ValueError(
                    f"List or tuple at {format_path(dotted_path_to_final_branch)} requires "
                    f"an int index, but got {leaf_name} while attempting to set override {key} {value}."
                )
            if not(-len(final_branch) <= idx < len(final_branch)):
                raise IndexError(
                    f"Index {idx} out of bounds for list of length {len(final_branch)} "
                    f"at {format_path(dotted_path_to_final_branch)} while attempting to set override {key} {value}."
                )
            final_branch[idx] = leaf_value
        elif isinstance(final_branch, tuple):
            raise TypeError(
                f"Final branch in '{format_path(dotted_path_to_final_branch)}' for key value pair {key} {value} "
                "is a tuple. Support for tuples has not yet been added (requires "
                "rebuilding due to their immutability)."
            )
        else:
            raise RuntimeError(
                f"Unexpected condition reached during attempted setting of leaf value {value} for key {key}. " 
                f"Got final branch {final_branch} of type {type(final_branch)} "
                "but currently supported types are dataclass, dict, list, and tuple."
            )

    return cfg_copy
# def apply_cli_override(cfg, override_list):
#     """ 
#     """
#     cfg_copy = copy.deepcopy(cfg)

#     for override in override_list:
#         if override.count('=') != 1:
#             raise ValueError(
#                 "Overrides passed in with the '--set' flag must contain " 
#                 f"exactly one '=', as in e.g. a=2; however, got {override}." 
#             )
        
#         dotted_cfg_path_string, value_string = override.split('=')
#         dotted_cfg_path_string_parts = dotted_cfg_path_string.split('.', -1)
#         try:
#             value = ast.literal_eval(value_string) 
#         except (ValueError, SyntaxError):
#             # Gracefully fall back to using string for actual string inputs.
#             value = value_string

#         # Iteratively work to leaf. branch is referenced in error message so it must be defined first.
#         branch = cfg_copy
#         retrieval_error_message = (
#             "Unexpected condition reached during attempted traversal of cfg " 
#             "tree. Expected a dataclass or dict instance for all branches, " 
#             f"but got type {type(branch)} for {branch} in override string '{override}'."
#         )
#         for part in dotted_cfg_path_string_parts[:-1]:
#             if is_dataclass(branch):
#                 branch = getattr(branch, part)
#             elif isinstance(branch, dict):
#                 branch = branch[part]
#             else:
#                 raise RuntimeError(retrieval_error_message)

#         # Set leaf value.
#         leaf = dotted_cfg_path_string_parts[-1]
#         if is_dataclass(branch):
#             setattr(branch, leaf, value)
#         elif isinstance(branch, dict):
#             branch[leaf] = value
#         else:
#             raise RuntimeError(retrieval_error_message)

#     return cfg_copy

def parse_and_apply_cli_overrides(cfg):
    """ 
    """
    parser = make_parent_parser()
    args = parser.parse_args()
    override_list = args.set or []
    return apply_cli_override(cfg, override_list), override_list

