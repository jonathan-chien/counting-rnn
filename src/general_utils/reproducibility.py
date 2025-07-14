import numpy as np
import torch

from .config import ReproducibilityConfig
from .serialization import shallow_asdict
from . import validation as validation_utils


def set_seed(torch_seed, cuda_seed):
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed(cuda_seed)

def set_torch_determinism(use_deterministic_algos, cudnn_deterministic, cudnn_benchmark):
    torch.use_deterministic_algorithms(use_deterministic_algos)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = cudnn_benchmark

# def generate_numpy_seed_sequence(num_children=1, num_words_per_child=1, dtype=np.uint32, return_as='torch'):
#     """ 
#     """
#     root_seed_seq = np.random.SeedSequence() # entropy=None
#     entropy = root_seed_seq.entropy
#     children = root_seed_seq.spawn(num_children)

#     # seeds is a list of numpy arrays, each of length num_words_per_child.
#     seeds = [
#         child.generate_state(n_words=num_words_per_child, dtype=dtype) 
#         for child in children
#     ]

#     if return_as == 'int':
#         seeds = [s.tolist() for s in seeds]
#     elif return_as != 'numpy':
#         raise ValueError(
#             f"Unrecognized value {return_as} for `return_as`. Must be 'int' "
#             "or 'numpy'."
#         )
    
#     return seeds, entropy, root_seed_seq, children

# def apply_reproducibility_settings(cfg: ReproducibilityConfig, split: str, seed_idx: int):
#     """ 
#     """
#     set_seed(
#         **shallow_asdict(cfg.seed_cfg_dict[split][seed_idx])
#     )
#     set_torch_determinism(
#         **shallow_asdict(cfg.torch_determinism_cfg_dict[split])
#     )




# def generate_seed_sequence(parent, num_children_per_level, dtype, depth=None):
    
#     validation_utils.validate_iterable_of_ints(num_children_per_level)
#     if not (depth is None or isinstance(depth, int)):
#         raise RuntimeError("`depth` should not be passed manually."
#         )

#     depth = 0 if depth is None else depth + 1

#     if depth > len(num_children_per_level):
#         # Base case.
#         return parent.generate_state(n_words=1, dtype=dtype) 
#     elif depth <= len(num_children_per_level):
#         # Recursion case.
#         children = parent.spawn(num_children_per_level[depth])
#         return [
#             generate_seed_sequence(
#                 parent=child, 
#                 num_children_per_level=num_children_per_level, 
#                 dtype=dtype, 
#                 depth=depth
#             )
#             for child in children
#         ]
#     else:
#         raise RuntimeError(
#             f"Unexpected value of {depth} for depth counter `depth`. "
#             "Should not exceed length of num_children_per_level "
#             f"({len(num_children_per_level)})."
#         )
    
def recursive_seed_sequence_spawn(
    parent, 
    num_children_per_level, 
    dtype='uint32', 
    return_as='int',
    depth=0, # Internal use during recursion, don't pass in manually
):
    """ 
    Recursively creates arbitrarily deeply nested lists of seeds, where each
    level of the hierarchy is extended using the spawn method of the 
    SeedSequence class. Base case results in a seed returned (either as int or 
    numpy array), whereas all other calls return lists.
    """
    validation_utils.validate_iterable_contents(
        num_children_per_level,
        predicate=validation_utils.is_pos_int,
        expected="a positive int"
    )
    validation_utils.validate_nonneg_int(depth)
    
    if depth == len(num_children_per_level):
        # Base case.
        seed = parent.generate_state(n_words=1, dtype=dtype)
        if return_as == 'int':
            seed = int(seed[0])
        elif return_as != 'numpy':
            raise ValueError(
                "Invalid value for `return_as`: expected 'int' or 'numpy' "
                f"but got {return_as}."
            )
        return seed
    elif depth < len(num_children_per_level):
        # Recursion case.
        children = parent.spawn(num_children_per_level[depth])
        return [
            recursive_seed_sequence_spawn(
                parent=child, 
                num_children_per_level=num_children_per_level, 
                dtype=dtype, 
                return_as=return_as,
                depth=depth+1
            )
            for child in children
        ]
    else:
        raise ValueError(
            f"Invalid value for `depth`: got {depth} but expected at most "
            f"{len(num_children_per_level)}."
        )
    
def generate_seed_sequence(
    num_children_per_level, 
    entropy=None, 
    dtype='uint32', 
    return_as='int'
):
    """ 
    """
    root_seed_seq = np.random.SeedSequence(entropy=entropy) 

    seeds = recursive_seed_sequence_spawn(
        root_seed_seq,
        num_children_per_level,
        dtype=dtype,
        return_as=return_as
    )

    return seeds, root_seed_seq, root_seed_seq.entropy

       
def apply_reproducibility_settings(cfg: ReproducibilityConfig, seed_idx: int, split: str):
    """ 
    """
    set_seed(
        **shallow_asdict(cfg.seed_cfg_list[seed_idx][split])
    )
    set_torch_determinism(
        **shallow_asdict(cfg.torch_determinism_cfg_dict[split])
    )

