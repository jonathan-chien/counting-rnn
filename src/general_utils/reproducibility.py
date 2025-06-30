import numpy as np
import torch

from .serialization import shallow_asdict
from .config import ReproducibilityConfig


def set_seed(torch_seed, cuda_seed):
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed(cuda_seed)

def set_torch_determinism(use_deterministic_algos, cudnn_deterministic, cudnn_benchmark):
    torch.use_deterministic_algorithms(use_deterministic_algos)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = cudnn_benchmark

def generate_numpy_seed_sequence(num_children=1, num_words_per_child=1, dtype=np.uint32, return_as='torch'):
    """ 
    """
    root_seed_seq = np.random.SeedSequence() # entropy=None
    entropy = root_seed_seq.entropy
    children = root_seed_seq.spawn(num_children)

    # seeds is a list of numpy arrays, each of length num_words_per_child.
    seeds = [
        child.generate_state(n_words=num_words_per_child, dtype=dtype) 
        for child in children
    ]

    if return_as == 'int':
        seeds = [s.tolist() for s in seeds]
    elif return_as != 'numpy':
        raise ValueError(
            f"Unrecognized value {return_as} for `return_as`. Must be 'int' "
            "or 'numpy'."
        )
    
    return seeds, entropy, root_seed_seq, children

def apply_reproducibility_settings(cfg: ReproducibilityConfig, split: str, seed_idx: int):
    """ 
    """
    set_seed(
        **shallow_asdict(cfg.seed_cfg_dict[split][seed_idx])
    )
    set_torch_determinism(
        **shallow_asdict(cfg.torch_determinism_cfg_dict[split])
    )
