import torch

from .config import ReproducibilityConfig
from ..config.serialization import shallow_asdict
from .. import seed as seed_utils


def set_seed(torch_seed, cuda_seed):
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed(cuda_seed)

def set_torch_determinism(use_deterministic_algos, cudnn_deterministic, cudnn_benchmark):
    torch.use_deterministic_algorithms(use_deterministic_algos)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = cudnn_benchmark

def recursive_seed_sequence_spawn(
    parent, 
    num_children_per_level, 
    dtype='uint32', 
    return_as='int',
    depth=0, # Internal use during recursion, don't pass in manually
):
    """ 
    Sheer wrapper around recursive_seed_sequence_spawn from general_utils.seed
    to provide unified API.
    """
    return seed_utils.recursive_seed_sequence_spawn(
        parent=parent, 
        num_children_per_level=num_children_per_level, 
        dtype=dtype, 
        return_as=return_as,
        depth=depth, 
    )

def generate_seed_sequence(
    num_children_per_level, 
    entropy=None, 
    dtype='uint32', 
    return_as='int'
):
    """ 
    Sheer wrapper around generate_seed_sequence from general_utils.seed to 
    to provide unified API.
    """
    return seed_utils.generate_seed_sequence(
        num_children_per_level=num_children_per_level, 
        entropy=entropy, 
        dtype=dtype, 
        return_as=return_as
    )
       
def apply_reproducibility_settings(reproducibility_cfg: ReproducibilityConfig, seed_idx: int, split: str):
    """ 
    """
    set_seed(
        **shallow_asdict(reproducibility_cfg.seed_cfg_list[seed_idx][split])
    )
    set_torch_determinism(
        **shallow_asdict(reproducibility_cfg.torch_determinism_cfg_dict[split])
    )

