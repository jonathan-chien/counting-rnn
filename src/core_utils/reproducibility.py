import numpy as np
import torch


def set_seed(torch_seed, cuda_seed, use_deterministic_algos, cudnn_deterministic, cudnn_benchmark):
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed(cuda_seed)

    torch.use_deterministic_algorithms(use_deterministic_algos)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = cudnn_benchmark


def generate_numpy_seed_sequence(num_children=1, dtype=np.uint32, return_as='torch'):
    root_seed_seq = np.random.SeedSequence() # entropy=None
    entropy = root_seed_seq.entropy
    children = root_seed_seq.spawn(num_children)

    seeds = [child.generate_state(n_words=1, dtype=dtype)[0] for child in children]

    if return_as == 'torch':
        seeds = torch.from_numpy(seeds)
    elif return_as != 'numpy':
        raise ValueError(
            f"Unrecognized value {return_as} for `return_as`. Must be 'torch' "
            "or 'numpy'."
        )
    
    return seeds, entropy, root_seed_seq, children



# Could also set seed for random module and numpy if needed.