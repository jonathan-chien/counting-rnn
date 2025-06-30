import copy
import torch

from .sequences import Sequences
from .config import SequencesConfig
from general_utils.reproducibility import apply_reproducibility_settings


def build_hypercube_sequences(cfg: SequencesConfig) -> Sequences:
    hypercube = cfg.elem
    embedder = cfg.embedder

    # Validate seq_lengths object.
    cfg.seq_lengths.validate()

    # Check that number of variables is large enough.
    if embedder.ambient_dim < hypercube.num_dims + 3: 
        raise ValueError(
            "`embedder.ambient_dim` must be at least 3 greater than hypercube " 
            f"dimensionality {hypercube.num_dims}, but got {embedder.ambient_dim}."
        )
    
    return Sequences(
        num_seq=cfg.num_seq,
        num_vars=hypercube.num_dims, # TODO: Eliminate this argument, see TODO in Sequences class
        len_distr=cfg.seq_lengths.lengths,
        elem_distr=hypercube.vertices,
        transform=embedder,
        seq_order=cfg.seq_order
    )

def get_tokens(sequences, device):
    return sequences.transform(
        torch.cat(
            (
                sequences.special_tokens['count']['token'].unsqueeze(0), 
                sequences.special_tokens['eos']['token'].unsqueeze(0)
            ), 
            dim=0
        )
    ).to(device)

def build_split_sequences(sequences_cfg_base, reproducibility_cfg, split_names, split_sizes, seed_ind):

    lengths = [len(item) for item in [split_names, split_sizes, seed_ind]]
    if len(set(lengths)) != 1:
        raise ValueError(
            "All input arguments must be of the same length but got: "
            f"len(split_sizes)={len(split_sizes)}, "
            f"len(split_names)={len(split_names)}, "
            f"len(seed_ind)={len(seed_ind)}."
        )
    
    sequences_cfgs = [copy.deepcopy(sequences_cfg_base) for _ in range(lengths[0])]
    
    splits = zip(sequences_cfgs, split_sizes, split_names, seed_ind)
    sequences = {}
    for (s_cfg, size, name, seed_idx) in splits:
        s_cfg.num_seq = size
        apply_reproducibility_settings(reproducibility_cfg, name, seed_idx)
        sequences[name] = build_hypercube_sequences(s_cfg)
    
    return sequences

