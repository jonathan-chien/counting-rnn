import torch

from .sequences import Sequences
from .config import SequencesConfig


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
