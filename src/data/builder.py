import copy
from dataclasses import fields
import torch
from typing import Iterable

from .sequences import Sequences
from .config import DataConfig, SequencesConfig
from general_utils import cli as cli_utils
from general_utils import reproducibility as reproducibility_utils
from general_utils import serialization as serialization_utils
from general_utils import fileio as fileio_utils

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

# def get_tokens(sequences, device):
#     return sequences.transform(
#         torch.cat(
#             (
#                 sequences.special_tokens['count']['token'].unsqueeze(0), 
#                 sequences.special_tokens['eos']['token'].unsqueeze(0)
#             ), 
#             dim=0
#         )
#     ).to(device)

def get_autoregressive_tokens(sequences):
    return sequences.transform.apply_deterministic_transform(
        torch.cat(
            (
                sequences.special_tokens['count']['token'].unsqueeze(0), 
                sequences.special_tokens['eos']['token'].unsqueeze(0)
            ), 
            dim=0
        )
    )

# def build_split_sequences(sequences_cfg_base, reproducibility_cfg, split_names, split_sizes, seed_ind):

#     lengths = [len(item) for item in [split_names, split_sizes, seed_ind]]
#     if len(set(lengths)) != 1:
#         raise ValueError(
#             "All input arguments must be of the same length but got: "
#             f"len(split_sizes)={len(split_sizes)}, "
#             f"len(split_names)={len(split_names)}, "
#             f"len(seed_ind)={len(seed_ind)}."
#         )
    
#     sequences_cfgs = [copy.deepcopy(sequences_cfg_base) for _ in range(lengths[0])]
    
#     splits = zip(sequences_cfgs, split_sizes, split_names, seed_ind)
#     sequences = {}
#     for (s_cfg, size, name, seed_idx) in splits:
#         s_cfg.num_seq = size
#         apply_reproducibility_settings(reproducibility_cfg, name, seed_idx)
#         sequences[name] = build_hypercube_sequences(s_cfg)
    
#     return sequences

# def build_split_sequences(sequences_cfg_base, reproducibility_cfg, split_cfgs):
#     """ 
#     """
#     sequences_cfgs = [copy.deepcopy(sequences_cfg_base) for _ in range(len(split_cfgs))]
    
#     sequences = {}
#     for spl_cfg, seq_cfg in zip(split_cfgs, sequences_cfgs):
#         seq_cfg.num_seq = spl_cfg.split_size
#         name = spl_cfg.split_name
#         seed_idx = spl_cfg.seed_idx
#         apply_reproducibility_settings(reproducibility_cfg, name, seed_idx)
#         sequences[name] = build_hypercube_sequences(seq_cfg)
    
#     return sequences

def build_split_sequences(data_cfg: DataConfig, build: Iterable[str], seed_idx: Iterable[int]):
    """ 
    """
    split_names = [f.name for f in fields(data_cfg.split_cfg)]
    num_splits = len(split_names)
    num_splits_to_build = len(build)

    # Validate args.
    for split in build:
        if split not in split_names:
            raise ValueError(
                f"Unrecognized value {split} in `build`. Must be in {split_names}."
            )
    if num_splits_to_build != len(set(build)):
        raise ValueError("`build` should not contain repeated elements.")
    if num_splits_to_build > num_splits:
        raise RuntimeError(
            f"Length of build ({len(build)}) exceeds the number of attributes "
            f"in data_cfg.split_cfg ({split_names})."
        )
    
    sequences_cfgs = [
        copy.deepcopy(data_cfg.sequences_cfg) for _ in range(num_splits_to_build)
    ]
    
    sequences = {}
    for split, seq_cfg in zip(build, sequences_cfgs):
        seq_cfg.num_seq = getattr(data_cfg.split_cfg, split)
        reproducibility_utils.apply_reproducibility_settings(
            data_cfg.reproducibility_cfg, 
            seed_idx=seed_idx,
            split=split
        )
        sequences[split] = build_hypercube_sequences(seq_cfg)
    
    return sequences


def build_sequences_from_filepath(
    data_cfg_filepath: str, 
    build: str, 
    seed_idx: int, 
    save_path: str = None,
    print_to_console: bool =True
):
    """ 
    """
    
    # Get config.
    data_cfg_dict = {}
    data_cfg_dict['base'] = serialization_utils.deserialize(data_cfg_filepath)
   
    # Apply recovery seed and recover.
    reproducibility_utils.apply_reproducibility_settings(
        data_cfg_dict['base'].reproducibility_cfg, 
        seed_idx=seed_idx,
        split='recovery'
    )
    data_cfg_dict['recovered'] = serialization_utils.recursive_recover(
        data_cfg_dict['base']
    )

    sequences = build_split_sequences(
        data_cfg_dict['recovered'],
        build=build,
        seed_idx=seed_idx
    )

    if print_to_console:
        for split, sequences_split in sequences.items():
            # Retrieve sample from current split.
            seq_idx = 0
            seq, labels, _, _, _ = sequences_split[seq_idx]

            # Get version where all labels in positive vs negative class are the same.
            labels_uniform_class = labels.clone()
            labels_uniform_class[labels >= 4] = 4
            labels_uniform_class[labels < 0] = -1

            print(sequences['train'].transform.lin_transform)
            print(sequences['val'].transform.lin_transform)

            print(f"Shape of single sequence from {split} split: {seq.shape}.")
            print(f"Shape of corresponding labels: {labels.shape}.")
            print(f"Sequence: \n {seq}.") 
            print(f"Labels by class: \n {labels_uniform_class}.") 
            print(f"Labels by stimulus: \n {labels}.") 
            print(f"Embedded count and EOS tokens from {split} split: \n {get_autoregressive_tokens(sequences[split])}.")
            print("\n")

    if save_path is not None:
        fileio_utils.torch_save(sequences, save_path)
        serialization_utils.serialize(data_cfg_dict['recovered'], save_path)

    return sequences, data_cfg_dict

