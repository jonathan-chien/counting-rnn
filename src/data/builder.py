import copy
from dataclasses import fields
import torch
from typing import Iterable

from .sequences import Sequences
from .config import DataConfig, SequencesConfig
from general_utils import config as config_utils
from general_utils import ml as ml_utils
from general_utils.ml.reproducibility import ReproducibilityConfig


def build_hypercube_sequences(sequences_cfg: SequencesConfig) -> Sequences:
    hypercube = sequences_cfg.elem
    embedder = sequences_cfg.embedder

    # Validate seq_lengths object.
    sequences_cfg.seq_lengths.validate()

    # Check that number of variables is large enough.
    if embedder.ambient_dim < hypercube.num_dims + 3: 
        raise ValueError(
            "`embedder.ambient_dim` must be at least 3 greater than hypercube " 
            f"dimensionality {hypercube.num_dims}, but got {embedder.ambient_dim}."
        )
    
    return Sequences(
        num_seq=sequences_cfg.num_seq,
        num_vars=hypercube.num_dims, # TODO: Eliminate this argument, see TODO in Sequences class
        len_distr=sequences_cfg.seq_lengths.lengths,
        elem_distr=hypercube.vertices,
        transform=embedder,
        seq_order=sequences_cfg.seq_order
    )

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

def build_split_sequences(
    data_cfg: DataConfig, 
    reproducibility_cfg: ReproducibilityConfig,
    build: Iterable[str], 
    seed_idx: Iterable[int]
):
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
        ml_utils.reproducibility.apply_reproducibility_settings(
            reproducibility_cfg, 
            seed_idx=seed_idx,
            split=split
        )
        sequences[split] = build_hypercube_sequences(seq_cfg)
    
    return sequences


def build_sequences_from_filepath(
    data_cfg_filepath: str, 
    reproducibility_cfg_filepath: str,
    build: str, 
    seed_idx: int, 
    save_path: str = None,
    print_to_console: bool =True
):
    """ 
    """
    # Get base data config, get and recover (recovery not currently necessary,
    # but more future proof) reproducibility config (recovery should be
    # deterministic).
    data_cfg_dict = {}
    data_cfg_dict['base'] = config_utils.serialization.deserialize(data_cfg_filepath)
    reproducibility_cfg_dict = {}
    reproducibility_cfg_dict['base'] = config_utils.serialization.deserialize(reproducibility_cfg_filepath)
    reproducibility_cfg_dict['recovered'] = config_utils.serialization.recursive_recover(
        reproducibility_cfg_dict['base']
    )

    # Apply recovery seed to deterministically recover all elements of data config.
    ml_utils.reproducibility.apply_reproducibility_settings(
        reproducibility_cfg_dict['recovered'], 
        seed_idx=seed_idx,
        split='recovery'
    )
    data_cfg_dict['recovered'] = config_utils.serialization.recursive_recover(
        data_cfg_dict['base']
    )

    # Apply split specific seeds to build data splits.
    sequences = build_split_sequences(
        data_cfg_dict['recovered'],
        reproducibility_cfg_dict['recovered'],
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
        torch.save(sequences, save_path)
        config_utils.serialization.serialize(data_cfg_dict['recovered'], save_path)

    return sequences, data_cfg_dict, reproducibility_cfg_dict

