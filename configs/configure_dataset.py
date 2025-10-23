import argparse
from dataclasses import dataclass, replace, field, fields
from datetime import date
from typing import Literal, Optional, Union
import warnings

import torch

from data import builder as data_builder 
from data.config import DataConfig, EmbedderConfig, HypercubeConfig, NormalDistrConfig, SeqLengths, SequencesConfig, SplitConfig
from data.sequences import Hypercube, Embedder
from data import utils as data_utils
from general_utils.config.types import CallableConfig, TensorConfig
from general_utils import config as config_utils
from general_utils import fileio as fileio_utils
from general_utils import tensor as tensor_utils


def build_arg_parser():
    parser = argparse.ArgumentParser(
        parents=[config_utils.ops.make_parent_parser()]
    )
    parser.add_argument('--idx', type=int, required=True, help="Zero-based integer to use for filename.")
    parser.add_argument('--zfill', type=int, default=4, help="Zero-pad width for filename (default=4).")
    parser.add_argument('--base_dir', default='configs/datasets')
    parser.add_argument('--sub_dir_1', default=str(date.today()))
    parser.add_argument('--sub_dir_2', default='a')
    
    return parser

def main():
    args = build_arg_parser().parse_args()

    # ---------------------------- Set directory ---------------------------- #
    base_dir = args.base_dir
    sub_dir_1 = args.sub_dir_1
    sub_dir_2 = args.sub_dir_2
    output_dir = fileio_utils.make_dir(base_dir, sub_dir_1, sub_dir_2)
    filename = str(args.idx).zfill(args.zfill)

    # Parse key-value pairs from the 'set' channel for runtime CLI injection.
    cli = config_utils.ops.parse_override_kv_pairs(args.ch0 or [])

    # ---------------------- Build hypercube config ------------------------- #
    # TODO: A better scheme may be to use a dataclass/dictionary to hold
    # defaults (e.g. for running the script directly without sweeping) and
    # create a function that allows retrieving all dotted paths distal to some
    # branch, e.g. pass 'hypercube' as an arg, and get all dotted paths with
    # 'hypercube', distal from 'hyercube'. Then use dotted path override
    # functionality to update from CLI. Thus, override occurs only if value was
    # passed in from CLI (note that None can be passed in from the CLI to
    # trigger logic to build defaults, even though it would probably be safer
    # to set this to None manually in the script). If a dataclass is used, need
    # to decide whether to define at the top or inline in the main function; or
    # could use dict inline in the main function. This is closer to how the 
    # seq_lengths config is currently handled. 
    # Deferring implenting this as it's not a current priority and enough time
    # has been spent on development for the time being.

    # Get CLI overrides first.
    num_hypercube_dims = config_utils.ops.select(cli, 'hypercube.num_dims', 3)
    manual_labels = config_utils.ops.select(cli, 'hypercube.manual_labels', True)
    random_labels = config_utils.ops.select(cli, 'hypercube.random_labels', False)
    manual_pmfs = config_utils.ops.select(cli, 'hypercube.manual_pmfs', False)

    # Local intermediates related to vertices.
    num_vertices = 2**num_hypercube_dims
    half = num_vertices // 2
    coords = torch.arange(num_hypercube_dims, dtype=torch.int64)
    if manual_labels:
        pos_vert_labels = torch.tensor([2, 3, 5, 7])
    else:
        if random_labels:
            g = torch.Generator()
            g.manual_seed(314159265)
            perm = torch.randperm(num_vertices, dtype=torch.int64)
            pos_vert_labels = perm[half:]
        else:
            pos_vert_labels = torch.arange(half, num_vertices)
    vertices = data_utils.get_lexicographic_ordering(
        num_hypercube_dims, 
        encoding=torch.tensor([0, 1]), 
        dtype=torch.int8
    )
    inclusion_set = vertices.index_select(0, pos_vert_labels)

    # Local intermediates related to PMFs.
    if manual_pmfs:
        pos_vert_pmf = torch.tensor([0.5, 0.5])
        neg_vert_pmf = torch.tensor([0.5, 0.5])
    else:
        pos_vert_pmf = data_utils.uniform_pmf(len(pos_vert_labels))
        neg_vert_pmf = data_utils.uniform_pmf(num_vertices-len(pos_vert_labels))

    # Build hypercube config object.
    hypercube_args_cfg = HypercubeConfig(
        num_dims=num_hypercube_dims,
        coords=TensorConfig.from_tensor(coords),
        inclusion_set=TensorConfig.from_tensor(inclusion_set),
        encoding=TensorConfig.from_tensor(torch.tensor([0, 1], dtype=torch.int8)),
        vertices_pmfs=(
            TensorConfig.from_tensor(pos_vert_pmf),
            TensorConfig.from_tensor(neg_vert_pmf)
        )
    )

    # ------------------------ Build seq length config ---------------------- #
    # Get CLI injectables.
    seq_lengths_helper = {
        'pos': {
            'max': config_utils.ops.select(cli, 'seq_lengths.pos.support.max', 19),
            'parity': config_utils.ops.select(cli, 'seq_lengths.pos.parity', 'odd'),
            'support': None,
            'pmf': None
        }, 
        'neg': {
            'max': config_utils.ops.select(cli, 'seq_lengths.neg.support.max', 9),
            'parity': config_utils.ops.select(cli, 'seq_lengths.neg.parity', 'odd'),
            'support': None,
            'pmf': None
        }
    }

    for class_ in ('pos', 'neg'):
        # Shorter reference to current values.
        parity = seq_lengths_helper[class_]['parity']
        max_length = seq_lengths_helper[class_]['max']
        support = seq_lengths_helper[class_]['support']
        pmf = seq_lengths_helper[class_]['pmf']

        # Validate parity.
        if parity not in ('odd', 'even', None):
            raise ValueError(
                f"Unrecognized value {parity} for class_ {class_}. Must be one of 'odd', 'even', or None."
            )
        
        # Construct support.
        if support is None:
            if max_length is not None:
                support = (
                    tensor_utils.single_parity_arange(max_length, parity)
                    if parity in ('odd', 'even')
                    else torch.arange(max_length) 
                )
            else:
                raise ValueError(
                    f"No sequence length support was provided for class_ '{class_}', "
                    "so a max length must be specified, but got None."
                )
        else:
            tensor_utils.validate_tensor(support, 1)
            
        # If no PMF specified, use uniform distribution.
        if pmf is None:
            pmf = data_utils.uniform_pmf(len(support))
        else:
            data_utils.validate_pmf(pmf, support_size=len(support))

        # Wrap as TensorConfig object.
        seq_lengths_helper[class_]['support'] = TensorConfig.from_tensor(support)
        seq_lengths_helper[class_]['pmf'] = TensorConfig.from_tensor(pmf)

    # Build config.
    seq_lengths = SeqLengths(
        lengths={
            class_: {
                key: seq_lengths_helper[class_][key]
                for key in ('support', 'pmf')
            }
            for class_ in ('pos', 'neg') 
        }
    )

    
    # --------------- Build remaining auxiliary object configs -------------- #
    embedder_cfg = EmbedderConfig(
        ambient_dim=6,
        mean_center=False,
        offset_1=TensorConfig.from_tensor(-torch.tile(torch.tensor([0.5]), (hypercube_args_cfg.num_dims + 3,))), # Plus 3 for the dimensions corresponding to special tokens
        offset_2=None,
        method='random_rotation',
        noise_distr=CallableConfig.from_callable(
            torch.distributions.Normal, 
            NormalDistrConfig(loc=0, scale=0.01),
            kind='class',
            recovery_mode='call'
        )
    )

    elem=CallableConfig.from_callable(
            Hypercube, 
            hypercube_args_cfg, 
            kind='class', 
            recovery_mode='call'
        )
    
    embedder=CallableConfig.from_callable(
            Embedder,
            embedder_cfg,
            kind='class',
            recovery_mode='call'
        )

    # ---------------------------- Sequences config -------------------------- #
    sequences_cfg = SequencesConfig(
        num_seq='num_seq___',
        seq_order='permute',
        seq_lengths=seq_lengths,
        elem=elem,
        embedder=embedder
    )

    # ----------------------------- Split sizes ------------------------------ #
    split_cfg = SplitConfig(
        train=5000,
        val=2500,
        test=2500
    )

    # --------------------------- Top level config --------------------------- #
    data_cfg = DataConfig(
        sequences_cfg=sequences_cfg,
        split_cfg=split_cfg
    )

    # -------------------------- Apply CLI overrides ------------------------- #
    data_cfg = config_utils.ops.apply_cli_override(data_cfg, args.ch1 or [])

    # ------------------------------ Serialize ------------------------------- #
    # Attempt to serialize and reconstruct full cfg tree, and use reconstructed
    # version to build, to ensure future reproducibility.
    data_cfg_filepath = output_dir / (filename + '.json')
    _ = config_utils.serialization.serialize(data_cfg, data_cfg_filepath)

    # -------------------- Test deserialization/execution -------------------- #
    # Attempt to build dataset from serialized file.
    data_builder.build_sequences_from_filepath(
        data_cfg_filepath, 
        build=['train', 'val'], 
        reproducibility_cfg_filepath='configs/reproducibility/0000-00-00/a/0000.json',
        seed_idx=0, 
        print_to_console=True, 
        save_path=None
    )

    # --------------------------- Summarize config --------------------------- #
    # Registry of items to extract from the config.
    REGISTRY = {
        'hypercube_dim': 'sequences_cfg.elem.args_cfg.num_dims',
        'pos_vertices': 'sequences_cfg.elem.args_cfg.inclusion_set.args_cfg.data',
        'pos_vertices_pmf': 'sequences_cfg.elem.args_cfg.vertices_pmfs.0.args_cfg.data',
        'neg_vertices_pmf': 'sequences_cfg.elem.args_cfg.vertices_pmfs.1.args_cfg.data',
        'pos_seq_lengths': 'sequences_cfg.seq_lengths.lengths.pos.support.args_cfg.data',
        'pos_seq_lengths_pmf': 'sequences_cfg.seq_lengths.lengths.pos.pmf.args_cfg.data',
        'neg_seq_lengths': 'sequences_cfg.seq_lengths.lengths.neg.support.args_cfg.data',
        'neg_seq_lengths_pmf': 'sequences_cfg.seq_lengths.lengths.neg.pmf.args_cfg.data',
        'train_size': 'split_cfg.train',
        'val_size': 'split_cfg.val',
        'test_size': 'split_cfg.test'
    }

    # Deserialize and summarize config to .xlsx file.
    config_utils.summary.summarize_cfg_to_xlsx(
        data_cfg_filepath, 
        config_kind='datasets', 
        config_id=str(data_cfg_filepath).removeprefix('configs/datasets/').removesuffix('.json'),
        dotted_path_registry=REGISTRY,
        note='',
        xlsx_filepath='configs/logs.xlsx'
    )

if __name__ == '__main__':
    main()
