import argparse
from dataclasses import dataclass, replace, field, fields
from datetime import date
from typing import Literal, Optional, Union

import torch

from data import builder as data_builder 
from data.config import DataConfig, EmbedderConfig, HypercubeConfig, NormalDistrConfig, SeqLengths, SequencesConfig, SplitConfig
from data.sequences import Hypercube, Embedder
from data import utils as data_utils
from general_utils.config.types import CallableConfig, ContainerConfig, TensorConfig
from general_utils import config as config_utils
from general_utils import ml as ml_utils
from general_utils import fileio as fileio_utils
from general_utils import tensor as tensor_utils
from general_utils import validation as validation_utils


# REPRODUCIBILITY_CFG_FILEPATH = 'configs/reproducibility/2025-08-11/a/0000.json'

def build_arg_parser():
    parser = argparse.ArgumentParser(
        parents=[config_utils.ops.get_parser()],
    )
    parser.add_argument('--index', type=int, required=True, help="Zero-based integer to use for filename.")
    parser.add_argument('--zfill', type=int, default=4, help="Zero-pad width for filename (default=4).")
    parser.add_argument('--base_dir', default='configs/datasets')
    parser.add_argument('--sub_dir_1', default=str(date.today()))
    parser.add_argument('--sub_dir_2', default='a')
    
    return parser

# def apply_reproducibility_settings_from_filepath(
#     reproducibility_cfg_filepath, 
#     split,
#     seed_idx
# ):
#     reproducibility_cfg_dict = {}
#     reproducibility_cfg_dict['base'] = config_utils.serialization.deserialize(reproducibility_cfg_filepath)
#     reproducibility_cfg_dict['recovered'] = config_utils.serialization.recursive_recover(
#         reproducibility_cfg_dict['base']
#     )

#     # Apply recovery seed to deterministically recover all elements of data config.
#     ml_utils.reproducibility.apply_reproducibility_settings(
#         reproducibility_cfg_dict['recovered'], 
#         seed_idx=seed_idx,
#         split=split
#     )

def main():
    args = build_arg_parser().parse_args()

    # ---------------------------- Set directory ----------------------------- #
    base_dir = args.base_dir
    sub_dir_1 = args.sub_dir_1
    sub_dir_2 = args.sub_dir_2
    output_dir = fileio_utils.make_dir(base_dir, sub_dir_1, sub_dir_2)
    filename = str(args.index).zfill(args.zfill)

    # Parse set key-value pairs for runtime CLI injection.
    cli = config_utils.ops.parse_override_kv_pairs(args.set or [])

    # ---------------------- Build hypercube config ------------------------- #
    # Get CLI overrides first.
    num_hypercube_dims = config_utils.ops.select(cli, 'hypercube.num_dims', 2)
    manual_labels = config_utils.ops.select(cli, 'hypercube.manual_labels', True)
    random_labels = config_utils.ops.select(cli, 'hypercube.random_labels', True)
    manual_pmfs = config_utils.ops.select(cli, 'hypercube.manual_pmfs', True)

    # Local intermediates related to vertices.
    num_vertices = 2**num_hypercube_dims
    half = num_vertices // 2
    coords = torch.arange(num_hypercube_dims, dtype=torch.int64)
    if manual_labels:
        pos_vert_labels = torch.tensor([2, 3])
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
        pos_vert_pmf = torch.tensor([0., 0., 1.])
        neg_vert_pmf = torch.tensor([1.])
    else:
        pos_vert_pmf = data_utils.uniform_pmf(len(pos_vert_labels))
        neg_vert_pmf = data_utils.uniform_pmf(len(num_vertices-len(pos_vert_labels)))

    # Build hypercube config object.
    hypercube_args_cfg = HypercubeConfig(
        num_dims=2,
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
    manual_seq_lengths_support = config_utils.ops.select(cli, 'seq_lengths.manual_support', True)
    manual_seq_lengths_pmf = config_utils.ops.select(cli, 'seq_lengths.manual_pmf', True)


    seq_length_helper = {'pos': {}, 'neg': {}}
    seq_length_helper['pos'].seq_len_max = config_utils.ops.select(
        cli, 
        'seq_lengths.pos.support.max', 
        5
    )
    seq_length_helper['neg'].seq_len_max = config_utils.ops.select(
        cli, 
        'seq_lengths.neg.support.max', 
        5
    )
    seq_length_helper['pos'].parity = config_utils.ops.select(cli, 'seq_lengths.pos.parity', None)
    neg_seq_len_max = config_utils.ops.pick(cli, 'seq_lengths.neg.support.max', 5)
    seq_lengths = SeqLengths(
        lengths={
            'pos' : {
                'support' : TensorConfig.from_tensor(
                    tensor_utils.single_parity_arange(pos_seq_len_max, 'even')
                ),
                'pmf' : TensorConfig.from_tensor(
                    data_utils.uniform_pmf(len(tensor_utils.single_parity_arange(pos_seq_len_max, 'even')))
                )
            },
            'neg' : {
                'support' : TensorConfig.from_tensor(
                    tensor_utils.single_parity_arange(neg_seq_len_max, 'even')
                ),
                'pmf' : TensorConfig.from_tensor(
                    data_utils.uniform_pmf(len(tensor_utils.single_parity_arange(neg_seq_len_max, 'even')))
                )
            }
        }
    )

    embedder_cfg = EmbedderConfig(
        ambient_dim=5,
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
        train=1000,
        val=500,
        test=500
    )

    # --------------------------- Top level config --------------------------- #
    data_cfg = DataConfig(
        sequences_cfg=sequences_cfg,
        split_cfg=split_cfg
    )

    # -------------------------- Apply CLI overrides ------------------------- #
    data_cfg = config_utils.ops.apply_cli_override(data_cfg, args.set or [])

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
        reproducibility_cfg_filepath='configs/reproducibility/2025-08-11/a/0000.json',
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





 

###############################################################################



# import argparse
# from datetime import date

# import torch

# from data import builder as data_builder 
# from data.config import DataConfig, EmbedderConfig, HypercubeConfig, NormalDistrConfig, SeqLengths, SequencesConfig, SplitConfig
# from data.sequences import Hypercube, Embedder
# from data import utils as data_utils
# from general_utils.config import CallableConfig, TensorConfig
# from general_utils import configops as config_utils.ops
# from general_utils import fileio as fileio_utils
# from general_utils import records as config_utils.summary
# from general_utils import serialization as config_utils.serialization
# from general_utils import tensor as tensor_utils


# def build_arg_parser():
#     parser = argparse.ArgumentParser(
#         parents=[config_utils.ops.get_parser()],
#     )
#     parser.add_argument('--index', type=int, required=True, help="Zero-based integer to use for filename.")
#     parser.add_argument('--zfill', type=int, default=4, help="Zero-pad width for filename (default=4).")
#     parser.add_argument('--base_dir', default='configs/datasets')
#     parser.add_argument('--sub_dir_1', default=str(date.today()))
#     parser.add_argument('--sub_dir_2', default='a')
    
#     return parser

# def main():
#     args = build_arg_parser().parse_args()

#     # ---------------------------- Set directory ----------------------------- #
#     base_dir = args.base_dir
#     sub_dir_1 = args.sub_dir_1
#     sub_dir_2 = args.sub_dir_2
#     output_dir = fileio_utils.make_dir(base_dir, sub_dir_1, sub_dir_2)
#     filename = str(args.index).zfill(args.zfill)
#     # # ---------------------------- Set directory ----------------------------- #
#     # base_dir = 'configs/datasets'
#     # # sub_dir_1 = 'aaaa'
#     # sub_dir_1 = str(date.today())
#     # sub_dir_2 = 'a'
#     # output_dir = fileio_utils.make_dir(base_dir, sub_dir_1, sub_dir_2)
#     # filename = fileio_utils.make_filename('0000')

#     # ------------------------- Get pre sweep items -------------------------- #
#     pre = config_utils.ops.parse_override_list(args.pre or [])

#     # ------------------------ Build auxiliary objects ----------------------- #
#     hypercube_args_cfg = HypercubeConfig(
#         num_dims=2,
#         coords=TensorConfig.from_tensor(
#             torch.tensor([0, 1], dtype=torch.int64)
#         ),
#         inclusion_set=TensorConfig.from_tensor(
#             torch.tensor(
#                 [[0, 1], [1, 0], [1, 1]], dtype=torch.int8
#             )
#         ),
#         encoding=TensorConfig.from_tensor(
#             torch.tensor([0, 1], dtype=torch.int8)
#         ),
#         vertices_pmfs=(
#             TensorConfig.from_tensor(
#                 data_utils.uniform_pmf(3)
#             ),
#             TensorConfig.from_tensor(
#                 torch.tensor([1.], dtype=torch.float32)
#             )
#         )
#     )


#     pos_seq_len_max = config_utils.ops.pick(pre, 'sequences_cfg.seq_lengths.lengths.pos.support.max', 5)
#     neg_seq_len_max = config_utils.ops.pick(pre, 'sequences_cfg.seq_lengths.lengths.neg.support.max', 5)
#     seq_lengths = SeqLengths(
#         lengths={
#             'pos' : {
#                 'support' : TensorConfig.from_tensor(
#                     tensor_utils.single_parity_arange(pos_seq_len_max, 'even')
#                 ),
#                 'pmf' : TensorConfig.from_tensor(
#                     data_utils.uniform_pmf(len(tensor_utils.single_parity_arange(pos_seq_len_max, 'even')))
#                 )
#             },
#             'neg' : {
#                 'support' : TensorConfig.from_tensor(
#                     tensor_utils.single_parity_arange(neg_seq_len_max, 'even')
#                 ),
#                 'pmf' : TensorConfig.from_tensor(
#                     data_utils.uniform_pmf(len(tensor_utils.single_parity_arange(neg_seq_len_max, 'even')))
#                 )
#             }
#         }
#     )

#     embedder_cfg = EmbedderConfig(
#         ambient_dim=5,
#         mean_center=False,
#         offset_1=TensorConfig.from_tensor(-torch.tile(torch.tensor([0.5]), (hypercube_args_cfg.num_dims + 3,))), # Plus 3 for the dimensions corresponding to special tokens
#         offset_2=None,
#         method='random_rotation',
#         noise_distr=CallableConfig.from_callable(
#             torch.distributions.Normal, 
#             NormalDistrConfig(loc=0, scale=0.01),
#             kind='class',
#             recovery_mode='call'
#         )
#     )

#     elem=CallableConfig.from_callable(
#             Hypercube, 
#             hypercube_args_cfg, 
#             kind='class', 
#             recovery_mode='call'
#         )
    
#     embedder=CallableConfig.from_callable(
#             Embedder,
#             embedder_cfg,
#             kind='class',
#             recovery_mode='call'
#         )

#     # ---------------------------- Sequences config -------------------------- #
#     sequences_cfg = SequencesConfig(
#         num_seq='num_seq___',
#         seq_order='permute',
#         seq_lengths=seq_lengths,
#         elem=elem,
#         embedder=embedder
#     )

#     # ----------------------------- Split sizes ------------------------------ #
#     split_cfg = SplitConfig(
#         train=1000,
#         val=500,
#         test=500
#     )

#     # --------------------------- Top level config --------------------------- #
#     data_cfg = DataConfig(
#         sequences_cfg=sequences_cfg,
#         split_cfg=split_cfg
#     )

#     # -------------------------- Apply CLI overrides ------------------------- #
#     data_cfg = config_utils.ops.apply_cli_override(data_cfg, args.set or [])

#     # ------------------------------ Serialize ------------------------------- #
#     # Attempt to serialize and reconstruct full cfg tree, and use reconstructed
#     # version to build, to ensure future reproducibility.
#     data_cfg_filepath = output_dir / (filename + '.json')
#     _ = config_utils.serialization.serialize(data_cfg, data_cfg_filepath)

#     # -------------------- Test deserialization/execution -------------------- #
#     # Attempt to build dataset from serialized file.
#     data_builder.build_sequences_from_filepath(
#         data_cfg_filepath, 
#         build=['train', 'val'], 
#         reproducibility_cfg_filepath='configs/reproducibility/2025-08-11/a/0000.json',
#         seed_idx=0, 
#         print_to_console=True, 
#         save_path=None
#     )

#     # --------------------------- Summarize config --------------------------- #
#     # Registry of items to extract from the config.
#     REGISTRY = {
#         'hypercube_dim': 'sequences_cfg.elem.args_cfg.num_dims',
#         'pos_vertices': 'sequences_cfg.elem.args_cfg.inclusion_set.args_cfg.data',
#         'pos_vertices_pmf': 'sequences_cfg.elem.args_cfg.vertices_pmfs.0.args_cfg.data',
#         'neg_vertices_pmf': 'sequences_cfg.elem.args_cfg.vertices_pmfs.1.args_cfg.data',
#         'pos_seq_lengths': 'sequences_cfg.seq_lengths.lengths.pos.support.args_cfg.data',
#         'pos_seq_lengths_pmf': 'sequences_cfg.seq_lengths.lengths.pos.pmf.args_cfg.data',
#         'neg_seq_lengths': 'sequences_cfg.seq_lengths.lengths.neg.support.args_cfg.data',
#         'neg_seq_lengths_pmf': 'sequences_cfg.seq_lengths.lengths.neg.pmf.args_cfg.data',
#         'train_size': 'split_cfg.train',
#         'val_size': 'split_cfg.val',
#         'test_size': 'split_cfg.test'
#     }

#     # Deserialize and summarize config to .xlsx file.
#     config_utils.summary.summarize_cfg_to_xlsx(
#         data_cfg_filepath, 
#         config_kind='datasets', 
#         config_id=str(data_cfg_filepath).removeprefix('configs/datasets/').removesuffix('.json'),
#         dotted_path_registry=REGISTRY,
#         note='',
#         xlsx_filepath='configs/logs.xlsx'
#     )

# if __name__ == '__main__':
#     main()


# # ------------------------------------------------------------------------------







# import argparse
# from datetime import date

# import torch

# from data import builder as data_builder 
# from data.config import DataConfig, EmbedderConfig, HypercubeConfig, NormalDistrConfig, SeqLengths, SequencesConfig, SplitConfig
# from data.sequences import Hypercube, Embedder
# from data import utils as data_utils
# from general_utils.config import CallableConfig, TensorConfig
# from general_utils import configops as config_utils.ops
# from general_utils import fileio as fileio_utils
# from general_utils import records as config_utils.summary
# from general_utils import serialization as config_utils.serialization
# from general_utils import tensor as tensor_utils


# def build_arg_parser():
#     parser = argparse.ArgumentParser(
#         parents=[config_utils.ops.get_parser()],
#     )
#     parser.add_argument('--index', type=int, required=True, help="Zero-based integer to use for filename.")
#     parser.add_argument('--zfill', type=int, default=4, help="Zero-pad width for filename (default=4).")
#     parser.add_argument('--base_dir', default='configs/datasets')
#     parser.add_argument('--sub_dir_1', default=str(date.today()))
#     parser.add_argument('--sub_dir_2', default='a')
    
#     return parser

# def main():
#     args = build_arg_parser().parse_args()

#     # ---------------------------- Set directory ----------------------------- #
#     base_dir = args.base_dir
#     sub_dir_1 = args.sub_dir_1
#     sub_dir_2 = args.sub_dir_2
#     output_dir = fileio_utils.make_dir(base_dir, sub_dir_1, sub_dir_2)
#     filename = str(args.index).zfill(args.zfill)
#     # # ---------------------------- Set directory ----------------------------- #
#     # base_dir = 'configs/datasets'
#     # # sub_dir_1 = 'aaaa'
#     # sub_dir_1 = str(date.today())
#     # sub_dir_2 = 'a'
#     # output_dir = fileio_utils.make_dir(base_dir, sub_dir_1, sub_dir_2)
#     # filename = fileio_utils.make_filename('0000')

#     # ------------------------ Build auxiliary objects ----------------------- #
#     hypercube_args_cfg = HypercubeConfig(
#         num_dims=2,
#         coords=TensorConfig.from_tensor(
#             torch.tensor([0, 1], dtype=torch.int64)
#         ),
#         inclusion_set=TensorConfig.from_tensor(
#             torch.tensor(
#                 [[0, 1], [1, 0], [1, 1]], dtype=torch.int8
#             )
#         ),
#         encoding=TensorConfig.from_tensor(
#             torch.tensor([0, 1], dtype=torch.int8)
#         ),
#         vertices_pmfs=(
#             TensorConfig.from_tensor(
#                 data_utils.uniform_pmf(3)
#             ),
#             TensorConfig.from_tensor(
#                 torch.tensor([1.], dtype=torch.float32)
#             )
#         )
#     )

#     seq_lengths = SeqLengths(
#         lengths={
#             'pos' : {
#                 'support' : TensorConfig.from_tensor(
#                     tensor_utils.single_parity_arange(20, 'even')
#                 ),
#                 'pmf' : TensorConfig.from_tensor(
#                     data_utils.uniform_pmf(len(tensor_utils.single_parity_arange(20, 'even')))
#                 )
#             },
#             'neg' : {
#                 'support' : TensorConfig.from_tensor(
#                     tensor_utils.single_parity_arange(20, 'even')
#                 ),
#                 'pmf' : TensorConfig.from_tensor(
#                     data_utils.uniform_pmf(len(tensor_utils.single_parity_arange(20, 'even')))
#                 )
#             }
#         }
#     )

#     embedder_cfg = EmbedderConfig(
#         ambient_dim=5,
#         mean_center=False,
#         offset_1=TensorConfig.from_tensor(-torch.tile(torch.tensor([0.5]), (hypercube_args_cfg.num_dims + 3,))), # Plus 3 for the dimensions corresponding to special tokens
#         offset_2=None,
#         method='random_rotation',
#         noise_distr=CallableConfig.from_callable(
#             torch.distributions.Normal, 
#             NormalDistrConfig(loc=0, scale=0.01),
#             kind='class',
#             recovery_mode='call'
#         )
#     )

#     elem=CallableConfig.from_callable(
#             Hypercube, 
#             hypercube_args_cfg, 
#             kind='class', 
#             recovery_mode='call'
#         )
    
#     embedder=CallableConfig.from_callable(
#             Embedder,
#             embedder_cfg,
#             kind='class',
#             recovery_mode='call'
#         )

#     # ---------------------------- Sequences config -------------------------- #
#     sequences_cfg = SequencesConfig(
#         num_seq='num_seq___',
#         seq_order='permute',
#         seq_lengths=seq_lengths,
#         elem=elem,
#         embedder=embedder
#     )

#     # ----------------------------- Split sizes ------------------------------ #
#     split_cfg = SplitConfig(
#         train=1000,
#         val=500,
#         test=500
#     )

#     # --------------------------- Top level config --------------------------- #
#     data_cfg = DataConfig(
#         sequences_cfg=sequences_cfg,
#         split_cfg=split_cfg
#     )

#     # -------------------------- Apply CLI overrides ------------------------- #
#     data_cfg = config_utils.ops.apply_cli_override(data_cfg, args.set or [])

#     # ------------------------------ Serialize ------------------------------- #
#     # Attempt to serialize and reconstruct full cfg tree, and use reconstructed
#     # version to build, to ensure future reproducibility.
#     data_cfg_filepath = output_dir / (filename + '.json')
#     _ = config_utils.serialization.serialize(data_cfg, data_cfg_filepath)

#     # -------------------- Test deserialization/execution -------------------- #
#     # Attempt to build dataset from serialized file.
#     data_builder.build_sequences_from_filepath(
#         data_cfg_filepath, 
#         build=['train', 'val'], 
#         reproducibility_cfg_filepath='configs/reproducibility/2025-08-11/a/0000.json',
#         seed_idx=0, 
#         print_to_console=True, 
#         save_path=None
#     )

#     # --------------------------- Summarize config --------------------------- #
#     # Registry of items to extract from the config.
#     REGISTRY = {
#         'hypercube_dim': 'sequences_cfg.elem.args_cfg.num_dims',
#         'pos_vertices': 'sequences_cfg.elem.args_cfg.inclusion_set.args_cfg.data',
#         'pos_vertices_pmf': 'sequences_cfg.elem.args_cfg.vertices_pmfs.0.args_cfg.data',
#         'neg_vertices_pmf': 'sequences_cfg.elem.args_cfg.vertices_pmfs.1.args_cfg.data',
#         'pos_seq_lengths': 'sequences_cfg.seq_lengths.lengths.pos.support.args_cfg.data',
#         'pos_seq_lengths_pmf': 'sequences_cfg.seq_lengths.lengths.pos.pmf.args_cfg.data',
#         'neg_seq_lengths': 'sequences_cfg.seq_lengths.lengths.neg.support.args_cfg.data',
#         'neg_seq_lengths_pmf': 'sequences_cfg.seq_lengths.lengths.neg.pmf.args_cfg.data',
#         'train_size': 'split_cfg.train',
#         'val_size': 'split_cfg.val',
#         'test_size': 'split_cfg.test'
#     }

#     # Deserialize and summarize config to .xlsx file.
#     config_utils.summary.summarize_cfg_to_xlsx(
#         data_cfg_filepath, 
#         config_kind='datasets', 
#         config_id=str(data_cfg_filepath).removeprefix('configs/datasets/').removesuffix('.json'),
#         dotted_path_registry=REGISTRY,
#         note='',
#         xlsx_filepath='configs/logs.xlsx'
#     )

# if __name__ == '__main__':
#     main()




