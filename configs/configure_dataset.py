import argparse
from datetime import date

import torch

from data import builder as data_builder 
from data.config import DataConfig, EmbedderConfig, HypercubeConfig, NormalDistrConfig, SeqLengths, SequencesConfig, SplitConfig
from data.sequences import Hypercube, Embedder
from data import utils as data_utils
from general_utils.config import CallableConfig, TensorConfig
from general_utils import configops as configops_utils
from general_utils import fileio as fileio_utils
from general_utils import records as records_utils
from general_utils import serialization as serialization_utils
from general_utils import tensor as tensor_utils


def build_arg_parser():
    parser = argparse.ArgumentParser(
        parents=[configops_utils.get_parser()],
    )
    parser.add_argument('--index', type=int, required=True, help="Zero-based integer to use for filename.")
    parser.add_argument('--zfill', type=int, default=4, help="Zero-pad width for filename (default=4).")
    parser.add_argument('--base_dir', default='configs/datasets')
    parser.add_argument('--sub_dir_1', default=str(date.today()))
    parser.add_argument('--sub_dir_2', default='a')
    
    return parser

def main():
    args = build_arg_parser().parse_args()

    # ---------------------------- Set directory ----------------------------- #
    base_dir = args.base_dir
    sub_dir_1 = args.sub_dir_1
    sub_dir_2 = args.sub_dir_2
    output_dir = fileio_utils.make_dir(base_dir, sub_dir_1, sub_dir_2)
    filename = str(args.index).zfill(args.zfill)

    # ------------------------- Get pre sweep items -------------------------- #
    pre = configops_utils.parse_override_list(args.pre or [])

    # ------------------------ Build auxiliary objects ----------------------- #
    hypercube_num_dims = configops_utils.select(pre, 'sequences_cfg.elem.args_cfg.num_dims', 2)

    num_vertices = 2**hypercube_num_dims
    MANUAL_LABELS = False
    RANDOM_LABELS = True
    if MANUAL_LABELS:
        vertices_labels = torch.tensor([], dtype=torch.int64)
    else:
        if RANDOM_LABELS:
            vertices_labels = torch.randperm(num_vertices, dtype=torch.int64)
            pos_labels = vertices_labels[num_vertices//2:]
            neg_labels = vertices_labels[:num_vertices//2]
        else:
            neg_labels = torch.arange(num_vertices//2)
            pos_labels = torch.arange(start=num_vertices//2, end=num_vertices)

    hypercube_args_cfg = HypercubeConfig(
        num_dims=hypercube_num_dims,
        coords=TensorConfig.from_tensor(
            torch.arange(hypercube_num_dims, dtype=torch.int64)
        ),
        inclusion_set=TensorConfig.from_tensor(
            data_utils.get_lexicographic_ordering(
                hypercube_num_dims, 
                encoding=torch.tensor([0, 1]),
                dtype=torch.int8
            )[pos_labels, :]
        ),
        # inclusion_set=TensorConfig.from_tensor(
        #     torch.tensor(
        #         [[0, 1], [1, 0], [1, 1]], dtype=torch.int8
        #     )
        # ),
        encoding=TensorConfig.from_tensor(
            torch.tensor([0, 1], dtype=torch.int8)
        ),
        vertices_pmfs=(
            TensorConfig.from_tensor(
                data_utils.uniform_pmf(int(num_vertices/2))
            ),
            TensorConfig.from_tensor(
                data_utils.uniform_pmf(int(num_vertices/2))
            )
        )
    )


    pos_seq_len_max = configops_utils.select(pre, 'sequences_cfg.seq_lengths.lengths.pos.support.max', 5)
    # neg_seq_len_max = configops_utils.pick(pre, 'sequences_cfg.seq_lengths.lengths.neg.support.max', 5)
    neg_seq_len_max = pos_seq_len_max
    parity_pos = configops_utils.select(pre, 'sequences_cfg.seq_lengths.lengths.pos.support.parity', None)
    parity_neg = configops_utils.select(pre, 'sequences_cfg.seq_lengths.lengths.neg.support.parity', None)
    seq_lengths = SeqLengths(
        lengths={
            'pos' : {
                'support' : TensorConfig.from_tensor(
                    tensor_utils.single_parity_arange(pos_seq_len_max, parity_pos)
                ),
                'pmf' : TensorConfig.from_tensor(
                    data_utils.uniform_pmf(len(tensor_utils.single_parity_arange(pos_seq_len_max, parity_pos)))
                )
            },
            'neg' : {
                'support' : TensorConfig.from_tensor(
                    tensor_utils.single_parity_arange(neg_seq_len_max, parity_neg)
                ),
                'pmf' : TensorConfig.from_tensor(
                    data_utils.uniform_pmf(len(tensor_utils.single_parity_arange(neg_seq_len_max, parity_neg)))
                )
            }
        }
    )

    embedder_cfg = EmbedderConfig(
        ambient_dim=80,
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
    data_cfg = configops_utils.apply_cli_override(data_cfg, args.set or [])

    # ------------------------------ Serialize ------------------------------- #
    # Attempt to serialize and reconstruct full cfg tree, and use reconstructed
    # version to build, to ensure future reproducibility.
    data_cfg_filepath = output_dir / (filename + '.json')
    _ = serialization_utils.serialize(data_cfg, data_cfg_filepath)

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
    records_utils.summarize_cfg_to_xlsx(
        data_cfg_filepath, 
        config_kind='datasets', 
        config_id=str(data_cfg_filepath).removeprefix('configs/datasets/').removesuffix('.json'),
        dotted_path_registry=REGISTRY,
        note="Grid over sequence length (single parity, pos=neg) and hypercube dim, but with half of vertices randomly assigned to pos class, and the remaining to neg class",
        xlsx_filepath='configs/logs.xlsx'
    )

if __name__ == '__main__':
    main()

# import argparse
# from datetime import date

# import torch

# from data import builder as data_builder 
# from data.config import DataConfig, EmbedderConfig, HypercubeConfig, NormalDistrConfig, SeqLengths, SequencesConfig, SplitConfig
# from data.sequences import Hypercube, Embedder
# from data import utils as data_utils
# from general_utils.config import CallableConfig, TensorConfig
# from general_utils import configops as configops_utils
# from general_utils import fileio as fileio_utils
# from general_utils import records as records_utils
# from general_utils import serialization as serialization_utils
# from general_utils import tensor as tensor_utils


# def build_arg_parser():
#     parser = argparse.ArgumentParser(
#         parents=[configops_utils.get_parser()],
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
#     pre = configops_utils.parse_override_list(args.pre or [])

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


#     pos_seq_len_max = configops_utils.pick(pre, 'sequences_cfg.seq_lengths.lengths.pos.support.max', 5)
#     neg_seq_len_max = configops_utils.pick(pre, 'sequences_cfg.seq_lengths.lengths.neg.support.max', 5)
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
#     data_cfg = configops_utils.apply_cli_override(data_cfg, args.set or [])

#     # ------------------------------ Serialize ------------------------------- #
#     # Attempt to serialize and reconstruct full cfg tree, and use reconstructed
#     # version to build, to ensure future reproducibility.
#     data_cfg_filepath = output_dir / (filename + '.json')
#     _ = serialization_utils.serialize(data_cfg, data_cfg_filepath)

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
#     records_utils.summarize_cfg_to_xlsx(
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
# from general_utils import configops as configops_utils
# from general_utils import fileio as fileio_utils
# from general_utils import records as records_utils
# from general_utils import serialization as serialization_utils
# from general_utils import tensor as tensor_utils


# def build_arg_parser():
#     parser = argparse.ArgumentParser(
#         parents=[configops_utils.get_parser()],
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
#     data_cfg = configops_utils.apply_cli_override(data_cfg, args.set or [])

#     # ------------------------------ Serialize ------------------------------- #
#     # Attempt to serialize and reconstruct full cfg tree, and use reconstructed
#     # version to build, to ensure future reproducibility.
#     data_cfg_filepath = output_dir / (filename + '.json')
#     _ = serialization_utils.serialize(data_cfg, data_cfg_filepath)

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
#     records_utils.summarize_cfg_to_xlsx(
#         data_cfg_filepath, 
#         config_kind='datasets', 
#         config_id=str(data_cfg_filepath).removeprefix('configs/datasets/').removesuffix('.json'),
#         dotted_path_registry=REGISTRY,
#         note='',
#         xlsx_filepath='configs/logs.xlsx'
#     )

# if __name__ == '__main__':
#     main()




