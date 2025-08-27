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
from general_utils import fileio as fileio_utils
from general_utils import tensor as tensor_utils
from general_utils import validation as validation_utils


@dataclass
class Spec(ContainerConfig):
    hypercube: ContainerConfig
    seq_lengths: ContainerConfig
    embedder: ContainerConfig
    sequences: ContainerConfig
    split: ContainerConfig

@dataclass
class HypercubeSpec:
    num_dims: int = 2
    random_labels: bool = False
    pos_vert_labels: Union[list, Literal["auto"]] = 'auto'
    coords = Union[list, Literal["auto"]] = 'auto'
    pos_vert_pmf = Union[list, Literal["auto"]] = 'auto'
    neg_vert_pmf = Union[list, Literal["auto"]] = 'auto'
    device: str = 'cpu'
    dtype_bit = torch.int8
    dtype_index = torch.int64
    dtype_pmf = torch.float32
    torch_generator_seed = 314159265

def get_hypercube_spec_with_manual_entry():
    """ Add manual specifications here to override defaults, will be overridden by CLI."""
    return replace(HypercubeSpec(), **{})

def build_hypercube_args_config(spec):
    """ 
    Compute things.
    """
    num_dims = spec.num_dims
    num_vertices = 2**num_dims
    half = num_vertices // 2

    # Local normalizers. Private, pure python.
    def _norm_coords(s: HypercubeSpec):
        if s.coords == 'auto':
            return list(range(num_dims))
        vals = list(s.coords)
        if len(vals) != num_dims:
            raise ValueError(
                f"`coords` must be of length {num_dims}, got length {len(vals)}."
            )
        if not all(isinstance(x, int) for x in vals):
            raise TypeError("`coords` must contain only ints.")
        return vals
    
    def _norm_pos_labels(s: HypercubeSpec):
        if s.pos_vert_labels == 'auto':
            if s.random_labels:
                g = torch.Generator()
                g.manual_seed(314159265)
                perm = torch.randperm(num_vertices, generator=g).tolist()
                return sorted(perm[half:])
            else:
                return list(range(half, num_vertices))
        vals = s.pos_vert_labels   
        if not all(isinstance(x, int) for x in vals):
            raise TypeError("Elements of `pos_vert_labels` must be int.")
        if not all(0 <= x <  num_vertices for x in vals):
            raise ValueError("Elements of `pos_vert_labels` must be in range(0, num_vertices).")
        return vals
    
    def _norm_pmf(values):
        if values == 'auto':
            return [1.0 / half] * half
        pmf = [float(x) for x in values]
        return pmf
    
    # Normalize inputs in python.
    coords_py = _norm_coords(spec)
    pos_labels_py = _norm_pos_labels(spec)
    pos_pmf_py = _norm_pmf(spec.pos_vert_pmf, half)
    neg_pmf_py = _norm_pmf(spec.neg_vert_pmf, half)

    # Convert to tensors.
    coords_t = torch.tensor(coords_py, dtype=spec.dtype_bit, device=spec.device)
    pos_labels_t = torch.tensor(pos_labels_py, dtype=spec.dtype_index, device=spec.device)

    # Get inclusion set. 
    encoding_t = torch.tensor([0, 1], dtype=spec.dtype_bit, device=spec.device)
    vertices = data_utils.get_lexicographic_ordering(num_dims, encoding_t, dtype=spec.dtype_bit)
    inclusion_set_t = vertices.index_select(0, pos_labels_t)

    # Get PMFs over vertices.
    pos_pmf_t = torch.tensor(pos_pmf_py, dtype=spec.dtype_pmf, device=spec.device)
    neg_pmf_t = torch.tensor(neg_pmf_py, dtype=spec.dtype_pmf, device=spec.device)

    return tensor_utils.recursive_tensor_to_tensor_config(
        HypercubeConfig(
            num_dims=num_dims,
            coords=coords_t,
            inclusion_set=inclusion_set_t,
            encoding=encoding_t,
            vertices_pmfs=(pos_pmf_t, neg_pmf_t)
        )
    )


@dataclass
class SeqLengthSpec:
    max_seq_lengths: dict[str, Optional[int]] = field(default_factory=lambda: {'pos': 20, 'neg': 20})
    seq_length_supports: dict[str, Union[list[int], Literal['auto']]] = field(default_factory=lambda: {'pos': 'auto', 'neg': 'auto'})
    parities: dict[str, Optional[str]] = field(default_factory=lambda: {'pos': None, 'neg': None})
    seq_length_pmfs: dict[str, Union[list[float], Literal['auto']]] = field(default_factory=lambda: {'pos': 'auto', 'neg': 'auto'})
    # max_pos_seq_len = Optional[int] = None
    # pos_seq_len_support = Union[list[int], Literal['auto']]
    # max_neg_seq_len = Optional[int] = None
    # neg_seq_len_support = Union[list[int], Literal['auto']]
    # pos_parity = Optional[str] = None
    # neg_parity = Optional[str] = None

def get_seq_len_spec_with_manual_entry():
    return replace(SeqLengthSpec, **{})

def build_seq_len_config(spec):




    def _norm_seq_len_support(spec):
        supports = {}

        for name in spec.seq_length_support.keys():
            if spec.seq_length_supports[name] == 'auto':
                if spec.max_seq_len[name] is None:
                    raise ValueError(
                        f"Class '{name}' has seq_len_support = 'auto', so "
                        "spec.max_pos_seq_len may not be None."
                    )
                if spec.parities[name] in ('odd', 'even'):
                    supports
                elif spec.parities[name] is None:
                    supports[name] = list(range(0, spec.max_seq_len[name]))
            else:
                validation_utils.validate_iterable_contents(
                    spec.seq_length_supports[name],
                    validation_utils.is_nonneg_float,
                    "a nonneg float"
                )
                supports[name] = spec.seq_length_supports[name]
                
            
                if spec.pos_parity in ('odd', 'even'):
                    pass
                    # return list(range(0, sp))


    # def get_coords(hc_spec):
    #     """ 
    #     hc_spec.coords should be list or 'auto'
    #     """
    #     return (
    #         torch.arange(hc_spec.num_dims, dtype=torch.int64) 
    #         if hc_spec.coords == 'auto'
    #         else torch.tensor(hc_spec.get_coords)
    #     )
        
    # def get_vertex_labels(hc_spec):
    #     """ 
    #     """
    #     if hc_spec.pos_vert_labels == 'auto':
    #         if hc_spec.random:
    #             vertices_labels = torch.randperm(num_vertices, dtype=torch.int64)
    #             pos_vertex_labels = vertices_labels[num_vertices//2:]
    #             neg_vertex_labels = vertices_labels[:num_vertices//2]
    #         else:
    #             pos_vertex_labels = torch.arange(start=num_vertices//2, end=num_vertices)
    #             neg_vertex_labels = torch.arange(num_vertices//2)
    #     else:
    #         validation_utils.validate_iterable_contents(
    #             hc_spec.pos_vert_labels, 
    #             validation_utils.is_nonneg_int, 
    #             "a nonneg int"
            
    #         ) 
    #         pos_vertex_labels = torch.tensor(hc_spec.pos_vert_labels)
    #         neg_vertex_labels = torch.arange(num_vertices)[~torch.isin(torch.arange(num_vertices), pos_vertex_labels)]
                
    #     return pos_vertex_labels, neg_vertex_labels
    
    # def get_vertex_pmf(hc_spec):
    #     if hc_spec.pos_vert_pmf == 'auto':
    #         pos_vert_pmf = data_utils.uniform_pmf(int(num_vertices/2))
    #     else:
    #         validation_utils.validate_iterable_contents(
    #             hc_spec.pos_vert_pmf,
    #             validation_utils.is_nonneg_float,
    #             "a nonneg float"
    #         )
    #         pos_vert_pmf = torch.tensor(hc_spec.pos_vert_pmf)
    #     if hc_spec.neg_vert_pmf == 'auto':
    #         neg_vert_pmf = data_utils.uniform_pmf(int(num_vertices/2))
    #     else:
    #         validation_utils.validate_iterable_contents(
    #             hc_spec.neg_vert_pmf,
    #             validation_utils.is_nonneg_float,
    #             "a nonneg float"
    #         )
    #         neg_vert_pmf = torch.tensor(hc_spec.neg_vert_pmf)

    #     return pos_vert_pmf, neg_vert_pmf
    
    # coords=TensorConfig.from_tensor(get_coords(hc_spec))
    # pos_vert_labels = get_vertex_labels(hc_spec)[0]
    # inclusion_set=TensorConfig.from_tensor(
    #     data_utils.get_lexicographic_ordering(
    #         hc_spec.num_dims, 
    #         encoding=torch.tensor([0, 1]),
    #         dtype=torch.int8
    #     )[pos_vert_labels, :]
    # ),
    # encoding=TensorConfig.from_tensor(
    #     torch.tensor([0, 1], dtype=torch.int8)
    # )
    # pos_vert_pmf, neg_vert_pmf = get_vertex_pmf(hc_spec)
    # vertices_pmfs=(
    #     TensorConfig.from_tensor(pos_vert_pmf),
    #     TensorConfig.from_tensor(neg_vert_pmf)
    # )
        
    
    # hypercube_args_cfg = HypercubeConfig(
    #     num_dims=hc_spec.num_dims,
    #     coords=coords,
    #     inclusion_set=inclusion_set,
    #     encoding=encoding,
    #     vertices_pmfs=vertices_pmfs
    # )

    
    # return hypercube_args_cfg


# @dataclass
# class SeqLengthSpec:
#     pos_seq_len_max: int = 10 # Default value if no CLI override
#     neg_seq_len_max: int = 10 # Default value if no CLI override
#     parity_pos: Optional[str] = None
#     parity_neg: Optional[str] = None





# def build_arg_parser():
#     parser = argparse.ArgumentParser(
#         parents=[config_utils.ops.make_parent_parser()],
#     )
#     parser.add_argument('--base_dir', default='configs/datasets')
#     parser.add_argument('--sub_dir_1', default=str(date.today()))
#     parser.add_argument('--sub_dir_2', default='a')
    
#     return parser

# # def resolve_hypercube_spec(spec):
# #     """ 
# #     Compute things.
# #     """
# #     def get_vertex_labels(num_vertices, random):
# #         """ 
# #         """
# #         if random:
# #             vertices_labels = torch.randperm(num_vertices, dtype=torch.int64)
# #             pos_vertex_labels = vertices_labels[num_vertices//2:]
# #             neg_vertex_labels = vertices_labels[:num_vertices//2]
# #         else:
# #             pos_vertex_labels = torch.arange(start=num_vertices//2, end=num_vertices)
# #             neg_vertex_labels = torch.arange(num_vertices//2)
            
# #         return pos_vertex_labels, neg_vertex_labels
    
# #     s = replace(spec)

# #     s.num_vertices = 2**s.num_hypercube_dims
# #     if s.pos_vert_labels == 'auto':
# #         s.pos_vert_labels, s.neg_vert_labels = get_vertex_labels(
# #             s.num_vertices, 
# #             s.random_labels
# #         )
# #     if s.pos_vert_pmf == 'auto':
# #         s.pos_vert_pmf = data_utils.uniform_pmf(int(s.num_vertices/2))
# #     if s.neg_vert_pmf == 'auto':
# #         s.pos_vert_pmf = data_utils.uniform_pmf(int(s.num_vertices/2))

# def main():
#     args = build_arg_parser().parse_args()

#     # ---------------------------- Set directory ---------------------------- #
#     base_dir = args.base_dir
#     sub_dir_1 = args.sub_dir_1
#     sub_dir_2 = args.sub_dir_2
#     output_dir = fileio_utils.make_dir(base_dir, sub_dir_1, sub_dir_2)
#     filename = str(args.idx).zfill(args.zfill)

#     # ------------------------- Get pre sweep items ------------------------- #
#     set_ = config_utils.ops.parse_override_kv_pairs(args.set or [])

#     # ----------------------------------------------------------------------- #
#     # --------------------------- Set parameters ---------------------------- #
#     # ----------------------------------------------------------------------- #

#     # ------------------------------ Hypercube ------------------------------ #
#     num_hypercube_dims = config_utils.ops.select(set_, 'num_hypercube_dims', 2)
#     random_labels = config_utils.ops.select(set_, 'random_vertex_labels', False)
#     # pos_vert_labels, neg_vert_labels = (
#     #     get_vertex_labels(num_vertices=2**num_hypercube_dims, random=random_labels) 
#     #     if hypercu
#     # )
#     coords = torch.arange(num_hypercube_dims, dtype=torch.int64)
#     pos_vert_pmf = data_utils.uniform_pmf(int(2**num_hypercube_dims/2))
#     neg_vert_pmf = data_utils.uniform_pmf(int(2**num_hypercube_dims/2))

#     # --------------------------- Sequence Lengths -------------------------- #
#     pos_seq_len_max = config_utils.ops.select(set_, 'sequences_cfg.seq_lengths.lengths.pos.support.max', 5)
#     neg_seq_len_max = pos_seq_len_max
#     parity_pos = config_utils.ops.select(set_, 'sequences_cfg.seq_lengths.lengths.pos.support.parity', None)
#     parity_neg = config_utils.ops.select(set_, 'sequences_cfg.seq_lengths.lengths.neg.support.parity', None)
#     if parity_pos is not None:
#         pos_seq_len_support = tensor_utils.single_parity_arange(pos_seq_len_max, parity_pos)
        
#     else:
#         pos_seq_len_support = torch.arange(pos_seq_len_max)
#     pos_seq_len_pmf = data_utils.uniform_pmf(len(pos_seq_len_support))
#     if parity_neg is not None:
#         neg_seq_len_support = tensor_utils.single_parity_arange(neg_seq_len_max, parity_neg)
        
#     else:
#         neg_seq_len_support = torch.arange(neg_seq_len_max)
#     neg_seq_len_pmf = data_utils.uniform_pmf(len(neg_seq_len_support))

#     # ------------------------------- Embedder ------------------------------ #
#     ambient_dim = config_utils.ops.select(set_, 'ambient_dim', 10)

#     # ----------------------------------------------------------------------- #
#     # ----------------------------------------------------------------------- #
#     # ----------------------------------------------------------------------- #

#     # ----------------------- Build auxiliary objects ----------------------- #
#     # # num_hypercube_dims = config_utils.ops.select(set_, 'sequences_cfg.elem.args_cfg.num_dims', 2)

#     # # num_vertices = 2**num_hypercube_dims
#     # # manual_labels = False
#     # # random_labels = True
#     # if manual_labels:
#     #     vertices_labels = torch.tensor([], dtype=torch.int64)
#     # else:
#     #     if random_labels:
#     #         vertices_labels = torch.randperm(num_vertices, dtype=torch.int64)
#     #         pos_vertex_labels = vertices_labels[num_vertices//2:]
#     #         neg_vertex_labels = vertices_labels[:num_vertices//2]
#     #     else:
#     #         neg_vertex_labels = torch.arange(num_vertices//2)
#     #         pos_vertex_labels = torch.arange(start=num_vertices//2, end=num_vertices)

#     # hypercube_args_cfg = HypercubeConfig(
#     #     num_dims=num_hypercube_dims,
#     #     coords=TensorConfig.from_tensor(
#     #         torch.arange(num_hypercube_dims, dtype=torch.int64)
#     #     ),
#     #     inclusion_set=TensorConfig.from_tensor(
#     #         data_utils.get_lexicographic_ordering(
#     #             num_hypercube_dims, 
#     #             encoding=torch.tensor([0, 1]),
#     #             dtype=torch.int8
#     #         )[pos_vert_labels, :]
#     #     ),
#     #     # inclusion_set=TensorConfig.from_tensor(
#     #     #     torch.tensor(
#     #     #         [[0, 1], [1, 0], [1, 1]], dtype=torch.int8
#     #     #     )
#     #     # ),
#     #     encoding=TensorConfig.from_tensor(
#     #         torch.tensor([0, 1], dtype=torch.int8)
#     #     ),
#     #     vertices_pmfs=(
#     #         TensorConfig.from_tensor(pos_vert_pmf),
#     #         TensorConfig.from_tensor(neg_vert_pmf)
#     #     )
#     # )


#     # pos_seq_len_max = config_utils.ops.select(set_, 'sequences_cfg.seq_lengths.lengths.pos.support.max', 5)
#     # # neg_seq_len_max = config_utils.ops.pick(set_, 'sequences_cfg.seq_lengths.lengths.neg.support.max', 5)
#     # neg_seq_len_max = pos_seq_len_max
#     # parity_pos = config_utils.ops.select(set_, 'sequences_cfg.seq_lengths.lengths.pos.support.parity', None)
#     # parity_neg = config_utils.ops.select(set_, 'sequences_cfg.seq_lengths.lengths.neg.support.parity', None)
#     seq_lengths = SeqLengths(
#         lengths={
#             'pos' : {
#                 'support' : TensorConfig.from_tensor(pos_seq_len_support),
#                 'pmf' : TensorConfig.from_tensor(pos_seq_len_pmf)
#             },
#             'neg' : {
#                 'support' : TensorConfig.from_tensor(neg_seq_len_support),
#                 'pmf' : TensorConfig.from_tensor(neg_seq_len_pmf)
#             }
#         }
#     )

#     embedder_cfg = EmbedderConfig(
#         ambient_dim=80,
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

#     # ----------------------- Apply late CLI overrides ---------------------- #
#     data_cfg = config_utils.ops.apply_cli_override_to_cfg(data_cfg, args.cfg or [])

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
#         reproducibility_cfg_filepath='configs/reproducibility/2025-08-25/a/0000.json',
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
#         note="Grid over sequence length (single parity, pos=neg) and hypercube dim, but with half of vertices randomly assigned to pos class, and the remaining to neg class",
#         xlsx_filepath='configs/logs.xlsx'
#     )

# if __name__ == '__main__':
#     main()


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




