from dataclasses import asdict

import torch

from data.builder import build_hypercube_sequences
from data.config import DataConfig, HypercubeConfig, EmbedderConfig, SequencesConfig, SeqLengths, NormalDistrConfig
from data.sequences import Hypercube, Embedder
from data import utils as data_utils
from general_utils.config import ReproducibilityConfig, SeedConfig, TorchDeterminismConfig, CallableConfig, TensorConfig
from general_utils import fileio as io_utils
from general_utils import reproducibility 
from general_utils.serialization import serialize, deserialize, recursive_instantiation



# @dataclass
# class SeedConfig(ArgsConfig):
#     torch_seed: int
#     cuda_seed: int


# @dataclass
# class TorchDeterminismConfig(ArgsConfig):
#     use_deterministic_algos: bool = False
#     cudnn_deterministic: bool = False
#     cudnn_benchmark: bool = True


# @dataclass
# class ReproducibilityConfig(ContainerConfig):
#     entropy: int
#     seed_cfg_dict: Dict[str, SeedConfig]
#     torch_determinisim_cfg_dict : Dict[str, TorchDeterminismConfig]


# @dataclass
# class HypercubeConfig(ArgsConfig):
#     num_dims: int
#     coords: Union[torch.Tensor, TensorConfig]
#     inclusion_set: Optional[Union[torch.Tensor, TensorConfig]] = None
#     encoding: Union[torch.Tensor, TensorConfig] = torch.tensor([0, 1], dtype=torch.int8)
    

# @dataclass
# class SeqLengths:
#     """ 
#     Helper class for validating and storing in a format compatible with the
#     Sequence class N distributions over the respective lengths of N sequences,
#     for a natural number N.

#     lengths : dict
#         dict where each key is the name of a kind of sequence (e.g. 'pos', 
#         'neg'), and each value is a dict with the following keys:
#             'support' : 1D tensor of non-negative ints.
#             'pmf' : 1D tensor of probability masses, same legnth as 'support'.
#     """
#     lengths: Dict[str, Dict[str, Union[torch.Tensor, TensorConfig]]]

#     def validate(self):
#         for name, entry in self.lengths.items():
#             try:
#                 support, pmf = entry['support'], entry['pmf']
#                 tensor_utils.validate_tensor(support, 1)
#                 data_utils.validate_pmf(pmf, len(support))
#             except Exception as e:
#                 raise ValueError(f"Validation failed for '{name}'.") from e
            

# @dataclass
# class EmbedderConfig(ArgsConfig):
#     ambient_dim: int
#     mean_center: bool = False
#     offset_1: Optional[Union[torch.Tensor, TensorConfig]] = None
#     offset_2: Optional[Union[torch.Tensor, TensorConfig]] = None
#     method: Union[str, Union[torch.Tensor, TensorConfig]] = 'random_rotation'
#     noise_distr: Optional[CallableConfig] = None


# @dataclass
# class SequencesConfig(ContainerConfig):
#     seq_lengths: SeqLengths
#     # elem_cls: type # E.g. Hypercube
#     # elem_cfg: Any # E.g. HypercubeConfig
#     elem: CallableConfig
#     embedder: CallableConfig
#     num_seq: int
#     seq_order: str = 'permute'


# @dataclass
# class NormalDistrConfig(ArgsConfig):
#     loc : float 
#     scale : float 


# @dataclass
# class DataConfig(ContainerConfig):
#     sequences_cfg: Dict[str, SequencesConfig]
#     reproducibility_cfg: ReproducibilityConfig
    

# def apply_reproducibility_cfg(seed_cfg: SeedConfig, torch_determinism_cfg: TorchDeterminismConfig):
#     reproducibility.set_seed(**asdict(seed_cfg))
#     reproducibility.set_torch_determinism(**asdict(torch_determinism_cfg))

# def build_hypercube_sequences(cfg: SequencesConfig) -> Sequences:
#     hypercube = cfg.elem
#     embedder = cfg.embedder

#     # Validate seq_lengths object.
#     cfg.seq_lengths.validate()

#     # Check that number of variables is large enough.
#     if embedder.ambient_dim < hypercube.num_dims + 3: 
#         raise ValueError(
#             "`embedder.ambient_dim` must be at least 3 greater than hypercube " 
#             f"dimensionality {hypercube.num_dims}, but got {embedder.ambient_dim}."
#         )
    
#     return Sequences(
#         num_seq=cfg.num_seq,
#         num_vars=hypercube.num_dims, # TODO: Eliminate this argument, see TODO in Sequences class
#         len_distr=cfg.seq_lengths.lengths,
#         elem_distr=hypercube.vertices,
#         transform=embedder,
#         seq_order=cfg.seq_order
#     )




if __name__ == '__main__':
    # ------------------------------ Set directory ------------------------------ #
    output_dir, filename = io_utils.make_file_dir_and_id(
        base_dir='configs/datasets',
        sub_dir_1='000',
        sub_dir_2='000',
        file_ind = ('000', '000', '000')
    )

    # ------------------------ Reproducibility settings ------------------------- #
    # 3 children for train, val, and test. num_words_per_child = number of desired
    # seed repeats.
    NUM_SEEDS = 50
    seed_lists, entropy, _, _ = reproducibility.generate_numpy_seed_sequence(
        num_children=3, num_words_per_child=NUM_SEEDS, dtype='uint32', return_as='int'
    )

    reproducibility_cfg = ReproducibilityConfig(
        entropy=entropy,
        seed_cfg_dict={
            split : [
                SeedConfig(torch_seed=s, cuda_seed=s) 
                for s in seeds
            ]
            for (split, seeds) in zip(('train', 'val', 'test'), seed_lists)
        },
        torch_determinisim_cfg_dict={
            'train' : TorchDeterminismConfig(
                use_deterministic_algos=False,
                cudnn_deterministic=False,
                cudnn_benchmark=True
            ),
            'val' : TorchDeterminismConfig(
                use_deterministic_algos=False,
                cudnn_deterministic=False,
                cudnn_benchmark=True
            ),
            'test' : TorchDeterminismConfig(
                use_deterministic_algos=False,
                cudnn_deterministic=False,
                cudnn_benchmark=True
            )
        }
    )

    # ------------------------- Build auxiliary objects ------------------------- #
    hypercube_args_cfg = HypercubeConfig(
        num_dims=2,
        coords=TensorConfig.from_tensor(
            torch.tensor([0, 1], dtype=torch.int64)
        ),
        inclusion_set=TensorConfig.from_tensor(
            torch.tensor(
                [[1, 0], [1, 1]],
                dtype=torch.int8
            )
        ),
        encoding=TensorConfig.from_tensor(
            torch.tensor([0, 1], dtype=torch.int8)
        )
    )

    seq_lengths = SeqLengths(
        lengths={
            'pos' : {
                'support' : TensorConfig.from_tensor(torch.arange(0, 10)),
                'pmf' : TensorConfig.from_tensor(data_utils.uniform_pmf(10))
            },
            'neg' : {
                'support' : TensorConfig.from_tensor(torch.arange(0, 5)),
                'pmf' : TensorConfig.from_tensor(data_utils.uniform_pmf(5))
            }
        }
    )

    embedder_cfg = EmbedderConfig(
        ambient_dim=5,
        mean_center=False,
        offset_1=TensorConfig.from_tensor(-torch.tile(torch.tensor([0.5]), (hypercube_args_cfg.num_dims + 3,))), # Plus 3 for the dimensions corresponding to special tokens
        offset_2=None,
        method='random_rotation',
        # noise_distr=torch.distributions.Normal(0, 0.05)
        noise_distr=CallableConfig.from_callable(
            torch.distributions.Normal, 
            NormalDistrConfig(loc=0, scale=0.05),
            kind='class',
            recovery_mode='call'
        )
    )

    # ----------------------------- Sequences config ---------------------------- #
    sequences_cfg = SequencesConfig(
        num_seq=1024,
        seq_order='permute',
        seq_lengths=seq_lengths,
        elem=CallableConfig.from_callable(
            Hypercube, 
            hypercube_args_cfg, 
            kind='class', 
            recovery_mode='call'
        ),
        embedder=CallableConfig.from_callable(
            Embedder,
            embedder_cfg,
            kind='class',
            recovery_mode='call'
        )
    )

    # ------------------------------- Serialize --------------------------------- #
    # Attempt to serialize and reconstruct full cfg tree, and use reconstructed
    # version to build, to ensure future reproducibility.
    data_cfg = DataConfig(
        sequences_cfg=sequences_cfg,
        reproducibility_cfg=reproducibility_cfg
    )
    # serializable_cfg_dict = r_utils.recursive(
    #     data_cfg,
    #     branch_conditionals=(
    #         r_utils.dict_branch, 
    #         r_utils.tuple_branch, 
    #         r_utils.list_branch, 
    #         r_utils.dataclass_branch_with_transform_to_dict
    #     ),
    #     leaf_fns=(
    #         tensor_to_tagged_dict,
    #     )
    # )
    # cfg_filepath = output_dir / (filename + '.json')
    # io_utils.save_to_json(serializable_cfg_dict, cfg_filepath, indent=2)

    # # ------------------------------- Reconstruct ------------------------------- #
    # deserialized_data_cfg_dict = io_utils.load_from_json(cfg_filepath)
    # reconstructed_data_cfg = r_utils.recursive(
    #     deserialized_data_cfg_dict,
    #     branch_conditionals=(
    #         r_utils.dict_branch_with_transform_to_dataclass,
    #         r_utils.tuple_branch, 
    #         r_utils.list_branch, 
    #     ),
    #     leaf_fns=(
    #         lambda x: x,
    #     )
    # )
    cfg_filepath = output_dir / (filename + '.json')
    _ = serialize(data_cfg, cfg_filepath)
    reconstructed_data_cfg = deserialize(cfg_filepath)

    # Recursive instantiation.
    reconstructed_data_cfg = recursive_instantiation(reconstructed_data_cfg)

    # ----------------------------- Build sequences ----------------------------- #
    # See if sequences build without error. Arbitrarily using the train split's 
    # first seed and determinism settings here.
    reproducibility.set_seed(
        **asdict(reconstructed_data_cfg.reproducibility_cfg.seed_cfg_dict['train'][0])
    )
    reproducibility.set_torch_determinism(
        **asdict(reconstructed_data_cfg.reproducibility_cfg.torch_determinisim_cfg_dict['train'])
    )
    # apply_reproducibility_cfg(
    #     seed_cfg=reconstructed_data_cfg.reproducibility_cfg.seed_cfg_dict['train'][0],
    #     torch_determinism_cfg=reconstructed_data_cfg.reproducibility_cfg.torch_determinisim_cfg_dict['train']
    # )
    sequences = build_hypercube_sequences(reconstructed_data_cfg.sequences_cfg) 

    PRINT_TO_CONSOLE = True
    if PRINT_TO_CONSOLE:
        # Retrieve sample from current split.
        seq_idx = 7
        seq, labels, _, _, _ = sequences[seq_idx]

        # Get version where all labels in positive vs negative class are the same.
        labels_uniform_class = labels.clone()
        labels_uniform_class[labels >= 4] = 4
        labels_uniform_class[labels < 0] = -1

        print(f"Shape of single sequence: {seq.shape}.")
        print(f"Shape of corresponding labels: {labels.shape}.")
        print(f"Sequence: \n {seq}.") 
        print(f"Labels by class: \n {labels_uniform_class}.") 
        print(f"Labels by stimulus: \n {labels}.") 
        print("\n")

    # Optionally save sequences.
    SAVE_DATASET = False
    if SAVE_DATASET:
        dataset_filepath = output_dir / (filename + '.pt')
        io_utils.torch_save(sequences, dataset_filepath)






