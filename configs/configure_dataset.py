import torch

from data import builder as data_builder 
from data.config import DataConfig, EmbedderConfig, HypercubeConfig, NormalDistrConfig, SeqLengths, SequencesConfig, SplitConfig
from data.sequences import Hypercube, Embedder
from data import utils as data_utils
from general_utils.config import ReproducibilityConfig, SeedConfig, TorchDeterminismConfig, CallableConfig, TensorConfig
from general_utils import fileio as fileio_utils
from general_utils import reproducibility as reproducibility_utils
from general_utils import serialization as serialization_utils


def main():
    # --------------------------- Set directory ----------------------------- #
    base_dir = 'configs/datasets'
    sub_dir = 'aa00'
    output_dir = fileio_utils.make_dir(base_dir, sub_dir)
    filename = fileio_utils.make_filename('0000')

    # ---------------------- Reproducibility settings ----------------------- #
    SEED_KINDS = ['recovery', 'train', 'val', 'test']
    seed_lists, _, entropy = reproducibility_utils.generate_seed_sequence(
        num_children_per_level=[50, len(SEED_KINDS)], entropy=None, dtype='uint32', return_as='int'
    )

    reproducibility_cfg = ReproducibilityConfig(
        entropy=entropy,
        seed_cfg_list=[
            {
                split: SeedConfig(torch_seed=seed, cuda_seed=seed)
                for split, seed in zip(SEED_KINDS, repeat)
            }
            for repeat in seed_lists
        ],
        torch_determinism_cfg_dict={
            'recovery' : TorchDeterminismConfig(
                use_deterministic_algos=False,
                cudnn_deterministic=False,
                cudnn_benchmark=True
            ),
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

    # ----------------------- Build auxiliary objects ----------------------- #
    hypercube_args_cfg = HypercubeConfig(
        num_dims=2,
        coords=TensorConfig.from_tensor(
            torch.tensor([0, 1], dtype=torch.int64)
        ),
        inclusion_set=TensorConfig.from_tensor(
            torch.tensor(
                [[0, 1], [1, 0], [1, 1]],
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
                'support' : TensorConfig.from_tensor(torch.tensor([5, 7, 9])),
                'pmf' : TensorConfig.from_tensor(data_utils.uniform_pmf(3))
            },
            'neg' : {
                'support' : TensorConfig.from_tensor(torch.arange(3)),
                'pmf' : TensorConfig.from_tensor(data_utils.uniform_pmf(3))
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
            NormalDistrConfig(loc=0, scale=0.05),
            kind='class',
            recovery_mode='call'
        )
    )

    # --------------------------- Sequences config -------------------------- #
    sequences_cfg = SequencesConfig(
        num_seq='num_seq___',
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

    # ---------------------------- Split sizes ------------------------------ #
    split_cfg = SplitConfig(
        train=1000,
        val=500,
        test=500
    )

    # ----------------------------- Serialize ------------------------------- #
    # Attempt to serialize and reconstruct full cfg tree, and use reconstructed
    # version to build, to ensure future reproducibility.
    data_cfg = DataConfig(
        sequences_cfg=sequences_cfg,
        split_cfg=split_cfg,
        reproducibility_cfg=reproducibility_cfg
    )

    data_cfg_filepath = output_dir / (filename + '.json')
    _ = serialization_utils.serialize(data_cfg, data_cfg_filepath)


    # -------------------- Test deserialization/execution ------------------- #
    # Attempt to build dataset from serialized file.
    data_builder.build_sequences_from_filepath(data_cfg_filepath, build=['train', 'val'], seed_idx=0, print_to_console=True, save_path=None)


if __name__ == '__main__':
    main()




