from datetime import date

import torch

from data import builder as data_builder 
from data.config import DataConfig, EmbedderConfig, HypercubeConfig, NormalDistrConfig, SeqLengths, SequencesConfig, SplitConfig
from data.sequences import Hypercube, Embedder
from data import utils as data_utils
from general_utils.config import CallableConfig, TensorConfig
from general_utils import fileio as fileio_utils
from general_utils import records as records_utils
from general_utils import serialization as serialization_utils
from general_utils import tensor as tensor_utils


def main():
    # --------------------------- Set directory ----------------------------- #
    base_dir = 'configs/datasets'
    # sub_dir_1 = 'aaaa'
    sub_dir_1 = str(date.today())
    sub_dir_2 = 'a'
    output_dir = fileio_utils.make_dir(base_dir, sub_dir_1, sub_dir_2)
    filename = fileio_utils.make_filename('0000')

    # ----------------------- Build auxiliary objects ----------------------- #
    hypercube_args_cfg = HypercubeConfig(
        num_dims=2,
        coords=TensorConfig.from_tensor(
            torch.tensor([0, 1], dtype=torch.int64)
        ),
        inclusion_set=TensorConfig.from_tensor(
            torch.tensor(
                [[0, 1], [1, 0], [1, 1]], dtype=torch.int8
            )
        ),
        encoding=TensorConfig.from_tensor(
            torch.tensor([0, 1], dtype=torch.int8)
        ),
        vertices_pmfs=(
            TensorConfig.from_tensor(
                data_utils.uniform_pmf(3)
            ),
            TensorConfig.from_tensor(
                torch.tensor([1.], dtype=torch.float32)
            )
        )
    )

    seq_lengths = SeqLengths(
        lengths={
            'pos' : {
                'support' : TensorConfig.from_tensor(
                    tensor_utils.single_parity_arange(20, 'even')
                ),
                'pmf' : TensorConfig.from_tensor(
                    data_utils.uniform_pmf(len(tensor_utils.single_parity_arange(20, 'even')))
                )
            },
            'neg' : {
                'support' : TensorConfig.from_tensor(
                    tensor_utils.single_parity_arange(20, 'even')
                ),
                'pmf' : TensorConfig.from_tensor(
                    data_utils.uniform_pmf(len(tensor_utils.single_parity_arange(20, 'even')))
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

    # --------------------------- Sequences config -------------------------- #
    sequences_cfg = SequencesConfig(
        num_seq='num_seq___',
        seq_order='permute',
        seq_lengths=seq_lengths,
        elem=elem,
        embedder=embedder
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
        split_cfg=split_cfg
    )

    data_cfg_filepath = output_dir / (filename + '.json')
    _ = serialization_utils.serialize(data_cfg, data_cfg_filepath)

    # -------------------- Test deserialization/execution ------------------- #
    # Attempt to build dataset from serialized file.
    data_builder.build_sequences_from_filepath(
        data_cfg_filepath, 
        build=['train', 'val'], 
        reproducibility_cfg_filepath='configs/reproducibility/2025-08-11/a/0000.json',
        seed_idx=0, 
        print_to_console=True, 
        save_path=None
    )

    # Deserialize and summarize config to .xlsx file.
    records_utils.summarize_cfg_to_xlsx(
        data_cfg_filepath, 
        config_kind='dataset', 
        config_id=str(data_cfg_filepath).strip('.json'),
        note='',
        xlsx_filepath='configs/logs.xlsx'
    )

if __name__ == '__main__':
    main()




