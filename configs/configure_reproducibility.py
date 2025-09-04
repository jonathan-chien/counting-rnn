from datetime import date

from general_utils import config as config_utils
from general_utils.config.types import SeedConfig
from general_utils import ml as ml_utils
from general_utils.ml.config import ReproducibilityConfig, TorchDeterminismConfig
from general_utils import fileio as fileio_utils


def main():
    # ---------------------------- Set directory ---------------------------- #
    base_dir = 'configs/reproducibility'
    # sub_dir_1 = str(date.today())
    sub_dir_1 = '0000-00-00'
    sub_dir_2 = 'a'
    output_dir = fileio_utils.make_dir(base_dir, sub_dir_1, sub_dir_2)
    filename = fileio_utils.make_filename('0000')

    # ----------------------- Reproducibility settings ---------------------- #
    SEED_KINDS = ['recovery', 'train', 'val', 'test']
    seed_lists, _, entropy = ml_utils.reproducibility.generate_seed_sequence(
        num_children_per_level=[100, len(SEED_KINDS)], entropy=None, dtype='uint32', return_as='int'
    )

    reproducibility_cfg = ReproducibilityConfig(
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
        },
        entropy=entropy,
        seed_cfg_list=[
            {
                split: SeedConfig(torch_seed=seed, cuda_seed=seed)
                for split, seed in zip(SEED_KINDS, repeat)
            }
            for repeat in seed_lists
        ]
    )

    reproducibility_cfg_filepath = output_dir / (filename + '.json')
    _ = config_utils.serialization.serialize(reproducibility_cfg, reproducibility_cfg_filepath)

    # Test deserialization but don't actually apply settings, in case this
    # interferes with other processes in unexpected ways.
    _ = config_utils.serialization.deserialize(reproducibility_cfg_filepath)

if __name__ == '__main__':
    main()