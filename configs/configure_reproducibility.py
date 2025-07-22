from general_utils.ml import reproducibility as reproducibility_utils
from general_utils.ml.config import ReproducibilityConfig, SeedConfig, TorchDeterminismConfig
from general_utils import fileio as fileio_utils
from general_utils import serialization as serialization_utils


def main():
    # ---------------------------- Set directory ---------------------------- #
    base_dir = 'configs/reproducibility'
    sub_dir = 'aa'
    output_dir = fileio_utils.make_dir(base_dir, sub_dir)
    filename = fileio_utils.make_filename('0000')

    # ----------------------- Reproducibility settings ---------------------- #
    SEED_KINDS = ['recovery', 'train', 'val', 'test']
    seed_lists, _, entropy = reproducibility_utils.generate_seed_sequence(
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
    _ = serialization_utils.serialize(reproducibility_cfg, reproducibility_cfg_filepath)

    # Test deserialization but don't actually apply settings, in case this
    # interferes with other processes in unexpected ways.
    _ = serialization_utils.deserialize(reproducibility_cfg_filepath)

if __name__ == '__main__':
    main()