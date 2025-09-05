import argparse
from datetime import date
from pathlib import Path

import torch

from data.sequences import Sequences
from engine.driver import run_and_save_training_from_filepath
from engine import utils as engine_utils
from engine.config import TrainFnConfig, TrainingConfig
from general_utils import config as config_utils
from general_utils.config.types import CallableConfig, TorchDeviceConfig
from general_utils import fileio as fileio_utils
from general_utils import ml as ml_utils

def build_arg_parser():
    parser = argparse.ArgumentParser(
        parents=[config_utils.ops.make_parent_parser()]
    )
    parser.add_argument('--idx', type=int, required=True, help="Zero-based integer to use for filename.")
    parser.add_argument('--zfill', type=int, default=4, help="Zero-pad width for filename (default=4).")
    parser.add_argument('--base_dir', default='configs/training')
    parser.add_argument('--sub_dir_1', default=str(date.today()))
    parser.add_argument('--sub_dir_2', default='a')

    return parser

def main():
    args = build_arg_parser().parse_args()

    # --------------------------- Set directory ----------------------------- #
    # base_dir = 'configs/training'
    # sub_dir_1 = str(date.today())
    # sub_dir_2 = 'a'
    # output_dir = fileio_utils.make_dir(base_dir, sub_dir_1, sub_dir_2)
    # filename = fileio_utils.make_filename('0000')
    base_dir = args.base_dir
    sub_dir_1 = args.sub_dir_1
    sub_dir_2 = args.sub_dir_2
    output_dir = fileio_utils.make_dir(base_dir, sub_dir_1, sub_dir_2)
    filename = str(args.idx).zfill(args.zfill)

    # Parse key value pairs from the 'set' channel for runtime CLI injection.
    cli = config_utils.ops.parse_override_kv_pairs(args.ch0 or [])

    # ----------------------------------------------------------------------- #
    REQUIRES_GRAD_REGISTRY = {
        'freeze_all_except_for_rnn_input': ml_utils.config.RequiresGradConfig(
            description="Freeze everything except for RNN input layer",
            networks={
                'input_network': [],
                'rnn': ['ih'],
                'readout_network': []
            },
            mode='exclusion',
            requires_grad=False,
            verbose=True
        ),
        'freeze_all_except_for_input_network_input_layer': ml_utils.config.RequiresGradConfig(
            description="Freeze everything except for input network input layer",
            networks={
                'input_network': ['0.weight', '0.bias'],
                'rnn': [],
                'readout_network': []
            },
            mode='exclusion',
            requires_grad=False,
            verbose=True
        ),
        'none': ml_utils.config.RequiresGradConfig(
            description="No parameters are frozen",
            networks={
                'input_network': [],
                'rnn': [],
                'readout_network': []
            },
            mode='inclusion',
            requires_grad=True,
            verbose=True
        )
    }

    # TODO: check how device is handled; consider making it more robust

    # ------------------------- Set training parameters --------------------- #
    loss_term_0 = CallableConfig.from_callable(
        ml_utils.loss.LossTerm,
        ml_utils.config.LossTermConfig(
            name='cross_entropy',
            loss_fn=CallableConfig.from_callable(
                ml_utils.loss.wrapped_cross_entropy_loss,
                args_cfg=None,
                kind='function',
                recovery_mode='get_callable'
            ),
            weight=1.,
            optimizer=CallableConfig.from_callable(
                torch.optim.Adam,
                ml_utils.config.AdamConfig(
                    lr=0.001, 
                    betas=(0.9, 0.999), 
                    eps=1e-08, 
                    amsgrad=False
                ),
                kind='class',
                recovery_mode='call',
                locked=True,
                if_recover_while_locked='print'
            ),
            mode='train'
        ),
        kind='class',
        recovery_mode='call'
    )

    weight_1_exp = config_utils.ops.select(cli, 'loss_term_1.weight.exp', 0.)
    loss_term_1 = CallableConfig.from_callable(
        ml_utils.loss.LossTerm,
        ml_utils.config.LossTermConfig(
            name='spectral_entropy',
            loss_fn=CallableConfig.from_callable(
                ml_utils.loss.spectral_entropy,
                args_cfg=None,
                kind='function',
                recovery_mode='get_callable'
            ),
            weight=10**weight_1_exp,
            optimizer=CallableConfig.from_callable(
                torch.optim.Adam,
                ml_utils.config.AdamConfig(
                    lr=0.001, 
                    betas=(0.9, 0.999), 
                    eps=1e-08, 
                    amsgrad=False
                ),
                kind='class',
                recovery_mode='call',
                locked=True,
                if_recover_while_locked='print'
            ),
            mode='train'
        ),
        kind='class',
        recovery_mode='call'
    )

    early_stopping = CallableConfig.from_callable(
        ml_utils.training.EarlyStopping,
        ml_utils.config.EarlyStoppingConfig(
            metric_name='cross_entropy_loss',
            strategy=CallableConfig.from_callable(
                ml_utils.training.NoImprovementStopping,
                ml_utils.config.NoImprovementStoppingConfig(
                    patience=8,
                    mode='min',
                    tol=1e-5,
                ),
                kind='class',
                recovery_mode='call'
            ),
            min_epochs_before_stopping=25,
            verbose=True,
            disabled=False
        ),
        kind='class',
        recovery_mode='call'
    )

    metric_tracker = CallableConfig.from_callable(
        ml_utils.training.MetricTracker,
        ml_utils.config.MetricTrackerConfig(
            metric_name='cross_entropy_loss',
            checkpoint_dir='dir_name__',
            mode='min',
            frequency='best'
        ),
        locked=True,
        kind='class',
        recovery_mode='call',
        if_recover_while_locked='print'
    )

    logger_train = CallableConfig.from_callable(
        ml_utils.logging.Logger,
        ml_utils.config.LoggerConfig(
            log_dir='log_dir_name__', # TODO: Add support for directory
            log_name='train',
            verbose_batch=False,
            verbose_epoch=True,
            print_flush_epoch=False,
            print_flush_batch=False
        ),
        kind='class',
        recovery_mode='call',
        locked=True,
        if_recover_while_locked='print'
    )

    dataloader_train = CallableConfig.from_callable(
        torch.utils.data.DataLoader,
        ml_utils.config.DataLoaderConfig(
            batch_size=128,
            shuffle=True,
            collate_fn=CallableConfig.from_callable(
                Sequences.pad_collate_fn,
                args_cfg=None,
                kind='static_method',
                recovery_mode='get_callable'
            )
        ),
        kind='class',
        recovery_mode='call',
        locked=True,
        if_recover_while_locked='print'
    )

    # Validation.
    logger_val = CallableConfig.from_callable(
        ml_utils.logging.Logger,
        ml_utils.config.LoggerConfig(
            log_dir='log_dir_name__',
            log_name='val',
            verbose_batch=False,
            verbose_epoch=True,
            print_flush_epoch=False,
            print_flush_batch=False
        ),
        kind='class',
        recovery_mode='call',
        locked=True,
        if_recover_while_locked='print'
    )

    dataloader_val = CallableConfig.from_callable(
        torch.utils.data.DataLoader,
        ml_utils.config.DataLoaderConfig(
            batch_size=128,
            shuffle=True,
            collate_fn=CallableConfig.from_callable(
                Sequences.pad_collate_fn,
                args_cfg=None,
                kind='static_method',
                recovery_mode='get_callable'
            )
        ),
        kind='class',
        recovery_mode='call',
        locked=True,
        if_recover_while_locked='print'
    )

    device = CallableConfig.from_callable(
        torch.device,
        TorchDeviceConfig(
            device='gpu__'
        ),
        kind='class',
        recovery_mode='call',
        locked=True
    )

    # Args for the eval.evaluate function called in the train.train function.
    evaluation = dict(
        dataloader=dataloader_val,
        switch_label='switch_label___',
        loss_terms=[loss_term_0],
        logger=logger_val,
        log_outputs=False,
        criteria={
            'accuracy' : CallableConfig.from_callable(
                engine_utils.compute_accuracy,
                args_cfg=None,
                kind='function',
                recovery_mode='get_callable',
                locked=False
            )
        },
        compute_mean_for=['cross_entropy_loss', 'accuracy'],
        h_0=None,
        deterministic=True,
        device=device,
        move_results_to_cpu=True,
    )

    train_fn_cfg = TrainFnConfig(
        dataloader=dataloader_train,
        loss_terms=[loss_term_0],
        evaluation=evaluation,
        h_0=None,
        logger_train=logger_train,
        criteria={
            'accuracy' : CallableConfig.from_callable(
                engine_utils.compute_accuracy,
                args_cfg=None,
                kind='function',
                recovery_mode='get_callable',
                locked=False
            )
        },
        compute_mean_for=['cross_entropy_loss', 'accuracy'],
        metric_tracker=metric_tracker,
        early_stopping=early_stopping,
        num_epochs=500,
        device=device,
        deterministic=True
    )

    training_cfg = TrainingConfig(
        train_fn_cfg=train_fn_cfg,
        train_split_seed_idx=0,
        val_split_seed_idx=0,
        requires_grad_cfg=REQUIRES_GRAD_REGISTRY['none']
    )


    # ----------------------------- Serialize ------------------------------- #
    training_cfg_filepath = output_dir / (filename + '.json')
    _ = config_utils.serialization.serialize(training_cfg, training_cfg_filepath)

    # -------------------- Test deserialization/execution ------------------- #
    # First delete, any file with suffix best, as a different epoch being best
    # will cause an error during the test part of configure_testing.py.
    model_dir = Path('experiments/0000-00-00/0000/output/seed00/models/')
    for file in model_dir.glob('*_best.pt'):
        file.unlink()

    _ = run_and_save_training_from_filepath(
        model_cfg_filepath='configs/models/0000-00-00/a/0000.json',
        data_train_cfg_filepath='configs/datasets/0000-00-00/a/0000.json',
        training_cfg_filepath=training_cfg_filepath,
        reproducibility_cfg_filepath='configs/reproducibility/0000-00-00/a/0000.json',
        exp_dir='experiments/0000-00-00/0000/',
        seed_idx=0,
        test_mode=True,
    )

    # --------------------------- Summarize config --------------------------- #
    # Registry of items to extract from the config.
    REGISTRY = {
        'loss_terms': 'train_fn_cfg.loss_terms.*.args_cfg.name',
        'loss_weights': 'train_fn_cfg.loss_terms.*.args_cfg.weight',
        'optimizers': 'train_fn_cfg.loss_terms.*.args_cfg.optimizer.path',
        'learning_rates': 'train_fn_cfg.loss_terms.*.args_cfg.optimizer.args_cfg.lr',
        'train_batch_size': 'train_fn_cfg.dataloader.args_cfg.batch_size',
        'val_batch_size': 'train_fn_cfg.evaluation.dataloader.args_cfg.batch_size',
        'early_stopping_strategy': 'train_fn_cfg.early_stopping.args_cfg.strategy.path',
        'early_stopping_patience': 'train_fn_cfg.early_stopping.args_cfg.strategy.args_cfg.patience',
        'early_stopping_tol': 'train_fn_cfg.early_stopping.args_cfg.strategy.args_cfg.tol',
        'num_epochs': 'train_fn_cfg.num_epochs',
    }

    # Deserialize and summarize config to .xlsx file.
    config_utils.summary.summarize_cfg_to_xlsx(
        training_cfg_filepath, 
        config_kind='training', 
        config_id=str(training_cfg_filepath).removeprefix('configs/datasets/').removesuffix('.json'),
        dotted_path_registry=REGISTRY,
        note='',
        xlsx_filepath='configs/logs.xlsx'
    )

if __name__ == '__main__':
    main()