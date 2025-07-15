import torch

from data.sequences import Sequences
from engine.driver import run_training_from_filepath
# from engine.utils import Logger, compute_accuracy
from engine import utils as engine_utils
# from engine.train import EarlyStopping, MetricTracker
# from engine.loss import LossTerm, spectral_entropy, wrapped_cross_entropy_loss
# from engine.config import (
#     AdamWConfig, DataLoaderConfig, EarlyStoppingConfig, LoggerConfig, LossTermConfig,
#     MetricTrackerConfig, RequiresGradConfig, TrainFnConfig, TrainValConfig
# )
from engine.config import TrainFnConfig, TrainValConfig
from general_utils.config import CallableConfig, TorchDeviceConfig
from general_utils import fileio as fileio_utils
from general_utils import serialization as serialization_utils
from general_utils import ml as ml_utils


def main():
    # --------------------------- Set directory ----------------------------- #
    base_dir = 'configs/training'
    sub_dir = '__00'
    output_dir = fileio_utils.make_dir(base_dir, sub_dir)
    filename = fileio_utils.make_filename('0000')

    # ----------------------------------------------------------------------- #
    REQUIRES_GRAD_REGISTRY = {
        'no_input_network_freeze_except_for_rnn_input': ml_utils.config.RequiresGradConfig(
            networks={
                'input_network': [],
                'rnn': ['ih'],
                'output_network': []
            },
            mode='exclusion',
            requires_grad=False
        ),
        'with_input_network_freeze_except_for_input_layer': ml_utils.config.RequiresGradConfig(
            networks={
                'input_network': ['0.weight', '0.bias'],
                'rnn': [],
                'output_network': []
            },
            mode='exclusion',
            requires_grad=False
        ),
        'none': ml_utils.config.RequiresGradConfig(
            networks={
                'input_network': [],
                'rnn': [],
                'output_network': []
            },
            mode='inclusion',
            requires_grad=True
        )
    }

    # TODO: check how device is handled; consider making it more robust

    # ------------------------- Set training parameters --------------------- #
    loss_term_1 = CallableConfig.from_callable(
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
                torch.optim.AdamW,
                ml_utils.config.AdamWConfig(
                    lr=0.001, 
                    betas=(0.9, 0.999), 
                    eps=1e-08, 
                    weight_decay=0.01, 
                    amsgrad=False
                ),
                kind='class',
                recovery_mode='call',
                locked=True,
                if_recover_while_locked='warn'
            ),
            mode='train'
        ),
        kind='class',
        recovery_mode='call'
    )

    loss_term_2 = CallableConfig.from_callable(
        ml_utils.loss.LossTerm,
        ml_utils.config.LossTermConfig(
            name='spectral_entropy',
            loss_fn=CallableConfig.from_callable(
                ml_utils.loss.spectral_entropy,
                args_cfg=None,
                kind='function',
                recovery_mode='get_callable'
            ),
            weight=1.,
            optimizer=CallableConfig.from_callable(
                torch.optim.AdamW,
                ml_utils.config.AdamWConfig(
                    lr=0.001, 
                    betas=(0.9, 0.999), 
                    eps=1e-08, 
                    weight_decay=0.01, 
                    amsgrad=False
                ),
                kind='class',
                recovery_mode='call',
                locked=True,
                if_recover_while_locked='warn'
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
            patience=4,
            mode='min',
            min_epochs_before_stopping=20,
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
        if_recover_while_locked='warn'
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
        if_recover_while_locked='warn'
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
        if_recover_while_locked='warn'
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
        if_recover_while_locked='warn'
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
        if_recover_while_locked='warn'
    )

    device = CallableConfig.from_callable(
        torch.device,
        TorchDeviceConfig(
            device=(
                'cuda' if torch.cuda.is_available() 
                else 'mps:0' if torch.backends.mps.is_available() 
                else 'cpu'
            )
        ),
        kind='class',
        recovery_mode='call',
    )

    # Args for the eval.evaluate function called in the train.train function.
    evaluation = dict(
        dataloader=dataloader_val,
        switch_label='switch_label___',
        loss_terms=[loss_term_1],
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
        loss_terms=[loss_term_1],
        evaluation=evaluation,
        # save_validation_logger=True,
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
        num_epochs=3,
        device=device,
        deterministic=False
    )

    train_val_cfg = TrainValConfig(
        train_fn_cfg=train_fn_cfg,
        train_split_seed_idx=0,
        val_split_seed_idx=0,
        requires_grad_cfg=REQUIRES_GRAD_REGISTRY['none']
    )


    # ----------------------------- Serialize ------------------------------- #
    train_val_cfg_filepath = output_dir / (filename + '.json')
    _ = serialization_utils.serialize(train_val_cfg, train_val_cfg_filepath)

    # -------------------- Test deserialization/execution ------------------- #
    (
        model, training, checkpoint_dir, train_val_cfg_dict, model_cfg_dict, data_cfg_dict
    ) = run_training_from_filepath(
        data_cfg_filepath='configs/datasets/__00/0000.json',
        model_cfg_filepath='configs/models/__01/0000.json',
        train_val_cfg_filepath=train_val_cfg_filepath,
        run_dir='experiments/__00/0000/',
        seed_idx=0,
        test_mode=True,
    )


if __name__ == '__main__':
    main()