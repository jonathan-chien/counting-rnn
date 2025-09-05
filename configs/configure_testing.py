from datetime import date

import torch

from data.sequences import Sequences
from engine.config import EvalFnConfig, TestingConfig
from engine.driver import run_and_save_testing_from_filepath
from engine import utils as engine_utils
from general_utils import config as config_utils
from general_utils.config.types import CallableConfig, TorchDeviceConfig
from general_utils import ml as ml_utils
from general_utils import fileio as fileio_utils


def main():
    # --------------------------- Set directory ----------------------------- #
    base_dir = 'configs/testing'
    sub_dir_1 = str(date.today())
    sub_dir_2 = 'a'
    output_dir = fileio_utils.make_dir(base_dir, sub_dir_1, sub_dir_2)
    filename = fileio_utils.make_filename('0000')

    # ------------------------- Set testing parameters ---------------------- #
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
            optimizer=None,
            mode='eval'
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
            optimizer=None,
            mode='eval'
        ),
        kind='class',
        recovery_mode='call'
    )

    logger = CallableConfig.from_callable(
        ml_utils.logging.Logger,
        ml_utils.config.LoggerConfig(
            log_dir='log_dir_name__',
            log_name='test',
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

    dataloader = CallableConfig.from_callable(
        torch.utils.data.DataLoader,
        ml_utils.config.DataLoaderConfig(
            batch_size='dataset_size__', # TODO: allow passing entire eval as single batch.
            shuffle=False,
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
            # device=(
            #     'cuda' if torch.cuda.is_available() 
            #     else 'mps:0' if torch.backends.mps.is_available() 
            #     else 'cpu'
            # )
            device='gpu__'  
        ),
        kind='class',
        recovery_mode='call',
        locked=True
    )

    # Args for the eval.evaluate function called in the test.test function.
    eval_fn_cfg = EvalFnConfig(
        dataloader=dataloader,
        switch_label='switch_label___',
        loss_terms=[loss_term_1],
        logger=logger,
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

    testing_cfg = TestingConfig(
        eval_fn_cfg=eval_fn_cfg,
        test_split_seed_idx=0
    )

    # ----------------------------- Serialize ------------------------------- #
    testing_cfg_filepath = output_dir / (filename + '.json')
    config_utils.serialization.serialize(testing_cfg, testing_cfg_filepath)

    # -------------------- Test deserialization/execution ------------------- #
    run_and_save_testing_from_filepath(
        model_cfg_filepath='configs/models/0000-00-00/b/0000.json',
        # model_filepath='experiments/__00/0000/output/seed00/models/0_best.pt',
        data_test_cfg_filepath='configs/datasets/0000-00-00/a/0000.json',
        testing_cfg_filepath=testing_cfg_filepath,
        reproducibility_cfg_filepath='configs/reproducibility/0000-00-00/a/0000.json',
        seed_idx=0,
        exp_dir='experiments/0000-00-00/0000/',
        train_run_id='md00000000b0000_dr00000000a0000_tr00000000b0000_re00000000a0000',
        model_suffix='_best.pt',
        weights_only=False
    )

    # --------------------------- Summarize config --------------------------- #
    # Registry of items to extract from the config.
    REGISTRY = {
        'loss_terms': 'eval_fn_cfg.loss_terms.*.args_cfg.name',
        'loss_weights': 'eval_fn_cfg.loss_terms.*.args_cfg.weight',
        'batch_size': 'eval_fn_cfg.dataloader.args_cfg.batch_size',
    }

    # Deserialize and summarize config to .xlsx file.
    config_utils.summary.summarize_cfg_to_xlsx(
        testing_cfg_filepath, 
        config_kind='testing', 
        config_id=str(testing_cfg_filepath).removeprefix('configs/datasets/').removesuffix('.json'),
        dotted_path_registry=REGISTRY,
        note='',
        xlsx_filepath='configs/logs.xlsx'
    )


if __name__ == '__main__':
    main()
