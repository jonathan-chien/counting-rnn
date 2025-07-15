import copy
from typing import List

import torch

from data import builder as data_builder
from .training import train
from .eval import evaluate
from models import builder as model_builder
from general_utils import fileio as fileio_utils
from general_utils import serialization as serialization_utils


def run_training_from_filepath(data_cfg_filepath, model_cfg_filepath, train_val_cfg_filepath, run_dir, seed_idx, test_mode=False):

    # -------------------------- Build dataset ------------------------------ #    
    # Build training, validation, and test splits. Embedding dim needed for 
    # model instantiation below.
    sequences, data_cfg_dict = data_builder.build_sequences_from_filepath(
        data_cfg_filepath=data_cfg_filepath,
        build=['train', 'val'],
        seed_idx=seed_idx,
        print_to_console=False
    )

    # ------------------------ Instantiate model ---------------------------- #    
    # Use embedding dimension and tokens to instantiate model.
    model, model_cfg_dict, _ = model_builder.build_model_from_filepath(
        data_cfg_filepath=data_cfg_filepath, 
        model_cfg_filepath=model_cfg_filepath, 
        seed_idx=seed_idx, 
        device='cpu', # No test pass here, model will be moved to device stored in train_val_cfg.train_fn_cfg in the train function 
        test_pass=False
    )

    # -------------------- Retrieve train/val config ------------------------ #
    train_val_cfg_dict = {}
    train_val_cfg_dict['base'] = serialization_utils.deserialize(train_val_cfg_filepath)
    train_val_cfg_dict['recovered'] = serialization_utils.recursive_recover(train_val_cfg_dict['base']) 

    # ------------------ Manually recover deferred items -------------------- #
    # Fill in model parameters (optimizers instantiated here).
    for term in train_val_cfg_dict['recovered'].train_fn_cfg.loss_terms:
        term.optimizer = term.optimizer.manually_recover(params=model.parameters())

    # Add dataset to dataloaders (dataloaders instantiated here).
    train_val_cfg_dict['recovered'].train_fn_cfg.dataloader \
        = train_val_cfg_dict['recovered'].train_fn_cfg.dataloader.manually_recover(
            dataset = sequences['train']
        )
    train_val_cfg_dict['recovered'].train_fn_cfg.evaluation['dataloader'] \
        = train_val_cfg_dict['recovered'].train_fn_cfg.evaluation['dataloader'].manually_recover(
            dataset = sequences['val']
        )
    
    # Build and add checkpoint save directory.
    checkpoint_dir = run_dir + f"output/seed{seed_idx:02d}/" + "models"
    train_val_cfg_dict['recovered'].train_fn_cfg.metric_tracker = \
        train_val_cfg_dict['recovered'].train_fn_cfg.metric_tracker.manually_recover(
            checkpoint_dir=checkpoint_dir
        )
    
    # Build and add logger save directory for train and val loggers..
    log_dir = run_dir + f"output/seed{seed_idx:02d}/"
    train_val_cfg_dict['recovered'].train_fn_cfg.logger_train \
        = train_val_cfg_dict['recovered'].train_fn_cfg.logger_train.manually_recover(
            log_dir=log_dir
        )
    train_val_cfg_dict['recovered'].train_fn_cfg.evaluation['logger'] \
        = train_val_cfg_dict['recovered'].train_fn_cfg.evaluation['logger'].manually_recover(
            log_dir=log_dir
        )

    # ------------------------ Manual add (no recover) ---------------------- #
    # Add switch label.
    train_val_cfg_dict['recovered'].train_fn_cfg.evaluation['switch_label'] \
        = sequences['train'].special_tokens['switch']['label'].to(
            train_val_cfg_dict['recovered'].train_fn_cfg.device
        )
    
    # ------------------------ Register used configs ------------------------ #
    # Create config directory and save stuff here.
    config_dir = fileio_utils.make_dir(run_dir + 'config')
    serialization_utils.serialize(data_cfg_dict['base'], config_dir / 'data_train.json')
    serialization_utils.serialize(model_cfg_dict['base'], config_dir / 'model.json')
    serialization_utils.serialize(train_val_cfg_dict['base'], config_dir / 'training.json')

    # --------------------------- Run training ------------------------------ #    
    if test_mode:
        train_fn_cfg = copy.deepcopy(train_val_cfg_dict['recovered'].train_fn_cfg)
        train_fn_cfg.num_epochs=2
        print(
            "Running in test mode, will use a deepcopy of "
            "`train_val_cfg_dict['recovered'].train_fn_cfg` with num_epochs = 2."
        )
    else:
        train_fn_cfg = train_val_cfg_dict['recovered'].train_fn_cfg
    
    training = train(
        model, 
        **serialization_utils.shallow_asdict(train_fn_cfg)
    )

    # Save logger.
    for logger in training[0], training[1]:
        if logger is not None:
            logger.convert_to_serializable_format(target=['batch_logs', 'epoch_logs'])
            logger.save()

    return model, training, checkpoint_dir, train_val_cfg_dict, model_cfg_dict, data_cfg_dict

def run_testing_from_filepath(data_cfg_filepath, model_cfg_filepath, test_cfg_filepath, run_dir, seed_idx, model_suffix='_best.pt', weights_only=False):

    # -------------------------- Build dataset ------------------------------ #    
    # Build training, validation, and test splits. Embedding dim needed for 
    # model instantiation below.
    sequences, data_cfg_dict = data_builder.build_sequences_from_filepath(
        data_cfg_filepath=data_cfg_filepath,
        build=['test'],
        seed_idx=seed_idx,
        print_to_console=False
    )

    # ------------------------ Load trained model --------------------------- #   
    model, model_cfg_dict, _ = model_builder.build_model_from_filepath(
        data_cfg_filepath=data_cfg_filepath, 
        model_cfg_filepath=model_cfg_filepath, 
        seed_idx=seed_idx, 
        device='cpu', # No test pass here, model will be moved to device stored in train_val_cfg.train_fn_cfg in the train function 
        test_pass=False
    )

    models_dir = run_dir + f"output/seed{seed_idx:02d}/models/"
    model_filepath = fileio_utils.get_filepath_with_suffix(models_dir, model_suffix)
    checkpoint = torch.load(model_filepath, weights_only=weights_only)
    model.load_state_dict(checkpoint['model_state_dict'])

    # -------------------- Retrieve train/val config ------------------------ #
    test_cfg_dict = {}
    test_cfg_dict['base'] = serialization_utils.deserialize(test_cfg_filepath)
    test_cfg_dict['recovered'] = serialization_utils.recursive_recover(test_cfg_dict['base']) 

    # ------------------ Manually recover deferred items -------------------- #
    # Add dataset to dataloader.
    test_cfg_dict['recovered'].eval_fn_cfg.dataloader \
        = test_cfg_dict['recovered'].eval_fn_cfg.dataloader.manually_recover(
            dataset = sequences['test']
        )
    
    # Build and add logger save directory for test logger.
    log_dir = run_dir + f"output/seed{seed_idx:02d}/"
    test_cfg_dict['recovered'].eval_fn_cfg.logger \
        = test_cfg_dict['recovered'].eval_fn_cfg.logger.manually_recover(
            log_dir=log_dir
        )
    
    # ------------------------ Manual add (no recover) ---------------------- #
    # Add switch label.
    test_cfg_dict['recovered'].eval_fn_cfg.switch_label \
        = sequences['test'].special_tokens['switch']['label'].to(
            test_cfg_dict['recovered'].eval_fn_cfg.device
        )
    
    # ------------------------ Register used configs ------------------------ #
    # Create config directory and save stuff here.
    config_dir = fileio_utils.make_dir(run_dir + 'config')
    serialization_utils.serialize(test_cfg_dict['base'], config_dir / 'testing.json')

    # ---------------------------- Run testing ------------------------------ #    
    logger_test = evaluate(
        model, 
        **serialization_utils.shallow_asdict(test_cfg_dict['recovered'].eval_fn_cfg)
    )

    # TODO: Save logger.
    if logger_test is not None:
        logger_test.convert_to_serializable_format(target=['batch_logs', 'epoch_logs'])
        logger_test.save()

    return logger_test, data_cfg_dict, model_cfg_dict

def single_train_single_test(
    data_train_cfg_filepath: str,
    model_cfg_filepath: str,
    train_val_cfg_filepath: str,
    data_test_cfg_filepath: str,
    test_cfg_filepath: str,
    exp_id: str,
    run_id: str,
    seed_idx: str,
    weights_only: bool =False
):
    """ 
    """

    run_dir = f'experiments/{exp_id}/{run_id}/'

    model, training, checkpoint_dir, train_val_cfg_dict, model_cfg_dict, data_cfg_dict \
        = run_training_from_filepath(
            data_cfg_filepath=data_train_cfg_filepath,
            model_cfg_filepath=model_cfg_filepath,
            train_val_cfg_filepath=train_val_cfg_filepath,
            run_dir=run_dir,
            seed_idx=seed_idx,
            test_mode=False,
        )

    
    logger_test, data_cfg_dict, model_cfg_dict = run_testing_from_filepath(
        data_cfg_filepath=data_test_cfg_filepath,
        model_cfg_filepath=model_cfg_filepath,
        test_cfg_filepath=test_cfg_filepath,
        run_dir=run_dir,
        seed_idx=seed_idx,
        weights_only=weights_only
    )

    results = {
        'model' : model,
        'training': training,
        'checkpoint_dir': checkpoint_dir, 
        'train_val_cfg_dict': train_val_cfg_dict,
        'model_cfg_dict': model_cfg_dict,
        'data_cfg_dict': data_cfg_dict,
        'testing': logger_test
    }

    torch.save(results, (run_dir + f'output/seed{seed_idx:02d}/results.pt'))

    return results


def single_train_multi_test(
    data_train_cfg_filepath: str,
    model_cfg_filepath: str,
    train_val_cfg_filepath: str,
    data_test_cfg_filepath_list: List[str],
    test_cfg_filepath: str,
    exp_id: str,
    run_id: str,
    seed_idx: str,
    weights_only: bool =False
):
    
    run_dir = f'experiments/{exp_id}/{run_id}/'

    model, training, checkpoint_dir, train_val_cfg_dict, model_cfg_dict, data_cfg_dict \
        = run_training_from_filepath(
            data_cfg_filepath=data_train_cfg_filepath,
            model_cfg_filepath=model_cfg_filepath,
            train_val_cfg_filepath=train_val_cfg_filepath,
            run_dir=run_dir,
            seed_idx=seed_idx,
            test_mode=False,
        )