import copy
from pathlib import Path
import re
from typing import List, Optional, Union

import torch

from data import builder as data_builder
from .training import train
from .eval import evaluate
from models import builder as model_builder
from general_utils import config as config_utils
from general_utils import ml as ml_utils
from general_utils import fileio as fileio_utils
from general_utils import validation as validation_utils


_GPU_PREFIX = re.compile(r'^gpu__:(\d+)$')

def get_id_from_filepath(filepath: str, depth: int, joiner: str ='', prefix='', suffix=''):
    """ 
    Get an ID from a filepath by taking the stem of the file and joining it with
    the names of its parent directories up to a specified depth. The ID is
    prefixed and suffixed with the provided strings.
    
    Parameters:
    -----------
    filepath : str
        The path to the file.
    depth : int
        The number of items (including both stem and parents) to include in the ID.
    joiner : str, optional
        The string to join the parts of the ID. Default is '' (no separator).
    prefix : str, optional
        A string to prepend to the ID. Default is '' (no prefix).
    suffix : str, optional
        A string to append to the ID. Default is '' (no suffix).
    
    Returns:
    --------
    id : str
        The constructed ID.
    """
    path = Path(filepath)
    parts = [path.stem] + [p.name for p in path.parents[:depth-1]]
    return prefix + joiner.join(reversed(parts)).replace('-', '') + suffix

def get_filepath_from_ref(config_kind, ref, file_ext):
    """ 
    Get a filepath from a config kind and reference name.
    
    Parameters:
    -----------
    config_kind : str
        The kind of config (e.g., 'datasets', 'models', etc.).
    ref : str
        The reference for the config (e.g. '2025-08-11/a/0000')
    file_ext : str
        The file extension to append to the filepath.

    Returns:
    --------
    filepath : str
        The constructed filepath.
    
    Example:
    --------
    >>> get_filepath_from_ref('datasets', '2025-08-11/a/0000', '.json')
    'configs/datasets/2025-08-11/a/0000.json'
    """
    return f'configs/{config_kind}/{ref}{file_ext}'

def get_device_name(device_str):
    """ 
    Get device name based on the provided device string.

    Parameters:
    -----------
    device_str : str
        The device string, which can be 'gpu__' for the default GPU, 
        'gpu__:<device_idx>' for a specific GPU index, or any other string 
        representing a device.  
    
    Returns:    
    --------
    device_name : str
        The name of the device to be used in PyTorch, such as 'cuda:0', 'mps:0', or 'cpu'.
    """
    # Check for valid GPU placeholder and retrieve device index.
    validation_utils.validate_str(device_str)
    if device_str == 'gpu__': 
        # Default to index 0 if no index specified.
        device_idx = 0
    else:
        # Check for 'gpu__:<device_idx>' format.
        m = _GPU_PREFIX.fullmatch(device_str)
        if not m:
            # Raise exception if string looks like malformed version of 'gpu__:<device_idx>'.
            if device_str.startswith('gpu__'):
                raise ValueError(
                    f"Invalid device string '{device_str}'. Expected format "
                    f"'gpu__:<device_idx>' but got '{device_str}'."
                )
            return device_str # Pass through
        device_idx = int(m.group(1))

    # `device_str` matches valid format. Check for CUDA first.
    if torch.cuda.is_available():
        if device_idx < 0 or device_idx > torch.cuda.device_count() - 1:
            raise ValueError(
                f"Invalid device index {device_idx} for CUDA. "
                f"Available CUDA devices: {torch.cuda.device_count()}"
            )
        return f'cuda:{device_idx}'
    
    # Check for MPS next.
    mps_backend = getattr(torch.backends, 'mps', None)
    if mps_backend and mps_backend.is_available():
        if device_idx != 0:
            raise ValueError(
                f"Invalid device index {device_idx} for MPS. "
                "MPS only supports a single device (index 0)."
            )
        return 'mps:0' 
    
    # Fallback to CPU.
    return 'cpu'  

def run_and_save_training_from_filepath(
    model_cfg_filepath, 
    data_train_cfg_filepath, 
    training_cfg_filepath, 
    reproducibility_cfg_filepath,
    seed_idx,
    exp_dir, 
    train_run_id_suffix='', # Optional additional identifier appended to end of concatenation of config names, inside the train/ directory
    pretrained_model_filepath=None,
    weights_only=False, 
    test_mode=False
):

    # ------------------------- Build directories --------------------------- #
    # Get train run ID from config names.
    model_id = get_id_from_filepath(model_cfg_filepath, depth=3, prefix='md')
    data_train_id = get_id_from_filepath(data_train_cfg_filepath, depth=3, prefix='dn')
    training_id = get_id_from_filepath(training_cfg_filepath, depth=3, prefix='tr')
    reproducibility_id = get_id_from_filepath(reproducibility_cfg_filepath, depth=3, prefix='rn')
    train_run_id = '_'.join([model_id, data_train_id, training_id, reproducibility_id]) + train_run_id_suffix

    dirs = {}

    # Create train run dir. Configs, models, and logged results all stored here.
    dirs['train_run'] = fileio_utils.make_dir(
         exp_dir, f'seed{seed_idx:02d}', 'train', train_run_id, exist_ok=False
    )
    
    # Passed to metric tracker constructor.
    dirs['checkpoint'] = fileio_utils.make_dir(dirs['train_run'], 'output', 'models')

    # Passed to Logger constructor and location where MetricTracker and
    # EarlyStopping objects will be manually saved.
    dirs['logs'] = fileio_utils.make_dir(dirs['train_run'], 'output', 'logs')
    
    # Used to manually save configs below after they are loaded in.
    dirs['configs'] = fileio_utils.make_dir(dirs['train_run'], 'configs')
    
    # -------------------------- Build dataset ------------------------------ #    
    # Build training, validation, and test splits. Embedding dim needed for 
    # model instantiation below.
    sequences, data_train_cfg_dict, reproducibility_cfg_dict \
        = data_builder.build_sequences_from_filepath(
            data_cfg_filepath=data_train_cfg_filepath,
            reproducibility_cfg_filepath=reproducibility_cfg_filepath,
            build=['train', 'val'],
            seed_idx=seed_idx,
            print_to_console=False
        )

    # ------------------------ Instantiate model ---------------------------- #    
    # Use embedding dimension and tokens to instantiate model.
    model, model_cfg_dict, _, _ = model_builder.build_model_from_filepath(
        model_cfg_filepath=model_cfg_filepath,
        data_cfg_filepath=data_train_cfg_filepath, 
        reproducibility_cfg_filepath=reproducibility_cfg_filepath,
        seed_idx=seed_idx, 
        device='cpu', # No test pass here, model will be moved to device stored in training_cfg.train_fn_cfg in the train function 
        test_pass=False
    )

    if pretrained_model_filepath is not None:
        checkpoint = torch.load(pretrained_model_filepath, weights_only=weights_only)
        model.load_state_dict(checkpoint['model_state_dict'])

    # -------------------- Retrieve train/val config ------------------------ #
    training_cfg_dict = {}
    training_cfg_dict['base'] = config_utils.serialization.deserialize(training_cfg_filepath)
    training_cfg_dict['recovered'] = config_utils.serialization.recursive_recover(training_cfg_dict['base']) 

    # --------------------- Register used (base) configs -------------------- #
    config_utils.serialization.serialize(data_train_cfg_dict['base'], dirs['configs'] / 'data_train.json')
    config_utils.serialization.serialize(model_cfg_dict['base'], dirs['configs'] / 'model.json')
    config_utils.serialization.serialize(training_cfg_dict['base'], dirs['configs'] / 'training.json')
    config_utils.serialization.serialize(reproducibility_cfg_dict['base'], dirs['configs'] / 'reproducibility.json')

    # ------------------ Manually recover deferred items -------------------- #
    # Fill in model parameters (optimizers instantiated here).
    for term in training_cfg_dict['recovered'].train_fn_cfg.loss_terms:
        term.optimizer = term.optimizer.manually_recover(params=model.parameters())

    # Add dataset to DataLoaders and instantiate.
    training_cfg_dict['recovered'].train_fn_cfg.dataloader \
        = training_cfg_dict['recovered'].train_fn_cfg.dataloader.manually_recover(
            dataset = sequences['train']
        )
    training_cfg_dict['recovered'].train_fn_cfg.evaluation['dataloader'] \
        = training_cfg_dict['recovered'].train_fn_cfg.evaluation['dataloader'].manually_recover(
            dataset = sequences['val']
        )
    
    # Set device. String placeholder 'gpu__' will be replaced with actual device name.
    device = get_device_name(
        device_str=training_cfg_dict['recovered'].train_fn_cfg.device.args_cfg.device
    )
    training_cfg_dict['recovered'].train_fn_cfg.device \
        = training_cfg_dict['recovered'].train_fn_cfg.device.manually_recover(
            device=device
        )
    training_cfg_dict['recovered'].train_fn_cfg.evaluation['device'] \
        = training_cfg_dict['recovered'].train_fn_cfg.evaluation['device'].manually_recover(
            device=device
        )
    
    # Add checkpoint_dir to MetricTracker and instantiate.
    training_cfg_dict['recovered'].train_fn_cfg.metric_tracker = \
        training_cfg_dict['recovered'].train_fn_cfg.metric_tracker.manually_recover(
            checkpoint_dir=dirs['checkpoint']
        )
    
    # Add log_dir to Loggers (for train and val) and instantiate.
    training_cfg_dict['recovered'].train_fn_cfg.logger_train \
        = training_cfg_dict['recovered'].train_fn_cfg.logger_train.manually_recover(
            log_dir=dirs['logs']
        )
    training_cfg_dict['recovered'].train_fn_cfg.evaluation['logger'] \
        = training_cfg_dict['recovered'].train_fn_cfg.evaluation['logger'].manually_recover(
            log_dir=dirs['logs']
        )

    # ------------------------ Manual add (no recover) ---------------------- #
    # Add switch label.
    training_cfg_dict['recovered'].train_fn_cfg.evaluation['switch_label'] \
        = sequences['train'].special_tokens['switch']['label'].to(
            training_cfg_dict['recovered'].train_fn_cfg.device
        )
    

    ml_utils.training.set_requires_grad(model, training_cfg_dict['recovered'].requires_grad_cfg)
    

    # --------------------------- Run training ------------------------------ #    
    # Re-apply split seed immediately prior to training 
    ml_utils.reproducibility.apply_reproducibility_settings(
        reproducibility_cfg=reproducibility_cfg_dict['recovered'],
        seed_idx=seed_idx,
        split='train'
    )

    if test_mode:
        train_fn_cfg = copy.deepcopy(training_cfg_dict['recovered'].train_fn_cfg)
        train_fn_cfg.num_epochs=2
        print(
            "Running in test mode, will use a deepcopy of "
            "`training_cfg_dict['recovered'].train_fn_cfg` with num_epochs = 2."
        )
    else:
        train_fn_cfg = training_cfg_dict['recovered'].train_fn_cfg
    
    logger_train, logger_val, metric_tracker, early_stopping = train(
        model, 
        **config_utils.serialization.shallow_asdict(train_fn_cfg)
    )

    # Save logger.
    for logger in logger_train, logger_val:
        if logger is not None:
            logger.convert_to_serializable_format(target=['batch_logs', 'epoch_logs'])
            logger.save()

    # Get final validation epoch log.
    num_epochs_trained = len(logger_val.epoch_logs)
    final_val_epoch_log = logger_val.get_entry(level='epoch', epoch_idx=num_epochs_trained-1)

    # Save metric tracker and early stopping objects.
    torch.save(metric_tracker, dirs['logs'] / 'metric_tracker.pt')
    torch.save(early_stopping, dirs['logs'] / 'early_stopping.pt')

    return {
        'model': model,
        'sequences' : sequences,
        'final_val_epoch_log': final_val_epoch_log,
        'logger_train': logger_train,
        'logger_val': logger_val,
        'metric_tracker': metric_tracker,
        'early_stopping': early_stopping,
        'training_cfg_dict': training_cfg_dict, 
        'model_cfg_dict': model_cfg_dict, 
        'data_cfg_dict': data_train_cfg_dict,
        'train_run_id' : train_run_id,
        'dirs': dirs
    }


def run_testing_from_filepath(
    model_cfg_filepath, 
    data_test_cfg_filepath, 
    testing_cfg_filepath, 
    reproducibility_cfg_filepath,
    seed_idx, 
    model_filepath=None,
    log_dir=None,
    weights_only=False,
    log_outputs_override=False
):
    """ 
    Runs testing process from config filepath names.
    """
    # -------------------------- Build dataset ------------------------------ #    
    # Build training, validation, and test splits. Embedding dim needed for 
    # model instantiation below.
    sequences, data_test_cfg_dict, reproducibility_cfg_dict \
        = data_builder.build_sequences_from_filepath(
            data_cfg_filepath=data_test_cfg_filepath,
            reproducibility_cfg_filepath=reproducibility_cfg_filepath,
            build=['test'],
            seed_idx=seed_idx,
            print_to_console=False
        )

    # ------------------------ Load trained model --------------------------- #   
    model, model_cfg_dict, _, _ = model_builder.build_model_from_filepath(
        model_cfg_filepath=model_cfg_filepath, 
        data_cfg_filepath=data_test_cfg_filepath, # Just need embedding dimension and tokens which should match.
        reproducibility_cfg_filepath=reproducibility_cfg_filepath,
        seed_idx=seed_idx, 
        device='cpu', # No test pass here, model will be moved to device stored in training_cfg.train_fn_cfg in the train function 
        test_pass=False
    )

    # models_dir = run_dir + f"output/seed{seed_idx:02d}/models/"
    checkpoint = torch.load(model_filepath, weights_only=weights_only)
    model.load_state_dict(checkpoint['model_state_dict'])

    # -------------------- Retrieve train/val config ------------------------ #
    testing_cfg_dict = {}
    testing_cfg_dict['base'] = config_utils.serialization.deserialize(testing_cfg_filepath)
    testing_cfg_dict['recovered'] = config_utils.serialization.recursive_recover(testing_cfg_dict['base']) 

    # ------------------ Manually recover deferred items -------------------- #
    # Add dataset to dataloader. If string placeholder was used for batch size,
    # replace with size of testing dataset.
    testing_cfg_dict['recovered'].eval_fn_cfg.dataloader \
        = testing_cfg_dict['recovered'].eval_fn_cfg.dataloader.manually_recover(
            dataset = sequences['test'],
            batch_size=len(sequences['test']) 
            if isinstance(testing_cfg_dict['recovered'].eval_fn_cfg.dataloader.args_cfg.batch_size, str)
            else testing_cfg_dict['recovered'].eval_fn_cfg.dataloader.args_cfg.batch_size
        )
    
    # Set device. String placeholder 'gpu__' will be replaced with actual device name.
    device = get_device_name(
        device_str=testing_cfg_dict['recovered'].eval_fn_cfg.device.args_cfg.device
    )
    testing_cfg_dict['recovered'].eval_fn_cfg.device \
        = testing_cfg_dict['recovered'].eval_fn_cfg.device.manually_recover(
            device=device
        )
    
    # Build and add logger save directory for test logger.
    # log_dir = run_dir + f"output/seed{seed_idx:02d}/"
    testing_cfg_dict['recovered'].eval_fn_cfg.logger \
        = testing_cfg_dict['recovered'].eval_fn_cfg.logger.manually_recover(
            log_dir=('' if log_dir is None else log_dir)
        )
    
    # Hidden states, logits etc. might not be saved at test time to save space.
    # To analyze these, re-run testing from configs, but with the log_outputs_override 
    # arg set to True. The log_outputs arg to the eval function does not affect
    # the logger object's attributes but causes hidden states etc. to be
    # registered to the logger.
    if log_outputs_override:
        testing_cfg_dict['recovered'].eval_fn_cfg.log_outputs = True
    
    # ------------------------ Manual add (no recover) ---------------------- #
    # Add switch label.
    testing_cfg_dict['recovered'].eval_fn_cfg.switch_label \
        = sequences['test'].special_tokens['switch']['label'].to(
            testing_cfg_dict['recovered'].eval_fn_cfg.device
        )

    # ---------------------------- Run testing ------------------------------ # 
    # Re-apply split seed immediately prior to testing.
    # Re-apply split seed immediately prior to training 
    ml_utils.reproducibility.apply_reproducibility_settings(
        reproducibility_cfg=reproducibility_cfg_dict['recovered'],
        seed_idx=seed_idx,
        split='test'
    )

    logger_test = evaluate(
        model, 
        **config_utils.serialization.shallow_asdict(testing_cfg_dict['recovered'].eval_fn_cfg)
    )

    return logger_test, model, sequences, model_cfg_dict, data_test_cfg_dict, testing_cfg_dict, reproducibility_cfg_dict

def get_model_filepath(exp_dir, seed_idx, train_run_id, model_suffix):
    """ 
    """
    models_dir = fileio_utils.get_dir(exp_dir, f'seed{seed_idx:02d}', 'train', train_run_id, 'output', 'models')
    return fileio_utils.get_filepath_with_suffix(models_dir, model_suffix)

def run_and_save_testing_from_filepath(
    model_cfg_filepath, 
    data_test_cfg_filepath, 
    testing_cfg_filepath, 
    reproducibility_cfg_filepath,
    seed_idx, 
    exp_dir, 
    train_run_id,
    test_run_id_suffix='',
    model_filepath=None,
    model_suffix='_best.pt', 
    weights_only=False
):
    """ 
    Wrapper around run_testing_from_filepath that also creates directories and
    orchestrates file operations/saving.
    """
    # ---------------------- Build all directories/paths -------------------- #
    # Build path to models (this is based solely on training params) if not provided.
    if model_filepath is None:
        model_filepath = get_model_filepath(exp_dir, seed_idx, train_run_id, model_suffix)
    
    # Build test run ID.
    data_test_id = get_id_from_filepath(data_test_cfg_filepath, depth=3, prefix='dt')
    testing_id = get_id_from_filepath(testing_cfg_filepath, depth=3, prefix='te')
    reproducibility_id = get_id_from_filepath(reproducibility_cfg_filepath, depth=3, prefix='rt')
    test_run_id = '_'.join([train_run_id, data_test_id, testing_id, reproducibility_id]) + test_run_id_suffix

    dirs = {}

    # Create test run dir. Test results and logger will be stored here.
    dirs['test_run'] = fileio_utils.make_dir(
        exp_dir, f'seed{seed_idx:02d}', 'test', test_run_id, exist_ok=False
    )

    # Build directory to pass to Logger constructor.
    dirs['logs'] = fileio_utils.make_dir(dirs['test_run'], 'output', 'logs')

    # Directory where used base configs will be saved directly.
    dirs['configs'] = fileio_utils.make_dir(dirs['test_run'], 'configs')

    (
        logger_test, 
        model, 
        sequences, 
        model_cfg_dict, 
        data_test_cfg_dict, 
        testing_cfg_dict, 
        reproducibility_cfg_dict
    ) = run_testing_from_filepath(
        model_cfg_filepath=model_cfg_filepath, 
        data_test_cfg_filepath=data_test_cfg_filepath, 
        testing_cfg_filepath=testing_cfg_filepath, 
        reproducibility_cfg_filepath=reproducibility_cfg_filepath,
        seed_idx=seed_idx, 
        log_dir=dirs['logs'],
        model_filepath=model_filepath,
        weights_only=weights_only
    )

    # -------------------- Register used (base) configs --------------------- #
    config_utils.serialization.serialize(model_cfg_dict['base'], dirs['configs'] / 'model.json')
    config_utils.serialization.serialize(data_test_cfg_dict['base'], dirs['configs'] / 'data_test.json')
    config_utils.serialization.serialize(testing_cfg_dict['base'], dirs['configs'] / 'testing.json')
    config_utils.serialization.serialize(reproducibility_cfg_dict['base'], dirs['configs'] / 'reproducibility.json')

    # TODO: Save logger.
    if logger_test is not None:
        logger_test.convert_to_serializable_format(target=['batch_logs', 'epoch_logs'])
        logger_test.save()

    return {
        'logger_test': logger_test, 
        'model' : model,
        'sequences' : sequences,
        'data_test_cfg_dict': data_test_cfg_dict, 
        'model_cfg_dict': model_cfg_dict,
        'testing_cfg_dict': testing_cfg_dict,
        'test_run_id': test_run_id,
        'dirs': dirs
    }

def expand_wildcard_ref(config_dir: Union[Path, str], config_kind, ref: list, sort=False):
    """ 
    """
    config_dir = Path(config_dir)

    #Validate.
    if len(ref) != 1 and any('*' in part for part in ref):
        raise ValueError(
            "Wildcard expansion is currently supported only for one arg "
            f"consisting of a single ref stem, but got {len(ref)} items: {ref}."
        )
    ref_parts = ref[0].split('/')
    if '*' not in ref_parts:
        return ref
    else:
        num_wildcards = ref_parts.count('*')
        if  num_wildcards != 1:
            raise ValueError(
                f"Only 1 wildcard is supported at present, but got {num_wildcards}."
            )
        if ref_parts[-1] != '*':
            raise ValueError(
                f"Wildcard '*' must be in final position of ref but got {ref}."
            )
        
        # Expand into list of refs.
        ref_base = '/'.join(ref_parts[:-1])
        dir_path = config_dir / config_kind / ref_base
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory {dir_path} does not exist.")
        if not dir_path.is_dir():
            raise NotADirectoryError(f"{dir_path} is not a directory.")
        filepaths = fileio_utils.get_filepaths(dir_path, return_as='Path')
        expanded_refs = [ref_base + '/' + str(f.stem) for f in filepaths]

        if sort:
            expanded_refs.sort()
        
        return expanded_refs
        
def run(
    model_cfg_ref_list,
    data_train_cfg_ref_list,
    training_cfg_ref_list,
    data_test_cfg_ref_list,
    testing_cfg_ref_list,
    reproducibility_cfg_ref_list,
    seed_idx_list,
    exp_date,
    exp_id,
    exp_dir_with_seed_exist_ok=False, # Can be set to True for curriculum learning, else leave False to help prevent overwriting.
    run_id_suffix='',
    pretrained_model_filepath_list=None,
    model_suffix='_best.pt',
    weights_only=False,
    cross_test=True
):
    # Wildcard expansion.
    data_train_cfg_ref_list = expand_wildcard_ref('configs', 'datasets', data_train_cfg_ref_list)
    data_test_cfg_ref_list = expand_wildcard_ref('configs', 'datasets', data_test_cfg_ref_list)

    # Build experiment directory.
    exp_dir = fileio_utils.make_dir('experiments', exp_date, exp_id)

    # Validate pretrained_model_filepath_list.
    if pretrained_model_filepath_list is not None:
        validation_utils.validate_iterable_contents(
            pretrained_model_filepath_list, 
            validation_utils.is_str, 
            expected_description='a string'
        )
        if len(pretrained_model_filepath_list) != len(model_cfg_ref_list):
            raise RuntimeError(
                "Length of `pretrained_model_filepath_list` "
                f"({len(pretrained_model_filepath_list)}) does not match that " 
                f"of `model_cfg_ref_list` ({len(model_cfg_ref_list)})."
            )
    else:
        pretrained_model_filepath_list = [None for _ in range(len(model_cfg_ref_list))]

    training = {}
    testing = {}
    for seed_idx in seed_idx_list:
        # Raise exception if a directory for the requested seed already exists.
        exp_dir_with_seed = exp_dir / f'seed{seed_idx:02d}'
        if exp_dir_with_seed.exists() and not exp_dir_with_seed_exist_ok:
            raise FileExistsError(
                f"{exp_dir_with_seed} already exists!"
            )
        for data_train_cfg_ref in data_train_cfg_ref_list:
            for (model_cfg_ref, pretrained_model_filepath) in zip(model_cfg_ref_list, pretrained_model_filepath_list):
                for training_cfg_ref in training_cfg_ref_list:
                    for reproducibility_cfg_ref in reproducibility_cfg_ref_list:

                        # Get config filepaths.
                        data_train_cfg_filepath = get_filepath_from_ref('datasets', data_train_cfg_ref, '.json')
                        model_cfg_filepath = get_filepath_from_ref('models', model_cfg_ref, '.json')
                        training_cfg_filepath = get_filepath_from_ref('training', training_cfg_ref, '.json')
                        reproducibility_cfg_filepath = get_filepath_from_ref('reproducibility', reproducibility_cfg_ref, '.json')

                        training_run = run_and_save_training_from_filepath(
                            data_train_cfg_filepath=data_train_cfg_filepath,
                            model_cfg_filepath=model_cfg_filepath,
                            pretrained_model_filepath=pretrained_model_filepath,
                            training_cfg_filepath=training_cfg_filepath,
                            reproducibility_cfg_filepath=reproducibility_cfg_filepath,
                            seed_idx=seed_idx,
                            exp_dir=exp_dir,
                            train_run_id_suffix=run_id_suffix,
                            test_mode=False
                        )
                        training[training_run['train_run_id']] = training_run

                        
                        for data_test_cfg_ref in data_test_cfg_ref_list:
                            if not cross_test and data_train_cfg_ref != data_test_cfg_ref:
                                continue
                            for testing_cfg_ref in testing_cfg_ref_list:

                                # Get config filepaths.
                                data_test_cfg_filepath = get_filepath_from_ref('datasets', data_test_cfg_ref, '.json')
                                testing_cfg_filepath = get_filepath_from_ref('testing', testing_cfg_ref, '.json')
                                
                                testing_run = run_and_save_testing_from_filepath(
                                    data_test_cfg_filepath=data_test_cfg_filepath,
                                    model_cfg_filepath=model_cfg_filepath,
                                    model_filepath=None, # If None, will build path to models from exp_dir and train_run_id
                                    testing_cfg_filepath=testing_cfg_filepath, 
                                    reproducibility_cfg_filepath=reproducibility_cfg_filepath, # Use same config as for training
                                    seed_idx=seed_idx,
                                    exp_dir=exp_dir,
                                    train_run_id=training_run['train_run_id'], # train_run_id from most recent iteration. This is the only output of the run_and_save_training_from_filepath function that is currently used
                                    test_run_id_suffix=run_id_suffix,
                                    model_suffix=model_suffix,
                                    weights_only=weights_only
                                )
                                testing[testing_run['test_run_id']] = testing_run
                
    return training, testing, exp_dir

def run_curriculum(
    model_cfg_ref_list: List[str],
    pretrained_model_filepath_list: Optional[List[str]],
    data_train_cfg_ref_list: List[str],
    training_cfg_ref_list: List[str],
    data_test_cfg_ref_list: List[str],
    testing_cfg_ref_list: List[str],
    reproducibility_cfg_ref_list: List[str],
    seed_idx_list,
    exp_date,
    exp_id,
    model_suffix='_best.pt',
    weights_only=False
):
    """ 
    """
    # Validate list of seed indices and reproducibility configs.
    validation_utils.validate_iterable_contents(
        seed_idx_list,
        validation_utils.is_int,
        expected_description="an int"
    )
    validation_utils.validate_iterable_contents(
        reproducibility_cfg_ref_list,
        validation_utils.is_str,
        expected_description="a str"
    )

    # Validate args specifying model configs and optional filepaths for pretrained models.
    validation_utils.validate_iterable_contents(
        model_cfg_ref_list,
        validation_utils.is_str,
        expected_description="a string"
    )
    if pretrained_model_filepath_list is not None:
        validation_utils.validate_iterable_contents(
            pretrained_model_filepath_list, 
            validation_utils.is_str,
            expected_description="None"
        )
        if len(model_cfg_ref_list) != len(pretrained_model_filepath_list):
            raise RuntimeError(
                "The lengths of model_cfg_ref_list and pretrained_model_filepath_list "
                f"must match, but got {len(model_cfg_ref_list)} and "
                f"{len(pretrained_model_filepath_list)}, respectively."
            )

    # Validate args specifying training and testing procedures.
    list_lengths = []
    for ref_list in [
        data_train_cfg_ref_list,
        training_cfg_ref_list,
        data_test_cfg_ref_list,
        testing_cfg_ref_list
    ]:
        validation_utils.validate_iterable_contents(
            ref_list,
            validation_utils.is_str,
            expected_description="a string"
        )
        list_lengths.append(len(ref_list))
    if len(set(list_lengths)) != 1:
        raise RuntimeError(
            "data_train_cfg_ref_list, training_cfg_ref_list, data_test_cfg_ref_list, " 
            "testing_cfg_ref_list must all be the same length."
        )

    # Store results.
    curriculum_results = {}

    # Seeds need to be iterated over at this level to prevent triggering error
    # in run function about seed already existing.
    for seed_idx in seed_idx_list:
        # Sequentially call run to execute curriculum.
        for i_stage, (
            data_train_cfg_ref,
            training_cfg_ref,
            data_test_cfg_ref,
            testing_cfg_ref
        ) in enumerate(zip(
            data_train_cfg_ref_list,
            training_cfg_ref_list,
            data_test_cfg_ref_list,
            testing_cfg_ref_list
        )):
            print("----------------------------------------------------------------------")
            print(f"Curriculum stage: {i_stage}")
            print("----------------------------------------------------------------------")

            training, testing, exp_dir = run(
                model_cfg_ref_list=model_cfg_ref_list, # Iterated over within run for each stage
                pretrained_model_filepath_list=pretrained_model_filepath_list, # Iterated over within run for each stage
                data_train_cfg_ref_list=[data_train_cfg_ref],
                training_cfg_ref_list=[training_cfg_ref],
                data_test_cfg_ref_list=[data_test_cfg_ref],
                testing_cfg_ref_list=[testing_cfg_ref],
                reproducibility_cfg_ref_list=reproducibility_cfg_ref_list, # Iterated over within run for each stage
                seed_idx_list=[seed_idx],
                exp_date=exp_date,
                exp_id=exp_id,
                exp_dir_with_seed_exist_ok=True,
                run_id_suffix=f'_c{i_stage:02d}', # append _c00, _c01, etc. for easier tracking of stages of learning
                model_suffix=model_suffix,
                weights_only=weights_only
            )

            # Store all model results for current stage.
            for model_cfg_ref, training_result, testing_result in zip(
                model_cfg_ref_list, training.values(), testing.values()
            ):
                if model_cfg_ref not in curriculum_results:
                    curriculum_results[model_cfg_ref] = {}
                
                curriculum_results[model_cfg_ref][i_stage] = {
                    'training': training_result,
                    'testing' : testing_result
                }

            # Update pretrained_model_filepath variable for next step.
            pretrained_model_filepath_list = [
                fileio_utils.get_filepath_with_suffix(
                    training[train_run_id]['dirs']['checkpoint'], 
                    model_suffix,
                    return_as='str'
                )
                for train_run_id in training.keys()
            ]

    return curriculum_results, pretrained_model_filepath_list, exp_dir


       



       


