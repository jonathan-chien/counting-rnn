from dataclasses import dataclass, asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Union

import pandas as pd

from general_utils import serialization as serialization_utils
from general_utils import configops as configops_utils


# Define registry of extractor functions.
REGISTRY = {
    'dataset': {
        'hypercube_dim': 'sequences_cfg.elem.hypercube_args_cfg.num_dims',
        'pos_vertices': 'sequences_cfg.elem.hypercube_args_cfg.inclusion_set',
        'pos_vertices_pmf': 'sequences_cfg.elem.hypercube_args_cfg.vertices_pmfs.0',
        'neg_vertices_pmf': 'sequences_cfg.elem.hypercube_args_cfg.vertices_pmfs.1',
        'pos_seq_lengths': 'sequences_cfg.seq_lengths.lengths.pos.support',
        'pos_seq_lengths_pmf': 'sequences_cfg.seq_lengths.lengths.pos.pmf',
        'neg_seq_lengths': 'sequences_cfg.seq_lengths.lengths.neg.support',
        'neg_seq_lengths_pmf': 'sequences_cfg.seq_lengths.lengths.neg.pmf',
        'train_size': 'split_cfg.train',
        'val_size': 'split_cfg.val',
        'test_size': 'split_cfg.test'
    },
    'model': {
        'input_network': 'input_network.path',
        'input_network_layer_sizes': 'input_network.args_cfg.layer_sizes',
        'input_network_nonlinearities': 'input_network.args_cfg.nonlinearities',
        'input_network_dropouts' : 'input_network.args_cfg.dropouts',
        'rnn_type': 'rnn.path',
        'rnn_input_size': 'rnn.args_cfg.input_size',
        'rnn_hidden_size': 'rnn.args_cfg.hidden_size',
        'rnn_nonlinearity': (
            lambda model_cfg: configops_utils.traverse_dotted_path(model_cfg, 'rnn.args_cfg.nonlinearity')
            if configops_utils.traverse_dotted_path(model_cfg, 'rnn.path').endswith('RNN') 
            else None
        ),
        'readout_network': 'readout_network.path',
        'readout_network_layer_sizes': 'readout_network.args_cfg.layer_sizes',
        'readout_network_nonlinearities': 'readout_network.args_cfg.nonlinearities',
        'readout_network_dropouts': 'readout_network.args_cfg.dropouts',
    },
    'training': {
        'loss_terms': 'train_fn_cfg.loss_terms.*.name',
        'loss_weights': 'train_fn_cfg.loss_terms.*.weight',
        'optimizers': 'train_fn_cfg.loss_terms.*.optimizer.path',
        'learning_rates': 'train_fn_cfg.loss_terms.*.optimizer.args_cfg.lr',
        'train_batch_size': 'train_fn_cfg.dataloader.args_cfg.batch_size',
        'val_batch_size': 'train_fn_cfg.evaluation.dataloader.args_cfg.batch_size',
        'early_stopping_strategy': 'train_fn_cfg.early_stopping.args_cfg.strategy.path',
        'early_stopping_patience': 'train_fn_cfg.early_stopping.args_cfg.strategy.args_cfg.patience',
        'early_stopping_tol': 'train_fn_cfg.early_stopping.args_cfg.strategy.args_cfg.tol',
        'num_epochs': 'train_fn_cfg.num_epochs',
    },
    'testing': {
        'loss_terms': 'eval_fn_cfg.loss_terms.*.name',
        'loss_weights': 'eval_fn_cfg.loss_terms.*.weight',
        'batch_size': 'eval_fn_cfg.dataloader.args_cfg.batch_size',
    }
}

def summarize_cfg(cfg, dotted_path_registry: Dict[str, Union[str, Callable]]) -> Dict[str, Any]:
    # return {
    #     key : configops_utils.traverse_dotted_path(cfg, val) if isinstance(val, str)
    #             else val(cfg) if callable(val) else val
    #     for key, val in dotted_path_registry.items()
    # }
    summary = {}
    for key, val in dotted_path_registry.items():
        if isinstance(val, str):
            summary[key] = configops_utils.traverse_dotted_path(cfg, val)
        elif callable(val):
            summary[key] = val(cfg)
        else:
            raise ValueError(
                f"Expected value for key '{key}' in dotted_path_registry to be either a str or a callable, "
                f"but got {type(val)}."
            )
    return summary



# # Decorator to register extractor functions.
# def register_extractor(config_kind):
#     """Decorator factory."""
#     def register_kind(fn):
#         """Register extractor function under provided key of _EXTRACTORS dict."""
#         _EXTRACTORS[config_kind] = fn
#         return fn
#     return register_kind

# # -------------------------- Extractor functions ----------------------------- #
# # Alias getter function.
# def _get(cfg, dotted_path: str) -> Any:
#     """Get value from config using dotted path."""
#     if not is_dataclass(cfg):
#         raise TypeError(
#             f"Expected config to be a dataclass, but got {type(cfg)}."
#         )
#     return configops_utils.get(cfg, dotted_path)

# @register_extractor('dataset')
# def extract_dataset_summary(data_cfg) -> Dict[str, Any]:
#     return {
#         'hypercube_dim': _get(data_cfg, 'sequences_cfg.elem.hypercube_args_cfg.num_dims'),
#         'pos_vertices': _get(data_cfg, 'sequences_cfg.elem.hypercube_args_cfg.inclusion_set'),
#         'pos_vertices_pmf': _get(data_cfg, 'sequences_cfg.elem.hypercube_args_cfg.vertices_pmfs.0'),
#         'neg_vertices_pmf': _get(data_cfg, 'sequences_cfg.elem.hypercube_args_cfg.vertices_pmfs.1'),
#         'pos_seq_lengths': _get(data_cfg, 'sequences_cfg.seq_lengths.lengths.pos.support'),
#         'pos_seq_lengths_pmf': _get(data_cfg, 'sequences_cfg.seq_lengths.lengths.pos.pmf'),
#         'neg_seq_lengths': _get(data_cfg, 'sequences_cfg.seq_lengths.lengths.neg.support'),
#         'neg_seq_lengths_pmf': _get(data_cfg, 'sequences_cfg.seq_lengths.lengths.neg.pmf'),
#         'train_size': _get(data_cfg, 'data_cfg.split_cfg.train'),
#         'val_size': _get(data_cfg, 'data_cfg.split_cfg.val'),
#         'test_size': _get(data_cfg, 'data_cfg.split_cfg.test')
#     }

# @register_extractor('model')
# def extract_model_summary(model_cfg) -> Dict[str, Any]:
#     return {
#         'input_network' : _get(model_cfg, 'input_network.path'),
#         'input_network_layer_sizes' : _get(model_cfg, 'input_network.args_cfg.layer_sizes'),
#         'input_network_nonlinearities' : _get(model_cfg, 'input_network.args_cfg.nonlinearities'),
#         'input_network_dropouts' : _get(model_cfg, 'input_network.args_cfg.dropouts'),
#         'rnn_type' : _get(model_cfg, 'rnn.path'),
#         'rnn_input_size' : _get(model_cfg, 'rnn.args_cfg.input_size'),
#         'rnn_hidden_size' : _get(model_cfg, 'rnn.args_cfg.hidden_size'),
#         'rnn_nonlinearity' : _get(model_cfg, 'rnn.args_cfg.nonlinearity') if _get(model_cfg, 'rnn.path').endswith('RNN') else None,
#         'readout_network' : _get(model_cfg, 'readout_network.path'),
#         'readout_network_layer_sizes' : _get(model_cfg, 'readout_network.args_cfg.layer_sizes'),
#         'readout_network_nonlinearities' : _get(model_cfg, 'readout_network.args_cfg.nonlinearities'),
#         'readout_network_dropouts' : _get(model_cfg, 'readout_network.args_cfg.dropouts'),
#     }

# @register_extractor('training')
# def extract_training_summary(training_cfg) -> Dict[str, Any]:
#     return {
#         'loss_terms': _get(training_cfg, 'train_fn_cfg.loss_terms.*.name'),
#         'loss_weights': 'train_fn_cfg.loss_terms.*.weight',
#         'optimizers': 'train_fn_cfg.loss_terms.*.optimizer.path',
#         'learning_rates': 'train_fn_cfg.loss_terms.*.optimizer.args_cfg.lr',
#         'train_batch_size': 'train_fn_cfg.dataloader.args_cfg.batch_size',
#         'val_batch_size': 'train_fn_cfg.evaluation.dataloader.args_cfg.batch_size',
#         'early_stopping_strategy': 'train_fn_cfg.early_stopping.args_cfg.strategy.path',
#         'early_stopping_patience': 'train_fn_cfg.early_stopping.args_cfg.strategy.args_cfg.patience',
#         'early_stopping_tol': 'train_fn_cfg.early_stopping.args_cfg.strategy.args_cfg.tol',
#         'num_epochs': 'train_fn_cfg.num_epochs',
#     }


# Dataclass for one row of csv file.
@dataclass(slots=True)
class ConfigSummaryRow:
    timestamp: str
    config_kind: str
    config_id: str
    note: str

def summarize_cfg_to_csv(
    cfg_filepath,
    *,
    config_kind: str,
    config_id: str,
    note: str,
    xlsx_filepath: Path = Path('configs/logs.csv')
):

    cfg = serialization_utils.deserialize(cfg_filepath)

    # Default items in each row.
    base_row = asdict(
        ConfigSummaryRow(
            timestamp = datetime.now().isoformat(sep=' ', timespec='seconds'),
            config_kind=config_kind,
            config_id=config_id,
            note=note
        )
    )

    # extractor = _EXTRACTORS.get(config_kind)
    # if extractor:
    #     summary = extractor(cfg) if extractor else {}
    # else:
    #     raise ValueError(
    #         f"Provided `config_kind` {config_kind} did not match any entries "
    #         "in the _EXTRACTORS dictionaries. These are the registered extractors: "
    #         f"{list(_EXTRACTORS.keys())}."
    #     )
    summary = summarize_cfg(cfg, REGISTRY[config_kind])

    new_row = {**base_row, **summary}
    _append_row_to_sheet_upgrading_shchema(
        xlsx_filepath=xlsx_filepath, 
        sheet_name=config_kind, 
        new_row=new_row
    )

def _append_row_to_sheet_upgrading_shchema(
    *,
    xlsx_filepath: Path, # Path to xlsx file
    sheet_name: str,
    new_row: Dict
):
    # Create directory housing 
    xlsx_filepath.parent.mkdir(parents=True, exist_ok=True)

    # Check for existing sheet and create if it doesn't exist.
    if xlsx_filepath.exists():
        try:
            # Retrieve existing sheet.
            sheet_existing = pd.read_excel(xlsx_filepath, sheet_name=sheet_name, engine='openpyxl')
        except ValueError:
            # Create sheet in existing notebook.
            sheet_existing = pd.DataFrame()
    else:
        # Workbook doesn't exist. Create sheet here first.
        sheet_existing = pd.DataFrame()

    # Union of old and new column entries, with new entries at the end.
    existing_cols = list(sheet_existing.columns)
    new_cols = [col for col in new_row.keys() if col not in existing_cols]
    all_cols = existing_cols + new_cols

    # Fill in blank entry for any existing entries not in new_row, using order of all_cols.
    new_row = {col: new_row.get(col, '') for col in all_cols}
    
    # Concatenate to prepare updated sheet.
    new_row_df = pd.DataFrame([new_row], columns=all_cols)
    if sheet_existing.empty and not existing_cols:
        # Sheet is empty (bc workbook/sheet did not exist, or bc previously
        # existing sheet is completely empty).
        sheet_updated = new_row_df
    else:
        # If sheet is not empty, add new columns and populate with empty string for all rows.
        for col in all_cols:
            if col not in existing_cols:
                sheet_existing[col] = ''
        sheet_updated = pd.concat([sheet_existing, new_row_df], ignore_index=True)

    # Add updated sheet to workbook.
    if xlsx_filepath.exists():
        # Write updated sheet back to existing workbook.
        with pd.ExcelWriter(
            xlsx_filepath, mode='a', engine='openpyxl', if_sheet_exists='replace'
        ) as writer:
            sheet_updated.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        # Workbook does not exist, so create it.
        with pd.ExcelWriter(
            path=xlsx_filepath, mode='w', engine='openpyxl'
        ) as writer:
            sheet_updated.to_excel(writer, sheet_name=sheet_name, index=False)

