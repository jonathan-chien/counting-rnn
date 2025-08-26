from dataclasses import dataclass, asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Union

import pandas as pd

from . import serialization
from . import ops 


@dataclass(slots=True)
class ConfigSummaryRow:
    """Dataclass for one row of csv file."""
    timestamp: str
    config_kind: str
    config_id: str
    note: str

def summarize_cfg(cfg, dotted_path_registry: Dict[str, Union[str, Callable]]) -> Dict[str, Any]:
    """ 
    """
    summary = {}
    for key, val in dotted_path_registry.items():
        if isinstance(val, str):
            summary[key] = ops.traverse_dotted_path(cfg, val)
        elif callable(val):
            summary[key] = val(cfg)
        else:
            raise ValueError(
                f"Expected value for key '{key}' in dotted_path_registry to be either a str or a callable, "
                f"but got {type(val)}."
            )
    return summary

def summarize_cfg_to_xlsx(
    cfg_filepath,
    *,
    config_kind: str,
    config_id: str,
    note: str,
    dotted_path_registry: Dict[str, Union[str, Callable]],
    xlsx_filepath: Union[str, Path] = Path('configs/logs.xlsx')
):
    """ 
    """
    cfg = serialization.deserialize(cfg_filepath)
    xlsx_filepath = Path(xlsx_filepath)

    # Default items in each row.
    base_row = asdict(
        ConfigSummaryRow(
            timestamp = datetime.now().isoformat(sep=' ', timespec='seconds'),
            config_kind=config_kind,
            config_id=config_id,
            note=note
        )
    )

    # Extract summary as dict and add to default items.
    summary = summarize_cfg(cfg, dotted_path_registry)
    new_row = {**base_row, **summary}

    # Append new row to xlsx file.
    _append_row_to_sheet_upgrading_schema(
        xlsx_filepath=xlsx_filepath, 
        sheet_name=config_kind, 
        new_row=new_row
    )

def _append_row_to_sheet_upgrading_schema(
    *,
    xlsx_filepath: Path, # Path to xlsx file
    sheet_name: str,
    new_row: Dict
):
    """ 
    """
    # Create directory housing workbook if it doesn't exist.
    xlsx_filepath.parent.mkdir(parents=True, exist_ok=True)

    # Check existing workbook for existing sheet and create if it doesn't exist.
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

