import torch
from typing import Callable

from .config import InitializationConfig


def for_params_in_module(model, module_name: str, param_pattern: str, action):
    for m_name, module in model.named_modules():
        if m_name != module_name:
            continue

        with torch.no_grad():
            for p_name, param in module.named_parameters():
                if param_pattern in p_name:
                    action(param)


def scale_params_by_module_name(
    model, 
    module_name: str,
    param_pattern: str,
    alpha: float

):
    def action(param):
        param.mul_(alpha)

    for_params_in_module(model, module_name, param_pattern, action)
    

def init_param_distr_by_module_name(
    model, 
    module_name: str, 
    param_pattern: str, 
    init_distr: str,
    distr_params: dict
):
    def action(param):
        if init_distr == 'gaussian':
            param.normal_(**distr_params)
        elif init_distr == 'uniform':
            param.uniform_(**distr_params)
        else:
            raise ValueError(
                f"Unrecognized value {init_distr} for `init_distr`. Must be one "
                "of ['gaussian', 'uniform']."
            )
    for_params_in_module(model, module_name, param_pattern, action)

def apply_init_fn_by_module_name(
    model, 
    module_name: str, 
    param_pattern: str, 
    init_fn: Callable,
    init_params: dict
):
    def action(param):
        init_fn(param, **init_params)
    for_params_in_module(model, module_name, param_pattern, action)

def init_params_from_cfg(model, cfg: InitializationConfig):
    """ 
    """
    if cfg is None:
        return model
    
    for step in cfg.steps:
        # manually_recover is called because the recover step needs to be
        # performed here to call the stored function with the stored args. Thus
        # locking prevents premature triggering during potential calls to
        # recursive_recover.
        step.manually_recover(model=model)

    return model
    

        
      
