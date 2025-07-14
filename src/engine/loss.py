from typing import Callable, Optional
import warnings

import torch
import torch.nn.functional as F


# class LossTerm:
#     """ 
#     """
#     def __init__(
#         self, 
#         name: str, 
#         loss_fn: Callable[..., torch.Tensor], 
#         weight: float = 1., 
#         optimizer: Optional[torch.optim.Optimizer] = None,
#         mode: str = 'train'
#     ):
#         self.name = name
#         self.loss_fn = loss_fn
#         self.weight = weight
#         self.optimizer = optimizer
#         if mode not in ['train', 'eval']: 
#             raise ValueError(
#                 f"Unrecognized value {mode} for `mode`. Must be in ['train', 'eval']."
#             )
#         elif mode == 'eval' and self.optimizer is not None: 
#             warnings.warn(
#                 f"`optimizer` was passed in as {self.optimizer} but `mode` = "
#                 "'eval'. Attempting to call self.step will results in a "
#                 "RuntimeError exception being raised. Set `optimizer` = None "
#                 "to avoid this warning."
#             )
#         self.mode = mode
#         self.loss = None

#     def compute_loss(self, output, targets, model):
#         self.loss = self.weight * self.loss_fn(output, targets, model)
#         return self.loss.item()
    
#     def step(self):
#         if self.mode == 'eval': raise RuntimeError(
#             "This LossTerm object is in 'eval' mode. Cannot call self.step."
#         )
#         if self.loss == None: 
#             raise RuntimeError("Attempting to step, but no loss has been computed yet.")
#         self.optimizer.zero_grad()
#         self.loss.backward()
#         self.loss = None
#         self.optimizer.step()
        

# def wrapped_cross_entropy_loss(output, target, model=None):
#     # if model is not None: 
#     #     raise ValueError(invalid_input_message(model, 'model'))
#     return F.cross_entropy(output, target)

# def get_weight_hh(model):
#     """ 
#     """
#     if not hasattr(model, 'rnn'):
#         raise AttributeError(
#             "`model` must have an attribute `rnn` but none was found."
#         )
#     if not isinstance(model.rnn, torch.nn.RNN):
#         raise ValueError(
#             "Spectral entropy computation is only available for subclasses of "
#             f"nn.RNN but got {type(model.rnn)}."
#         )
    
#     # Ensure single recurrent weight matrix present.
#     candidates = []
#     for name, param in model.rnn.named_parameters():
#         if '.' in name: raise RuntimeError(
#             "`model.rnn` seems to be an nn.Module object as an attribute. This "
#             "is currently unexpected."
#         )
#         if 'weight_hh' in name: candidates.append(param)
#     if len(candidates) == 0: 
#         raise RuntimeError("`model.rnn` has no recurrent weight matrix.")
#     if len(candidates) > 1:
#         raise ValueError(
#             "Multiple recurrent weight matrices found, but currently only" 
#             "single hidden layer networks are supported."
#         )
    
#     weight_hh = candidates[0]
#     return weight_hh

# def spectral_entropy(output, target, model):
#     """ 
#     """
#     EPS = 1e-12
#     # if output is not None:
#     #     raise ValueError(invalid_input_message(output, 'output'))
#     # if target is not None:
#     #     raise ValueError(invalid_input_message(target, 'target'))
    
#     weight_hh = get_weight_hh(model)
#     _, s, _ = torch.linalg.svd(weight_hh)
#     p = s / torch.sum(s)
#     h = -torch.sum(p * torch.log2(p + EPS))
#     return h

# def nuclear_norm(output, target, model):
#     # if output is not None:
#     #     raise ValueError(invalid_input_message(output, 'output'))
#     # if target is not None:
#     #     raise ValueError(invalid_input_message(target, 'target'))
    
#     weight_hh = get_weight_hh(model)
#     _, s, _ = torch.linalg.svd(weight_hh)
#     return torch.sum(s)

