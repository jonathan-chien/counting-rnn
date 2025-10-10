import torch
import torch.nn.functional as F
from typing import Callable, Optional
import warnings


class LossTerm:
    """ 
    """
    def __init__(
        self, 
        name: str, 
        loss_fn: Callable[..., torch.Tensor], 
        weight: float = 1., 
        optimizer: Optional[torch.optim.Optimizer] = None,
        mode: str = 'train'
    ):
        self.name = name
        self.loss_fn = loss_fn
        self.weight = weight
        self.optimizer = optimizer
        if mode not in ['train', 'eval']: 
            raise ValueError(
                f"Unrecognized value {mode} for `mode`. Must be in ['train', 'eval']."
            )
        elif mode == 'eval' and self.optimizer is not None: 
            warnings.warn(
                f"`optimizer` was passed in as {self.optimizer} but `mode` = "
                "'eval'. Attempting to call self.step will results in a "
                "RuntimeError exception being raised. Set `optimizer` = None "
                "to avoid this warning."
            )
        self.mode = mode
        self.loss = None

    def compute_loss(self, output, targets, model):
        self.loss = self.weight * self.loss_fn(output, targets, model)
        return self.loss.item()
    
    def step(self):
        if self.mode == 'eval': raise RuntimeError(
            "This LossTerm object is in 'eval' mode. Cannot call self.step."
        )
        if self.loss == None: 
            raise RuntimeError("Attempting to step, but no loss has been computed yet.")
        self.optimizer.zero_grad()
        self.loss.backward()
        self.loss = None
        self.optimizer.step()


def wrapped_cross_entropy_loss(output, target, model=None):
    # if model is not None: 
    #     raise ValueError(invalid_input_message(model, 'model'))
    cross_entropy = torch.nn.CrossEntropyLoss()
    return cross_entropy(output, target)
    # return F.cross_entropy(output, target)

def get_weight_hh(model):
    """ 
    """
    if not hasattr(model, 'rnn'):
        raise AttributeError(
            "`model` must have an attribute `rnn` but none was found."
        )
    if not isinstance(model.rnn, torch.nn.RNN):
        raise ValueError(
            "Spectral entropy computation is only available for subclasses of "
            f"nn.RNN but got {type(model.rnn)}."
        )
    
    # Ensure single recurrent weight matrix present.
    candidates = []
    for name, param in model.rnn.named_parameters():
        if '.' in name: raise RuntimeError(
            "`model.rnn` seems to be an nn.Module object as an attribute. This "
            "is currently unexpected."
        )
        if 'weight_hh' in name: candidates.append(param)
    if len(candidates) == 0: 
        raise RuntimeError("`model.rnn` has no recurrent weight matrix.")
    if len(candidates) > 1:
        raise ValueError(
            "Multiple recurrent weight matrices found, but currently only" 
            "single hidden layer networks are supported."
        )
    
    weight_hh = candidates[0]
    return weight_hh

def spectral_entropy(output, target, model):
    """ 
    """
    weight_hh = get_weight_hh(model)
    eps = torch.finfo(weight_hh.dtype).eps

    s = torch.linalg.svdvals(weight_hh)

    denominator = torch.sum(s).clamp(min=eps)
    p = (s / denominator).clamp(min=eps)
    h = -torch.sum(p * torch.log(p))

    return h

def nuclear_norm(output, target, model):
    
    weight_hh = get_weight_hh(model)
    s = torch.linalg.svdvals(weight_hh)
    return torch.sum(s)

def spectral_norm_power_method(output, target, model):

    weight_hh = get_weight_hh(model)


def alt_power_method(w, num_iter=10, num_init=50, method='average'):

    eps = torch.finfo(w.dtype).eps

    if len(w.shape) != 2:
        raise ValueError(f"w should be a tensor of dim 2, but got dim {len(w.shape)}.")
    _, n = w.shape # w is of shape (m, n)

    # Initialize random vectors.
    v = torch.randn(n, num_init) # W*W
    v /= v.norm(p=2, dim=0, keepdim=True).clamp(min=eps)

    for i_iter in range(num_iter):
        u = w @ v
        u /= u.norm(p=2, dim=0).clamp(min=eps)

        v = w.mH @ u
        v /= v.norm(p=2, dim=0, keepdim=True).clamp(min=eps)


    if method == 'best':
        y = w @ v
        sigmas = y.norm(p=2, dim=0)

        sigma_hat, best_idx = torch.max(sigmas, dim=0)
        u_hat = y[:, best_idx] / sigma_hat
        v_hat = v[:, best_idx]
        
        return sigma_hat.squeeze(), u_hat, v_hat
    
    if method == 'average':
        # Align phase/(sign in real valued case) to that of first vector. First
        # compute relative phase. 
        # u_aligned = u * torch.inner(u[:, 0].conj(), u.T)
        # v_aligned = v * torch.inner(v[:, 0].conj(), v.T)
        a = u[:, 0].conj() @ u # (1, n) @ (n, num_inits)
        b = v[:, 0].conj() @ v # (1, m) @ (m, num_inits)

        rel_phase_u = a / torch.abs(a).clamp(min=eps)
        rel_phase_v = b / torch.abs(b).clamp(min=eps)

        u_aligned = rel_phase_u.conj() * u
        v_aligned = rel_phase_v.conj() * v
    
        u_hat = torch.mean(u_aligned, dim=1)
        v_hat = torch.mean(v_aligned, dim=1)
        u_hat /= u_hat.norm(p=2).clamp(min=eps)
        v_hat /= v_hat.norm(p=2).clamp(min=eps)

        return u_hat.conj() @ w @ v_hat, u_hat, v_hat

    elif method == 'none':
        if num_init != 1:
            raise ValueError(
                f"If method == 'none', num_init must be 1, but got num_init={num_init}."
            )
        u_hat = u
        v_hat = v

        return u_hat[:, 0].conj() @ w @ v_hat[:, 0], u_hat[:, 0], v_hat[:, 0]
    else:
        raise ValueError(
            f"Unrecognized value for `method` {method}. Must be in ('average', 'best', 'none')."
        )
