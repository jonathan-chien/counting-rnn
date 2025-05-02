import torch

from . import utils
from ..utils import tensor as tensor_utils 


def process_batch_eval(model, h_0, batch_full_seq, labels, lengths, masks, switch_label, loss_fn, criteria, device, deterministic):
    # Get batch of demo phase portions of sequences by indexing up to 
    # latest occurence, across seqeunces, of switch token.
    _, switch_ind = torch.where(labels == switch_label)
    max_switch_idx = torch.max(switch_ind) 
    batch_demo_phase = batch_full_seq[:, :max_switch_idx+1, :]
    
    # Lengths must be on CPU, thus move things to GPU after this step.
    demo_lengths = switch_ind + 1 # Up to and including switch token
    resp_lengths = lengths - switch_ind - 1 # From first count token to EOS, inclusive

    batch_demo_phase = batch_demo_phase.to(device)
    labels = labels.to(device)
    masks = masks.to(device)

    generated, on_input, joined = model.generate(
        batch_demo_phase, 
        input_lengths=demo_lengths,
        h_0=h_0,  
        resp_lengths=resp_lengths,
        deterministic=deterministic,
    )     

    # Get reshaped views for computation of loss.
    gen_logits_reshaped = torch.reshape(
        generated['logits'],
        (-1, generated['logits'].shape[-1])
    )
    gen_resp_masks_reshaped = torch.reshape(generated['resp_masks'], (-1,))
    
    loss = loss_fn(
        gen_logits_reshaped[gen_resp_masks_reshaped, :], 
        labels[masks].to(torch.int64)
    )
    if criteria is not None:
        # Need to get response portions of ground truth but left-aligned
        # at first count token after switch token.
        resp_start_col_ind = switch_ind + 1
        resp_stop_col_ind = switch_ind + 1 + torch.max(resp_lengths)
        resp_col_ind = torch.stack([
            torch.arange(start, stop, device=device) 
            for start, stop in zip(resp_start_col_ind, resp_stop_col_ind)    
        ])
        true_resp_labels_aligned = torch.gather(
            labels, dim=1, index=resp_col_ind
        )

        # Output should be detached.
        performance = {
            criterion_name : criterion(
                generated['labels'],  
                true_resp_labels_aligned,
                generated['resp_masks']
            )
            for criterion_name, criterion in criteria.items()
        }
        # performance = [
        #     criterion(
        #         generated['labels'],  
        #         true_resp_labels_aligned,
        #         generated['resp_masks']
        #     ) 
        #     for criterion in criteria
        # ]
    else:
        performance = None

    return generated, on_input, joined, loss, performance

def evaluate(
    model, 
    dataloader, 
    switch_label,
    loss_fn, 
    logger=None,
    criteria=None,
    h_0=None,
    move_results_to_cpu=True,
    deterministic=True,
    device='cuda',
):
    """ 
    """
    model.to(device)
    model.eval()
    
    running_loss = 0
    running_correct = 0

    h_0 = utils.validate_h_0_config(h_0)

    with torch.no_grad():
        for i_batch, (batch_full_seq, labels, lengths, masks, seq_ind) in enumerate(dataloader):
            
            generated, on_input, joined, loss, performance = process_batch_eval(
                model, 
                h_0, 
                batch_full_seq, 
                labels, 
                lengths, 
                masks, 
                switch_label, 
                loss_fn, 
                criteria, 
                device, 
                deterministic
            )

            batch_size = batch_full_seq.shape[0]
            running_loss += loss.item() * batch_size
            if 'accuracy' not in performance:
                raise KeyError(
                    "Expected 'accuracy' to be a key in the returned performance dict."
                )
            else:
                running_correct += performance['accuracy'] * batch_size

            if logger:
                batch_log = {
                    'generated' : generated,
                    'on_input' : on_input,
                    'joined' : joined,
                    'performance' : performance,
                    'seq_ind' : seq_ind, 
                    'loss' : loss,
                    'batch_size' : batch_full_seq.shape[0]
                }
                if move_results_to_cpu:
                    for key, value in batch_log.items(): 
                        batch_log[key] = tensor_utils.move_to_device(value, 'cpu')

                logger(epoch=0, batch=i_batch, **batch_log)

    mean_loss = running_loss / len(dataloader.dataset)
    mean_accuracy = running_correct / len(dataloader.dataset)

    return mean_loss, mean_accuracy, logger

