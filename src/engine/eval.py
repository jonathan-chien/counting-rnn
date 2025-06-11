from itertools import chain

import torch

from . import utils
from utils import tensor as tensor_utils 


def process_batch_eval(
    model, 
    h_0, 
    batch_full_seq, 
    labels, 
    lengths, 
    masks, 
    switch_label, 
    deterministic,
    move_results_to_cpu=False,
    detach=False
):
    # XXX: The below code is from the original implementation (note that
    # everything ran in a big flat `train` function.). switch_ind is derived
    # from labels and used to build both the lengths and batch_demo_phase.
    # Therefore, if labels were passed in on the GPU, this would cause an error
    # since the lengths would be on the GPU too (and they need to be on the CPU
    # due to packing inputs for the RNN). Thus, moving the labels was
    # originally deferred till after creation of lengths, and masks and
    # batch_demo_phase were moved with the labels together so that all the
    # movements happen in one place. The pattern is now going to be reversed
    # here in the sense that all items will be moved to the GPU in the outer
    # `evaluate` function, right after they are provided by the dataloader, and
    # only the lengths will be moved to the CPU after they are created here.
    # # Get batch of demo phase portions of sequences by indexing up to 
    # # latest occurence, across seqeunces, of switch token.
    # _, switch_ind = torch.where(labels == switch_label)
    # max_switch_idx = torch.max(switch_ind) 
    # batch_demo_phase = batch_full_seq[:, :max_switch_idx+1, :]
    
    # # Lengths must be on CPU, thus move things to GPU after this step.
    # demo_lengths = switch_ind + 1 # Up to and including switch token
    # resp_lengths = lengths - switch_ind - 1 # From first count token to EOS, inclusive

    # batch_demo_phase = batch_demo_phase.to(device)
    # labels = labels.to(device)
    # masks = masks.to(device)

    # Get batch of demo phase portions of sequences by indexing up to 
    # latest occurence, across seqeunces, of switch token.
    _, switch_ind = torch.where(labels == switch_label)
    max_switch_idx = torch.max(switch_ind) 
    batch_demo_phase = batch_full_seq[:, :max_switch_idx+1, :]
    
    # Lengths must be on CPU.
    demo_lengths = switch_ind + 1 # Up to and including switch token
    resp_lengths = lengths - switch_ind - 1 # From first count token to EOS, inclusive
    demo_lengths = demo_lengths.to('cpu')
    resp_lengths = resp_lengths.to('cpu')

    generated, on_input, joined = model.generate(
        batch_demo_phase, 
        input_lengths=demo_lengths,
        h_0=h_0,  
        resp_lengths=resp_lengths,
        deterministic=deterministic,
    )    

    outputs = {
        'generated' : generated, 
        'on_input' : on_input, 
        'joined' : joined,
        'switch_ind' : switch_ind,
        'resp_lengths' : resp_lengths,
        'labels' : labels,
        'masks' : masks
    }

    if move_results_to_cpu or detach: 
        outputs = tensor_utils.recursive(
            outputs, tensor_utils.move_to_device('cpu'), tensor_utils.detach
        )

    return outputs

def evaluate_outputs(
    output, 
    model, 
    loss_terms, 
    criteria, 
    move_results_to_cpu=False,
    detach=False
):
    """ 
    """
    generated = output['generated']
    labels = output['labels']
    switch_ind = output['switch_ind']
    resp_lengths = output['resp_lengths']
    masks = output['masks']

    # Get reshaped views for computation of loss.
    gen_logits_reshaped = torch.reshape(
        generated['logits'],
        (-1, generated['logits'].shape[-1])
    )
    gen_resp_masks_reshaped = torch.reshape(generated['resp_masks'], (-1,))
    losses = {
        name : loss_term.compute_loss(
            output=gen_logits_reshaped[gen_resp_masks_reshaped, :],
            targets=labels[masks].to(torch.int64),
            model=model
        )
        for name, loss_term in loss_terms.items()
    }

    # Need to get response portions of ground truth but left-aligned
    # at first count token after switch token.
    resp_start_col_ind = switch_ind + 1
    resp_stop_col_ind = switch_ind + 1 + torch.max(resp_lengths)
    resp_col_ind = torch.stack([
        torch.arange(start, stop, device=labels.device) 
        for start, stop in zip(resp_start_col_ind, resp_stop_col_ind)    
    ])
    true_resp_labels_aligned = torch.gather(
        labels, dim=1, index=resp_col_ind
    )

    performance = {
        criterion_name : criterion(
            generated['labels'],  
            true_resp_labels_aligned,
            generated['resp_masks']
        )
        for criterion_name, criterion in criteria.items()
    }
   
    if move_results_to_cpu or detach:
        losses = tensor_utils.recursive(
            losses, 
            tensor_utils.move_to_device(
                'cpu' if move_results_to_cpu else generated['logits'].device
            ),
            tensor_utils.detach
        )
        performance = tensor_utils.recursive(
            performance, 
            tensor_utils.move_to_device(
                'cpu' if move_results_to_cpu else generated['labels'].device
            ),
            tensor_utils.detach
        )
            # losses = tensor_utils.move_to_device(losses, 'cpu', detach=detach)
            # performance = tensor_utils.move_to_device(performance, 'cpu', detach=detach)

    return losses, performance

def evaluate(
    model, 
    dataloader, 
    switch_label,
    loss_terms, 
    logger,
    compute_mean_for=None,
    log_outputs=True, # Whether or not to log outputs like logits, hidden states etc. in addition to loss/accuracy
    criteria={'accuracy' : utils.compute_accuracy},
    h_0=None,
    deterministic=True,
    device='cuda',
    move_results_to_cpu=True,
    verbose=True
):
    """ 
    Logger instance is required, since the point of this function is to evaluate and record the performance/loss etc.
    """
    model.to(device)
    model.eval()

    h_0 = utils.validate_h_0_config(h_0)

    with torch.no_grad():
        for i_batch, (batch_full_seq, labels, lengths, masks, seq_ind) in enumerate(dataloader):
            # Move everything to specified device here. 
            batch_full_seq = batch_full_seq.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)
            masks = masks.to(device)
            seq_ind = seq_ind.to(device)
            
            # Process input and generate output.
            outputs = process_batch_eval(
                model, 
                h_0, 
                batch_full_seq, 
                labels, 
                lengths, 
                masks, 
                switch_label, 
                deterministic,
                move_results_to_cpu=False
            )

            # Evaluate generated output.
            losses, performance = evaluate_outputs(
                outputs, 
                model, 
                loss_terms, 
                criteria, 
                move_results_to_cpu=False,
            )

            # Optionally log outputs.
            if log_outputs:
                batch_log = {
                    'generated' : outputs['generated'],
                    'on_input' : outputs['on_input'],
                    'joined' : outputs['joined'],
                    'seq_ind' : seq_ind, 
                    'batch_size' : batch_full_seq.shape[0]
                }
            else:
                batch_log = {}

            # Log the loss values.
            batch_log.update(
                {f'{name}_loss' : value for name, value in losses.items()}
            )

            # Log the performance metrics. name, output['value'] logs the 
            # scalar metric, e.g. accuracy, whereas name_diagnostics, output
            # logs full dict output of the criterion function.
            batch_log.update(dict(
                chain.from_iterable(
                    [
                        (name, output['value']),
                        (f'{name}_diagnostics', output)
                    ]
                    for name, output in performance.items()
                )     
            ))
            
            if move_results_to_cpu:
                batch_log = tensor_utils.recursive(
                    batch_log, 
                    tensor_utils.move_to_device('cpu')
                )

            logger.log_batch(epoch=0, batch=i_batch, **batch_log, verbose=False)

        # Compute mean across batches for specified losses and performance metrics.
        batch_sizes = logger.get_logged_values(key='batch_size', level='batch')
        mean_values = {
            key : logger.compute_weighted_sum(
                key=key,
                level='batch',
                weights=batch_sizes/len(dataloader.dataset)
            )
            for key in batch_log.keys() 
            if compute_mean_for is not None  
            and key in compute_mean_for
        }

        # Optionally print.
        if len(mean_values) > 0 and verbose:
            for key, value in mean_values.items():
                print(f"Mean {key}: {value}.")

    return logger, mean_values

