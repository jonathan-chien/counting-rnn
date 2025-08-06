from itertools import chain

import torch

from . import utils
from general_utils import recursion as recursion_utils
from general_utils import tensor as tensor_utils

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
    # NB: The below code is from the original implementation (note that
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
    
    # Input lengths must be on CPU for sequence packing.
    demo_lengths = switch_ind + 1 # Up to and including switch token
    resp_lengths = lengths - switch_ind - 1 # From first count token to EOS, inclusive
    demo_lengths = demo_lengths.to('cpu')
    # resp_lengths = resp_lengths.to('cpu')

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
        outputs = recursion_utils.recursive(
            outputs, 
            branch_conditionals=(
                recursion_utils.dict_branch,
                recursion_utils.list_branch,
                recursion_utils.tuple_branch
            ),
            leaf_fns=(
                tensor_utils.move_to_device('cpu'), 
                tensor_utils.detach_tensor
            )
        )

    return outputs

def get_true_resp_labels_aligned(switch_ind, resp_lengths, labels):
    """ 
    """
    # Validate.
    tensor_utils.validate_tensor(switch_ind, dim=1)
    tensor_utils.validate_tensor(resp_lengths, dim=1)
    tensor_utils.validate_tensor(labels, dim=2)
    if len(switch_ind) != len(resp_lengths):
        raise ValueError(
            "Lengths of `switch_ind` and `resp_lengths` must match, but got "
            f"{len(switch_ind)} and {len(resp_lengths)}, respectively."
        )
    
    if len(switch_ind) != labels.shape[0]:
        raise ValueError(
            "Lengths of switch_ind and resp_lengths must equal first dimension " \
            f"size of labels, but got {len(switch_ind)} and {len(resp_lengths)} "
            f"for the former, respectively, and {labels.shape[0]} for the latter."
        )

    # Get indices of start and stop of true response portion of labels. Note
    # that many neg tokens on seq k can result in a large switch index S but
    # short response length (so k's labels are not max overall length); if max
    # resp length R comes from seq j, it may be that S + R >= L, for L as the
    # second dim size of labels, L. Thus, need to clip stop indices to L.
    
    # label_length = labels.shape[1]
    # resp_start_col_ind = switch_ind + 1
    # resp_stop_col_ind = torch.minimum(
    #     switch_ind + 1 + max_resp_length, torch.tensor(label_length)
    # )

    # Get mask and use it to zero out indices greater than labels second dim size.
    max_resp_length = torch.max(resp_lengths)
    arange = torch.arange(max_resp_length, device=switch_ind.device)[None, :] 
    mask = arange < resp_lengths[:, None]
    resp_col_ind = (switch_ind[:, None] + 1 + arange) * mask
    
    # Get response portion of labels. Excess indices will now point to first 
    # element of each row, which we will zero out.
    true_resp_labels_aligned = torch.gather(labels, dim=1, index=resp_col_ind)
    return true_resp_labels_aligned * mask

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
        loss_term.name : loss_term.compute_loss(
            output=gen_logits_reshaped[gen_resp_masks_reshaped, :],
            targets=labels[masks].to(torch.int64),
            model=model
        )
        for loss_term in loss_terms
    }

    # # Need to get response portions of ground truth but left-aligned
    # # at first count token after switch token.
    # resp_start_col_ind = switch_ind + 1
    # resp_stop_col_ind = switch_ind + 1 + torch.max(resp_lengths)

    # # The sequence with the longest generated response might not be the
    # # one with the highest switch index. E.g. seq j has 20 count tokens and 
    # # switch index of 30, and seq k has only 18 count tokens but switch index of
    # # 33 because of many negative class tokens. If we take 20 tokens from switch
    # # index 33, we could index past the second dim size of labels. Thus, need to
    # # create augmented labels for indexing.
    # overshoot = resp_stop_col_ind.max()-labels.shape[1]
    # if overshoot < 0:
    #     raise RuntimeError(
    #         "Unexpected runtime condition: max value of resp_stop_col_ind "
    #         f"({resp_stop_col_ind.max()}) is less than second dim size of labels "
    #         f"({labels.shape[1]}). This should not occur."
    #     )
    # labels_augmented = torch.cat(
    #     (labels, torch.zeros(labels.shape[0], overshoot, dtype=torch.int64, device=labels.device)),
    #     dim=1
    # )

    # # max_col = labels.shape[1]
    # # print(f"max col: {max_col}")
    # # # Before stacking
    # # for i, (start, stop) in enumerate(zip(resp_start_col_ind, resp_stop_col_ind)):
    # #     if start.item() < 0 or stop.item() > max_col:
    # #         print(f"Invalid index range in sample {i}: [{start.item()}, {stop.item()}) vs max_col={max_col}")
    # # switch_ind_cpu = switch_ind.cpu()
    # # labels_cpu = labels.cpu()
    # # resp_start_col_ind_cpu = resp_start_col_ind.cpu()
    # # resp_stop_col_ind_cpu = resp_stop_col_ind.cpu()
    # # resp_lengths_cpu = resp_lengths.cpu()
    # # generated_resp_masks_cpu = generated['resp_masks'].cpu()


    # resp_col_ind = torch.stack([
    #     torch.arange(start, stop, device=labels.device) 
    #     for start, stop in zip(resp_start_col_ind, resp_stop_col_ind)    
    # ])

    # # resp_col_ind_cpu = resp_col_ind.cpu()
    # try: # For debugging
    #     true_resp_labels_aligned = torch.gather(
    #         labels_augmented, dim=1, index=resp_col_ind
    #     )
    # except:
    #     a = 1

    # Need to get response portions of ground truth but left-aligned at first 
    # count token after switch token.
    true_resp_labels_aligned = get_true_resp_labels_aligned(
        switch_ind=switch_ind, resp_lengths=resp_lengths, labels=labels
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
        losses = recursion_utils.recursive(
            losses,
            branch_conditionals=(
                recursion_utils.dict_branch,
                recursion_utils.list_branch,
                recursion_utils.tuple_branch
            ),
            leaf_fns=(
                tensor_utils.move_to_device(
                    'cpu' if move_results_to_cpu else generated['logits'].device
                ),
                tensor_utils.detach_tensor
            ) 
        )
        performance = recursion_utils.recursive(
            performance, 
            branch_conditionals=(
                recursion_utils.dict_branch,
                recursion_utils.list_branch,
                recursion_utils.tuple_branch
            ),
            leaf_fns=(
                tensor_utils.move_to_device(
                    'cpu' if move_results_to_cpu else generated['labels'].device
                ),
                tensor_utils.detach_tensor
            )
        )

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
    move_results_to_cpu=True
):
    """ 
    Logger instance is required, since the point of this function is to evaluate and record the performance/loss etc.
    """
    model.to(device)
    model.eval()
    switch_label_on_device = switch_label.to(device)

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
                switch_label_on_device, 
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

            # Log batch size.
            batch_log = {'batch_size' : batch_full_seq.shape[0]}

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

            # Optionally log outputs.
            if log_outputs:
                batch_log.update({
                    'generated' : outputs['generated'],
                    'on_input' : outputs['on_input'],
                    'joined' : outputs['joined'],
                    'seq_ind' : seq_ind
                })
            
            if move_results_to_cpu:
                batch_log = recursion_utils.recursive(
                    batch_log, 
                    branch_conditionals=(
                        recursion_utils.dict_branch,
                        recursion_utils.list_branch,
                        recursion_utils.tuple_branch
                    ),
                    leaf_fns=(
                        tensor_utils.move_to_device('cpu'),
                    )
                )

            logger.log_batch(epoch_idx=0, batch_idx=i_batch, **batch_log, verbose=False)

        # Compute mean across batches for specified losses and performance metrics.
        batch_sizes = logger.get_all_entries(key='batch_size', level='batch')
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

        # Log mean batch values as a single epoch.
        logger.log_epoch(epoch_idx=0, **mean_values)

    return logger

