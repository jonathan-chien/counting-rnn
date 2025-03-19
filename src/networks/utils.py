import torch
from torch import nn
import warnings
from data import utils as data_utils


def move_to_device(data, device):
    """ 
    Utility for moving tensors, including tensors in arbitrarily nested lists/tuple/dicts.
    """
    if isinstance(data, torch.Tensor): 
        return data.to(device, non_blocking=True)
    elif isinstance(data, (list, tuple)): 
        return type(data)(move_to_device(elem, device) for elem in data)
    elif isinstance(data, dict):
        return {key : move_to_device(value, device) for key, value in data.items()}
    else:
        return data
    
def early_stopping(
    loss_by_epoch, 
    delta, 
    epsilon, 
    criterion=lambda x : torch.mean(torch.diff(x, n=1))
):
    """ 
    """
    if len(loss_by_epoch) < delta: 
        return False
    elif criterion(loss_by_epoch) > epsilon:
        return True
    else:
        return False
    
def shift_masks(masks):
    """ 
    """
    if (masks[:, 0]).any().item():
        invalid_rows = torch.where(masks[:, 0])[0]
        raise ValueError(
            "The first column of masks contains a True value in rows "
            f"{invalid_rows.tolist()}. Since masks should be of the response "
            "phase, there should always be sequence elements preceding the "
            "masked part, and the first element in each mask should thus be False."
        )
    else:
        return masks.roll(-1, dims=1)

def calculate_prop_correct(pred_labels, true_labels, masks):
    """ 
    """
    matches = torch.eq(pred_labels, true_labels) | ~masks
    seq_matches = matches.all(dim=1)

    num_correct = seq_matches.sum().item()
    num_seq = pred_labels.shape[0]

    correct_ind = torch.where(seq_matches)[0]
    incorrect_ind = torch.where(~seq_matches)[0]

    return num_correct/num_seq, num_correct, num_seq, correct_ind, incorrect_ind

def train(
    model, 
    dataloader, 
    optimizer, 
    loss_fn=nn.CrossEntropyLoss(),
    h_0=None,
    early_stopping_params=None, # Dict with delta, epsilon, and criterion_fn keys, corresponding to parameters for early_stopping function
    num_epochs=20, 
    device='cuda',
    pack=True,
    deterministic=True,
):
    """ 
    Parameters
    ----------
    model (AutoRNN) : 
    dataloader (DataLoader) : 
    optimizer ()
    loss_fn 
    h_0
    early_stopping_params
    num_epochs
    device
    pack
    deterministic (Bool): Controls whether tokens/labels based on logits are 
        generated deterministically vai argmax or probabilistically via softmax 
        and sampling.

    Returns
    -------
    out 
    """
    model.to(device)
    model.train()
    
    # Ensure tokens device matches that specified by user.
    if model.tokens.device != device:
        raise RuntimeError(
            f"model.tokens must be on user specified device {device} but is " 
            f"on {model.tokens.device}."
        )

    # Store relevant items from training. 
    out = {
        'logits' : [], 
        'hidden' : [], 
        'lengths' : [],
        'seq_ind' : [],
        'loss' : torch.full((num_epochs,), torch.nan),
        'accuracy' : torch.full((num_epochs,), torch.nan), 
        'early_stop_epoch' : None
    }

    for i_epoch in range(num_epochs):
        running_loss = 0
        batch_counter = 0

        epoch = {
            'logits' : [],
            'hidden' : [],
            'lengths' : [],
            'seq_ind' : [],
            'accuracy' : []
        }

        for batch, labels, lengths, masks, seq_ind in dataloader:
            batch, labels, masks = batch.to(device), labels.to(device), masks.to(device)

            optimizer.zero_grad()
        
            if pack:
                lengths = lengths 
            else:
                warnings.warn(
                    "`pack` has been passed in as False. This means that all "
                    "sequences should be of the same length."
                )
                lengths = None

            logits, rnn_output, _ = model(
                batch, h_0=h_0, lengths=lengths, output_type='many_to_many'
            )

            # Need to shift masks forward by one since prediction is of next token.
            shifted_masks = shift_masks(masks)

            # Compute accuracy (values already extracted as python float). Note 
            # that, in addition to using shifted masks, the columns of labels 
            # are shifted cyclically by one to align prediction of next tokens 
            # with ground truth tokens. This should be safe to do since the 
            # response phase is always preceded in a sequence by other tokens.
            _, pred_labels = model.to_token(
                logits, deterministic=deterministic
            )
            epoch['accuracy'].append(
                calculate_prop_correct(
                    pred_labels, 
                    labels.roll(-1, dims=1), 
                    shifted_masks
                )
            )

            # Compute loss. 
            loss = loss_fn(logits[shifted_masks], labels[masks])
            running_loss += loss
            batch_counter += 1 # TODO: should maybe count samples to get unbiased estimate of loss

            loss.backward()
            optimizer.step()

            epoch['logits'].append(logits.detach().cpu())
            epoch['hidden'].append(rnn_output.detach().cpu())
            epoch['lengths'].append(lengths.detach().cpu())
            epoch['seq_ind'].append(seq_ind.detach().cpu())

        # Compute and print epoch loss and accuracy.
        out['loss'][i_epoch] = running_loss / batch_counter
        print(f"Mean batch loss for epoch {i_epoch}: {out['loss'][i_epoch]}.")
        out['accuracy'][i_epoch] = torch.mean(
            torch.tensor([batch_acc[0] for batch_acc in epoch['accuracy']])
        )
        print(f"Mean batch accuracy for epoch {i_epoch}: {out['accuracy'][i_epoch]}.")

        # Move items to CPU to save GPU memory and store.
        for key, value in epoch.items(): 
            if isinstance(out[key], list): out[key].append(value) 

        # Optional early stopping.
        if early_stopping_params is not None and early_stopping(
            out['loss'][~torch.isnan(out['loss'])],
            **early_stopping_params,
        ):
            loss_not_nan = out['loss'][~torch.isnan(out['loss'])]
            print(
                f"Early stopping condition reached after epoch {i_epoch}. "
                f"Loss over last {early_stopping_params['delta']} epochs: "
                f"{loss_not_nan[-early_stopping_params['delta']:]}."
                "\nFinal loss changes prior to early stopping: "
                f"{torch.diff(loss_not_nan[-early_stopping_params['delta']:])}."
            )
            out['early_stop_epoch'] = i_epoch

            return out
        
    return out

def test(
    model, 
    dataloader, 
    switch_label,
    loss_fn, 
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
    
    out = {
        'mean_loss' : None,
        'loss' : [],
        'performance' : [],
        'generated' : [],
        'on_input' : [],
        'joined' : [],
        'labels' : [],
        'masks' : [],
        'seq_ind' : []
    }

    loss = 0
    num_batches = len(dataloader)
    batch_sizes = torch.full((num_batches,), torch.nan)

    with torch.no_grad():
        for i_batch, (batch_full_seq, labels, lengths, masks, seq_ind) in enumerate(dataloader):
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
            
            # if max_resp_len is None: max_resp_len = torch.max(resp_lengths)

            generated, on_input, joined = model.generate(
                batch_demo_phase, 
                input_lengths=demo_lengths,
                h_0=h_0,  
                resp_lengths=resp_lengths,
                deterministic=deterministic,
            )     

            # Now implemented inside generate method:
            # # Mask to recover valid portion of generated logit tensor. 
            # batch_size = batch_demo_phase.shape[0]
            # generated['resp_masks'] = torch.full(
            #     (batch_size, max_resp_len),
            #     fill_value=False,
            #     device=device
            # )
            # for i_seq in range(batch_size):
            #     generated['resp_masks'][i_seq, :resp_lengths[i_seq]] = True

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
                # true_resp_masks_aligned = torch.gather(
                #     masks, dim=1, index=resp_col_ind
                # )

                # Output should be detached.
                performance = [
                    criterion(
                        generated['labels'],  
                        true_resp_labels_aligned,
                        generated['resp_masks']
                    ) 
                    for criterion in criteria
                ]
            else:
                performance = None

            batch_sizes[i_batch] = batch_full_seq.shape[0]

            out['loss'].append(loss)
            out['performance'].append(performance)
            out['generated'].append(generated) 
            out['on_input'].append(on_input)
            out['joined'].append(joined)
            out['labels'].append(labels)
            out['masks'].append(masks)
            out['seq_ind'].append(seq_ind)
        
    out['loss'] = torch.tensor(out['loss'])
    out['mean_loss'] = torch.sum(
        (batch_sizes/len(dataloader.dataset)) * out['loss']
    )

    if move_results_to_cpu:
        for key, value in out.items(): out[key] = move_to_device(value, 'cpu')

    return out
 
# def join_demo_gen_resp(demo, demo_lengths, resp, resp_lengths):
#     """ 
#     Utility to concatenate itmes (e.g. logits/hidden states but most the latter) 
#     from demonstration phase input and generated output, accounting
#     for timesteps that may appear in both (e.g. at the switch index). 

#     demo and resp should be of shape (b, n1, h) and (b, n2, h), where b and h 
#     are the batch and hidden size, respectively, and n1 and n2 are the padded 
#     lengths of the demonstration phase input and generated response, repsectively.
#     """
#     # Validate inputs.
#     data_utils.validate_tensor(demo, 3)
#     data_utils.validate_tensor(resp, 3)
#     if not(demo.shape[0] == resp.shape[0] and demo.shape[2] == resp.shape[2]):
#         raise ValueError(
#             "The sizes of the first and third dimensions of `demo` and `resp` must match."
#         )
#     data_utils.validate_tensor(demo_lengths, 1)
#     data_utils.validate_tensor(resp_lengths, 1)
#     if len(demo_lengths) != len(resp_lengths):
#         raise ValueError(
#             "The lengths of `demo_lengths` and `resp_lengths` must match, but " 
#             f"got {len(demo_lengths)} and {len(resp_lengths)}, respectively."
#         )

#     # Preallocate.
#     batch_size, _, hidden_size = demo.shape
#     max_input_len = torch.max(demo_lengths)
#     max_resp_len = torch.max(resp_lengths)
#     joined_padded = torch.full(
#         (batch_size, max_input_len+max_resp_len, hidden_size), fill_value=0.0
#     )
#     joined_lengths = torch.full((batch_size,), 0, dtype=torch.int64)

#     # Benchmarking suggests using masks and list comprehension is slightly slower.
#     for i_seq in range(batch_size):
#         # Get new set of lengths that identifies joined sequences as valid 
#         # portions of padded output.
#         joined_lengths[i_seq] = demo_lengths[i_seq] + resp_lengths[i_seq]

#         # Join each sequence.
#         joined_padded[i_seq, :joined_lengths[i_seq], :] = torch.cat(
#             (
#                 demo[i_seq, :demo_lengths[i_seq], :], 
#                 resp[i_seq, :resp_lengths[i_seq], :]
#             ),
#             dim=0
#         )
        
#     return joined_padded, joined_lengths



