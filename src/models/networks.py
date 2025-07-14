import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from general_utils import tensor as tensor_utils


class FFN(nn.Module):
    """
    Feedfoward network constructed from a list using nn.Sequential.
    """
    def __init__(self, layers):
        super().__init__()
        self.model = nn.Sequential(*layers)

    def forward(self, input):
        return self.model(input)


class FCN(nn.Module):
    """ 
    Feedforward network with an input layer and N-1 linear layers, where 
    each non-input layer may have some activation function and dropout.
    """

    def __init__(self, layer_sizes, nonlinearities=nn.ReLU, dropouts=0.0):
        """ 
        Parameters
        ----------
        layer_sizes : list or None
            Of length N. k_th element is int size of k_th layer including input 
            layer. If None, the network is set as a single nn.Identity layer.
        nonlinearities : list, nn.Module subclass
            For list input, of length N - 1, where k_th element corresponds to
            k_th non-input layer and is an activation function object of
            nn.Module subclass like nn.ReLU(). Can also pass a single nn.Module
            subclass like nn.ReLU, which will be broadcast to all
            layers and instantiated. Default=nn.ReLU.
        dropouts : list, float
            For list input, of length N-1, where k_th element is float-valued
            probability of dropping for k_th non-input layer. If a float,
            should be in [0.0, 1.0] and will be broadcast as probability of
            dropping for all non-input layers. Default=0.0.
        """
        super().__init__()

        # Validate layer sizes.
        if layer_sizes is None:
            self.model = nn.Identity()
            return
        elif (
            not isinstance(layer_sizes, list) 
            or not all(isinstance(size, int) for size in layer_sizes)
        ):
            raise ValueError(
                "`layer_sizes` must be a list of integers."
            )
        elif len(layer_sizes) < 2:
            raise ValueError(
                f"`layer_sizes` must have at least 2 elements, got {len(layer_sizes)}."
            )
        num_layers = len(layer_sizes)

        if not isinstance(nonlinearities, list):
            if isinstance(nonlinearities, type) and issubclass(nonlinearities, nn.Module):
                nonlinearities = [nonlinearities() for _ in range(num_layers-1)]
            else: raise TypeError(
                "`nonlinearities` must either be a list or an subclass of nn.Module "
                f"such as nn.ReLU, but got object of type {type(nonlinearities)}."
            )
        elif len(nonlinearities) != num_layers - 1:
            raise ValueError(
                f"The length of `nonlinearities` ({len(nonlinearities)}) must be " 
                f"one less than that of `layer_sizes` ({len(layer_sizes)})."
            )
        elif not all(isinstance(item, nn.Module) for item in nonlinearities):
            raise TypeError(
                "If `nonlinearities` is a list, its elements must be objects "
                "of the nn.Module class like nn.ReLU()."
            )
        
        # Validate dropouts.
        if isinstance(dropouts, float):
            if not 0.0 <= dropouts <= 1.0:
                raise ValueError(
                    "If passed as a float, `dropouts` must be in [0.0, 1.0] "
                    f"but got {dropouts}."
                )
            else:
                dropouts = [dropouts] * (num_layers - 1)
        elif not isinstance(dropouts, list):
            raise TypeError(
                "`dropouts` must either be a list or float in [0.0, 1.0] but "
                f"got type {type(dropouts)}."
            )
        elif len(dropouts) != num_layers - 1:
            raise ValueError(
                f"The length of 'dropouts' ({len(dropouts)}) must be one less" 
                f"than that of 'layer_sizes' ({len(layer_sizes)})."
            )
        elif not all(
            (isinstance(prob, float) and 0.0 <= prob <= 1.0) or prob is None
            for prob in dropouts
        ):
            raise ValueError(
                "If `dropouts` is a list, each element must be a float in [0.0, 1.0] or None."
            )
        dropouts = [
            nn.Dropout(p=prob, inplace=False) if prob is not None else nn.Identity() 
            for prob in dropouts
        ]
        
        # Construct network.
        layers = []
        for in_dim, out_dim, nonlinearity, dropout in zip(
            layer_sizes[:-1], layer_sizes[1:], nonlinearities, dropouts
        ):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nonlinearity)
            layers.append(dropout)
        self.model = nn.Sequential(*layers)

        self.layer_sizes = layer_sizes
        self.nonlinearities = nonlinearities
        self.dropouts = dropouts

    def forward(self, input):
        """Pass input through model."""
        return self.model(input)  
         

class AutoRNN(nn.Module):
    """ 
    Recurrent neural network with an arbitrary number of FC input and readout 
    layers preceding and following, respectively, a torch.nn.RNNBase subclass.
    Capable of autoregressive generation upon being provided an input sequence.
    """

    def __init__(
            self, 
            input_network,
            rnn,
            readout_network,
            tokens 
        ):
        """ 
        Parameters
        ----------
        input_network : instance of nn.Module subclass
            Neural network instance for processing inputs before passing result
            to the recurrent network.
        rnn : instance of nn.Module subclass
            Recurrent core between input and readout layers.
        readout_network : instance of nn.Module subclass
            Neural network for reading out recurrent hidden activity. Output of 
            this network are logits.
        tokens : torch.Tensor
            Of shape (V, F), where F is the input size of the first layer of
            the entire network (first input layer if input_network_args is
            not None, else first layer of the recurrent core network), V is the
            number of tokens in the generating vocabulary of the network, and
            the k_th row corresponds to the k_th token.
        """
        super().__init__()

        # Set up input transformation network, RNN, and readout network.
        self.input_network = input_network
        self.rnn = rnn
        self.readout_network = readout_network
        self.register_buffer('tokens', tokens)

        # Can be used to generate tokens probabilistically.
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input, lengths=None, h_0=None, output_type='many_to_many'):
        """ 
        Passes a sequence or batch thereof through all layers of the full
        network. Batched input sequences should be uniform length or padded and
        can be packed by this method.

        Dimensions
        ----------
        B : Batch size.
        L : (Padded) sequence length.
        F : Size of first layer in full network (either input to input FCN or 
            to recurrent network).
        
        Parameters
        ----------
        input : torch.Tensor
            2D or 3D, of shape (L, F) for a single sequence or (B, L, F) for a 
            batch thereof.
        lengths : torch.Tensor 
            Of shape (B,). Contains lengths of each sequence used to pack 
            sequences with torch.nn.utils.rnnpack_pad_sequences. If None, this
            will be skipped. Default is None.
        h_0 : torch.Tensor or None
            2D or 3D. Initial hidden state. If None, initialized to all zeros.
        output_type : string 
            {'many_to_many' | 'many_to_one'} Specifies whether to return 
            readout of hidden state on each timestep or only final timestep.

        Returns
        -------
        logits : torch.Tensor
        h_n : torch.Tensor 
        rnn_output : torch.Tensor
        """
        input = self.input_network(input)

        if h_0 is None: 
            h_0 = self.initialize_h_0(input.shape, input.device)

        # Optionally pack padded sequence, if lengths provided.
        if lengths is not None:
            input = pack_padded_sequence(
                input, lengths, batch_first=True, enforce_sorted=False
            )

        # Forward pass.
        rnn_output, h_n = self.rnn(input, h_0)

        # If sequences were packed, unpack. rnn_output.shape = (batch_size, 
        # seq_len, hidden_size).
        if lengths is not None: 
            rnn_output, _ = pad_packed_sequence(rnn_output, batch_first=True)

        if output_type == 'many_to_many':
            logits = self.readout_network(rnn_output) 
        elif output_type == 'many_to_one' :
            logits = self.readout_network(rnn_output[:, -1, :]) # (batch_size, 2)
        
        return logits, rnn_output, h_n

    def to_token(self, logits, deterministic=True):
        """ 
        Takes in logits and generates tokens either deterministically via 
        argmax, or probabilistically by applying softmax and sampling.

        Dimensions
        ----------
        B : Batch size.
        T : Number of timepoints.
        V : Generating vocabulary size.

        Parameters
        ----------
        logits : torch.Tensor
            2D or 3D. Either logits of shape (B, V) or (B, T, V), where the 
            former is for single timepoints per batch.

        Returns
        -------
        torch.Tensor 
            Tensor of tokens. Same dim as `logits` with first two dimensions
            matching in size, but last dimension size equal to embedding
            dimension for tokens.
        ind : torch.Tensor
            Of dim one less than logits. Either of shape (B, T) or (T,). Each
            element is the index of the token selected for that timepoint.
        """
        if deterministic:
            ind = torch.argmax(logits, dim=-1)
        else:
            pmfs = self.softmax(logits)
            categorical = torch.distributions.Categorical(probs=pmfs)
            ind = categorical.sample()
            
        return self.tokens[ind], ind

    def generate(
            self, 
            input_seqs, 
            input_lengths=None, 
            resp_lengths=None, 
            h_0=None, 
            deterministic=True,
        ):
        """ 
        Runs forward method on input sequence and then autoregressively 
        generates output.

        Dimensions
        ----------
        B : Batch_size.
        I : (Padded) input sequence length.
        R : (Padded) generated response length. Computed as max of 
            `resp_lengths` (see below) or set to I if `resp_lengths` is None.
        H : Size of output hidden layer.
        F : Size of first layer in entire network.
        V : Size of network's generating vocabulary. E.g., if network outputs
            Count and EOS tokens, V = 2.

        Parameters
        ----------
        input_seqs : torch.Tensor
            Of shape (I, F) for a single sequence or (B, I, F) for a batch
            thereof. Note that 2D inputs are promoted to 3D by adding a
            singleton dimension 0.
        input_lengths : torch.Tensor
            Of shape (B,). Contains lengths of each input sequence, used to
            pack inputs for forward pass and to join hidden states and logits
            from processing input with the same during generation. If
            input_lengths is None, packing will be skipped, and will use I for
            joining. Default is None.
        resp_lengths : torch.Tensor
            Of shape (B,). Contains lengths of valid generated responses for
            each sequence. Max of these will be used to preallocate response
            tensor and to compute lengths of joined output (see `joined`
            below). If `resp_lengths` is None, will use I. Default is None.
        h_0 : torch.Tensor or None
            Of shape matching that documented for hx in nn.RNN, h_0 in nn.GRU
            etc. If None, initialized to zeros. Default is None.
        deterministic : bool 
            Specify whether to select output token during generation
            deterministically via argmax over logits or probabilitstically by
            applying softmax and sampling.

        Returns
        -------
        generated : dict
            'hidden' : torch.Tensor
                Of shape (B, R, H). Hidden states of the recurrent network
                during generation. First slice along dim 1 corresponds to the
                hidden state on the final timestep of the input, as this state
                gives rise to the first set of predicted logits/labels.
            'logits' : torch.Tensor
                Of shape (B, R, V). Logits from the readout layer during
                generation. First slice along dim 1 corresponds to the logits
                resulting from the final timestep of the input.
            'tokens' : torch.Tensor
                Of shape (B, R, F). Tokens produced by applying either argmax 
                on generated logits (deterministic), or softmax plus sampling 
                (probabilistic). First slice along dim 1 corresponds to tokens
                output as a result of receiving the last timestep of the input.
            'labels' : torch.Tensor
                Of shape (R,). Generated labels. Elements are in {0, ..., V-1}
                and correspond to row indices of the `tokens` attribute.
            'resp_masks' : torch.Tensor or None
                Of shape (B, R). Boolean-valued. The k_th row is a mask whose
                values are true for the first P_k elements and False otherwise, 
                with P_k <= R being the k_th element of `resp_lengths`.
            'resp_lengths' : torch.Tensor or None
                Of shape (B,). A copy of `resp_lengths` included for downstream
                access in a unified place.
        on_input : dict
            'hidden' : torch.Tensor
                Of shape (B, I, H). Contains final hidden layer state for each
                timepoint in the input sequence (including final timepoint).
            'logits' : torch.Tensor
                Of shape (B, I, V). Contains readout of network for each 
                timepoint in the input sequence (including final timepoint).
            'input_lengths' : 
                Of shape (B,). A copy of `input_lengths` included for 
                downstream access in a unified place.
        joined (dict) : 
            'hidden' : torch.Tensor
                Of shape (B, I + R - 1, H). Contains concatenation of hidden 
                states in 'on_input' and 'generated'. Note that both of these 
                contain a hidden state for the final timestep of the input 
                sequence, and this redundancy is removed here.
            'logits' : torch.Tensor
                Of shape (B, I + R - 1, V). Contains concatenation of readout
                of network for 'on_input' and 'generated'. Note that both of
                these contain logits for the final timestep of the input
                sequence, and this redundancy is removed here.
        """
        if input_seqs.dim() == 2:
            input_seqs = input_seqs.clone().unsqueeze(0)
        elif input_seqs.dim() != 3:
            raise ValueError(
                f"`input_seqs` must be of dim 2 or 3 but got dim {input_seqs.dim()}."
            )
        
        batch_size, max_input_length, num_dims = input_seqs.shape

        # Ensure tokens are on same device as inputs (move if not).
        input_device = input_seqs.device
        if self.tokens.device != input_device:
            tokens_original_device = self.tokens.device
            self.tokens = self.tokens.to(input_device)
            tokens_moved = True
        else:
            tokens_moved = False
        
        # Forward pass on input sequence, e.g. demo phase with final token being switch. 
        input_seqs_logits, input_seqs_rnn_output, h_n = self.forward(
            input_seqs, 
            h_0=h_0, 
            lengths=input_lengths, 
            output_type='many_to_many'
        ) 

        # Store output of forward pass on inputs to be returned.
        on_input = {
            'hidden' : input_seqs_rnn_output,
            'logits' : input_seqs_logits,
            'input_lengths' : input_lengths
        }

        # input_seqs_logits is padded and of shape (batch_size, max_seq_len, 2). 
        # Need to recover correct final logits for each sequence based on its length.
        if input_lengths is not None:
            input_seqs_final_logits = input_seqs_logits[
                torch.arange(batch_size), input_lengths-1, :
            ]
        else:
            input_seqs_final_logits = input_seqs_logits[:, -1, :]

        # Preallocate.
        if resp_lengths is None: 
            max_resp_len = max_input_length
        else:
            max_resp_len = torch.max(resp_lengths)
        generated = {
            'hidden' : torch.full(
                (batch_size, max_resp_len, self.rnn.hidden_size), 
                torch.nan
            ).to(input_device),
            'logits' : torch.full(
                (batch_size, max_resp_len, self.readout_network.layer_sizes[-1]),
                torch.nan
            ).to(input_device),
            'tokens' : torch.full(
                (batch_size, max_resp_len, num_dims), 
                torch.nan
            ).to(input_device),
            'labels' : torch.full(
                (batch_size, max_resp_len), 
                0, 
                dtype=torch.int64
            ).to(input_device)
        }

        # Store hidden state and logits produced by input of switch token. This
        # is the first timestep of the "generated" portion, since the output
        # will be the first generated value.
        generated['hidden'][:, 0, :] = h_n.squeeze()
        generated['logits'][:, 0, :] = input_seqs_final_logits
        
        # Store predicted token from last timepoint of forward pass above as 
        # first generated token.
        generated['tokens'][:, 0, :], generated['labels'][:, 0] \
            = self.to_token(input_seqs_final_logits, deterministic=deterministic) # previous_output.shape = (batch_size, 1, num_dims)

        # Generation loop for remaining timesteps.
        for i_step in range(1, max_resp_len):
            # Prepare for current iteration. Input token has already been predicted.
            current_input = generated['tokens'][:, i_step-1, :].unsqueeze(1)
            h_0 = generated['hidden'][:, i_step-1, :].unsqueeze(0)
            
            # Forward pass on current timestep.
            current_logits, _, h_n = self.forward(
                current_input, 
                lengths=None,
                h_0=h_0, 
                output_type='many_to_one'
            ) 

            # Store network state/readout from current step.
            generated['hidden'][:, i_step, :] = h_n.squeeze()
            generated['logits'][:, i_step, :] = current_logits

            # Predict next timestep.
            generated['tokens'][:, i_step, :], generated['labels'][:, i_step] \
                = self.to_token(current_logits, deterministic=deterministic)
            
        # If tokens were moved, move them back.
        if tokens_moved: self.tokens = self.tokens.to(tokens_original_device)

        # Create masks to recover valid portion of generated logit tensor.
        if resp_lengths is not None: 
            generated['resp_masks'] = torch.full(
                (batch_size, max_resp_len),
                fill_value=False,
                device=input_device
            )
            for i_seq in range(batch_size):
                generated['resp_masks'][i_seq, :resp_lengths[i_seq]] = True
        else:
            generated['resp_masks'] = None

        # Store response lengths.
        generated['resp_lengths'] = resp_lengths

        # Join hidden states and logits from input and generated output. Minus
        # one on second argument bc last input timestep will also be the first
        # generated timestep.
        joined = {
            key : self.join_input_gen_resp(
                on_input[key],
                (   
                    input_lengths if input_lengths is not None 
                    else torch.tensor(max_input_length).repeat(batch_size,)
                )-1,
                generated[key],
                (
                    resp_lengths if resp_lengths is not None 
                    else torch.tensor(max_resp_len).repeat(batch_size,) 
                )
            )
            for key in ['hidden', 'logits']
        }
          
        return generated, on_input, joined
    
    def get_h_0_shape(self, batch_size=False):
        """ 
        Get shape of initial hidden state h_0 based on network parameters and
        whether or not input is batched.
        """
        if not batch_size:
            h_0_shape = (
                (self.rnn.bidirectional+1)*self.rnn.num_layers, 
                self.rnn.hidden_size
            )
        else:
            h_0_shape = (
                (self.rnn.bidirectional+1)*self.rnn.num_layers, 
                batch_size, 
                self.rnn.hidden_size
            )

        return h_0_shape
    
    def initialize_h_0(self, input_shape, device):
        """ 
        Initialize hidden state to all zeros.

        Parameters
        ----------
        input_shape : torch.Size 
            Shape of input tensor to network.
        device : 
            Device on which to initialize hidden state.

        Returns
        -------
        h_0 : torch.Tensor
            2D or 3D. Matches shape documented for hx in nn.RNN, h_0 in nn.GRU etc.
        """
        if len(input_shape) == 2:
            h_0 = torch.zeros(
                self.get_h_0_shape(batch_size=False),
                device=device
            )
        elif len(input_shape) == 3:
            batch_size = input_shape[0]
            h_0 = torch.zeros(
                self.get_h_0_shape(batch_size=batch_size),
                device=device
            )
        else:
            raise ValueError(
                "input to network should be a 2D or 3D tensor but was " 
                f"{len(input_shape)}d."
            )
        return h_0
       
    @staticmethod
    def join_input_gen_resp(input, input_lengths, resp, resp_lengths):
        """ 
        Utility to concatenate items (e.g. logits/hidden states) 
        produced during processing input and generating output, accounting
        for timesteps that may appear in both (e.g. final timestep of input). 

        Dimensions
        ----------
        B : Batch size.
        I : (Padded) input sequence length.
        R : (Padded) generated output length.
        D : Dimension of input. E.g., for hidden states, D = final hidden layer 
            size; for logits, D = size of generating vocabulary.

        Parameters
        ----------
        input : torch.Tensor
            Of shape (B, I, D). Items from processing the input sequences.
        input_lengths : torch.Tensor
            Of shape (B,). Lengths of the valid portion of each sequence in batch.
        resp : torch.Tensor
            Of shape (B, R, D). Items from generating output for each sequence.
        resp_lengths : torch.Tensor
            Of shape (B,). Lengths of the valid portion of each generated 
            output in batch.

        Returns
        -------
        joined : torch.Tensor
            Of shape (B, I + R, D). The k_th slice along the first dimension
            is the concatenation along dimension 0 of the k_th slices of 
            `input` and `resp`.
        joined_lengths : torch.Tensor
            Of shape (B,). k_th element is equal to L_k + R_k for L_k and R_k
            as the k_th elements of `input_lenths` and `resp_lengths`, 
            respectively.
        """
        # Validate inputs.
        tensor_utils.validate_tensor(input, 3)
        tensor_utils.validate_tensor(resp, 3)
        if not(input.shape[0] == resp.shape[0] and input.shape[2] == resp.shape[2]):
            raise ValueError(
                "The sizes of the first and third dimensions of `input` and `resp` must match."
            )
        tensor_utils.validate_tensor(input_lengths, 1)
        tensor_utils.validate_tensor(resp_lengths, 1)
        if len(input_lengths) != len(resp_lengths):
            raise ValueError(
                "The lengths of `input_lengths` and `resp_lengths` must match, but " 
                f"got {len(input_lengths)} and {len(resp_lengths)}, respectively."
            )

        # Preallocate.
        batch_size, _, hidden_size = input.shape
        max_input_len = torch.max(input_lengths)
        max_resp_len = torch.max(resp_lengths)
        joined = torch.full(
            (batch_size, max_input_len+max_resp_len, hidden_size), fill_value=0.0
        )
        joined_lengths = torch.full((batch_size,), 0, dtype=torch.int64)

        for i_seq in range(batch_size):
            # Get new set of lengths that identifies joined sequences as valid 
            # portions of padded output.
            joined_lengths[i_seq] = input_lengths[i_seq] + resp_lengths[i_seq]

            # Join each sequence.
            joined[i_seq, :joined_lengths[i_seq], :] = torch.cat(
                (
                    input[i_seq, :input_lengths[i_seq], :], 
                    resp[i_seq, :resp_lengths[i_seq], :]
                ),
                dim=0
            )
            
        return joined, joined_lengths

