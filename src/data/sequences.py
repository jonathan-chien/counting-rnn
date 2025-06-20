from dataclasses import dataclass
from typing import Dict
import warnings

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from . import utils
from general_utils import tensor as tensor_utils


class Hypercube:
    """ 
    For an N-dimensional hypercube, defines a dichotomy by taking a specified
    cylinder set as the positive class and places PMFs over vertices on both sides.
    """

    def __init__(
            self, 
            num_dims, 
            coords, 
            inclusion_set=None, 
            encoding=torch.tensor([0, 1], dtype=torch.int8),
            vertices_pmfs=None
        ):
        """ 
        Dimensions
        ----------
        N : Dimensionality of hypercube.
        P : Number of coordinates with restricted values.

        Parameters
        ----------
        num_dims : int
            N.
        coords : torch.Tensor
            Of shape (P,). Indices of the dimensions which have restricted
            values. Setting to arange(0, N) allows specifying the positive
            class explicitly by enumerating all vertices in `inclusion_set`.
        inclusion_set : torch.Tensor
            Of shape (M, N), where the k_th row corresponds to the k_th element
            of the "inclusion set" B, with |B| = M, to which the P coordinates,
            indexed in `coords`, of a vertex x in the positive class must
            belong. E.g. if `coords` = {0, 1, 3}, then the positive class
            consists of all x for which (x_0, x_1, x_3) in B. If `coords` = {0,
            1, ..., N}, leaving no free coordinates, the positive class can be
            explicitly specified by enumerating all its vertices.
        encoding : torch.Tensor
            Of shape (2,). Values used to define hypercube. Default is 
            torch.tensor([0, 1]).
        vertices_pmfs : 2-tuple
            Argument to self.set_vertices_pmfs.
        """
        if inclusion_set is None: 
            inclusion_set = torch.tile(torch.tensor([1]), (1, len(coords)))
        self._validate(num_dims, inclusion_set, coords, encoding)
        self.num_dims = num_dims
        self.coords = coords.to(torch.int64)
        self.num_coords = len(self.coords)
        self.inclusion_set = inclusion_set
        self.encoding = encoding

        self.set_vertices_distrs(vertices_pmfs)

    def _validate(self, num_dims, inclusion_set, coords, encoding):
        """ 
        Ensure valid inputs. Parameters match that of constructor method.

        Parameters
        ----------
        See constructor method.
        """
        tensor_utils.validate_tensor(inclusion_set, 2)
        tensor_utils.validate_tensor(coords, 1)
        if inclusion_set.shape[1] != len(coords):
            raise ValueError(
                f"Number of columns of `inclusion_set` ({inclusion_set.shape[1]}) "
                f"must match length of `coords` ({len(coords)})."
            )
        if len(coords) > num_dims:
            raise ValueError(
                f"Length of `coords` should be <= `num_dims` ({num_dims}) " 
                f"but got {len(coords)}."
            )
        if coords.dtype != torch.int64:
            raise ValueError(
                f"Expected torch.int64, but got {coords.dtype}."
            )
        if not torch.isin(inclusion_set, encoding).all():
            raise ValueError(
                f"Elements of `inclusion set` ({torch.unique(inclusion_set)}) "
                f"don't match that of `encoding` ({encoding})."
            )
        
    def check_if_included(self, vertices):
        """
        For a stack of vertices, check whether each vertex belongs to the 
        positive class.
        
        Parameters
        ----------
        vertices : torch.Tensor
            Of shape (V, N) where the k_th row is the k_th vertex of a stack of
            vertices, V is the number of vertcies, and N the dimensionality of
            the hypercube.

        Returns
        -------
        ind : torch.Tensor
            Of shape (V,), Boolean valued, where V is the number of vertices.
            The k_th element is True if the k_th vertex is in the positive
            class and False otherwise.
        """
        if not torch.isin(torch.unique(vertices), self.encoding).all():
            warnings.warn(
                f"Some elements of `vertices` ({torch.unique(vertices)}) are "
                f"not in `self.encoding` ({self.encoding})."
            )
        vertices_expanded = vertices.unsqueeze(-2) # shape (seq_len, 1, num_dims)
        inclusion_set_expanded = self.inclusion_set.unsqueeze(0) # shape (1, n_rows, n_cyl_coord), where n_rows is number of rows of the inclusion set matrix.
        matches = (
            vertices_expanded[:, :, self.coords] == inclusion_set_expanded
        )
        ind = matches.all(-1).any(-1)
        return ind
    
    def set_vertices_distrs(self, pmfs=None):
        """ 
        Set user supplied PMFs over respective sides of dichotomy, or 
        initialize uniform distribution.

        Parameters
        ----------
        pmfs : 2-tuple
            First and second elements are pmfs over positive and negative class 
            elements, respectively, consisting of a 1D tensor containing
            probability masses.
        """
        truth_table = utils.get_lexicographic_ordering(self.num_dims, self.encoding)
        pos_ind = self.check_if_included(truth_table)

        if pmfs is not None:
            utils.validate_pmf(pmfs[0], torch.sum(pos_ind))
            utils.validate_pmf(pmfs[1], torch.sum(~pos_ind))
        else:
            pmfs = (
                utils.uniform_pmf(torch.sum(pos_ind)), utils.uniform_pmf(torch.sum(~pos_ind))
            )
        
        self.vertices = {
            'pos' : {
                'support' : truth_table[pos_ind],
                'pmf' : pmfs[0]
            },
            'neg' : {
                'support' : truth_table[~pos_ind],
                'pmf' : pmfs[1]
            }
        }


@dataclass
class SeqLengths:
    """ 
    Helper class for validating and storing in a format compatible with the
    Sequence class N distributions over the respective lengths of N sequences,
    for a natural number N.

    lengths : dict
        dict where each key is the name of a kind of sequence (e.g. 'pos', 
        'neg'), and each value is a dict with the following keys:
            'support' : 1D tensor of non-negative ints.
            'pmf' : 1D tensor of probability masses, same legnth as 'support'.
    """
    lengths: Dict[str, Dict[str, torch.Tensor]]

    def __post_init__(self):
        for name, entry in self.lengths.items():
            try:
                support, pmf = entry['support'], entry['pmf']
                tensor_utils.validate_tensor(support, 1)
                utils.validate_pmf(pmf, len(support))
            except Exception as e:
                raise ValueError(f"Validation failed for '{name}'.") from e

# class SeqLengths:
#     """ 
#     Helper class for validating and storing in a format compatible with the
#     Sequence class N distributions over the respective lengths of N sequences,
#     for a natural number N.
#     """
#     def __init__(self, lengths):
#         """ 
#         Parameters
#         ----------
#         lengths : dict
#             dict where each key is the name of a kind of sequence (e.g. 'pos', 
#             'neg'), and each value is a dict with the following keys:
#                 'support' : 1D tensor of non-negative ints.
#                 'pmf' : 1D tensor of probability masses, same legnth as 'support'.
#                 'dtype' : dict with the following keys:
#                     'support' : torch.dtype of support tensor
#                     'pmf' : torch.dtype of support tensor

#         Returns
#         -------
#         Sets attribute `lengths`, which is a dictionary with keys corresponding
#         to `names`; each value is a dict with keys 'support' and 'pmf', 
#         containing the supplied support and pmf.
#         """
#         self.lengths = dict()
#         for name, entry in lengths.items():
#             support, pmf = entry['support'], entry['pmf']
#             tensor_utils.validate_tensor(support)
#             utils.validate_pmf(pmf, len(support))
#             self.lengths[name] = {'support' : support, 'pmf' : pmf}
                
# class SeqLengths:
#     """ 
#     Helper class for validating and storing in a format compatible with the
#     Sequence class N distributions over the respective lengths of N sequences,
#     for a natural number N.
#     """
#     def __init__(self, *args):
#         """ 
#         Parameters
#         ----------
#         Each argument should be a triple (support, pmf, name). For the k_th
#         argument, `support` is a list/tuple of non-negative integers
#         corresponding to the support of the k_th distribution over sequence
#         lengths for a Sequences object, `pmf` is a 1d tensor consisting of the
#         corresponding pmf, and `name` is a string corresponding to that
#         distribution/Sequences object.

#         Returns
#         -------
#         Sets attribute `lengths`, which is a dictionary with keys corresponding
#         to `names`; each value is a dict with keys 'support' and 'pmf', 
#         containing the supplied support and pmf.
#         """
#         self.lengths = dict()
#         for arg in args:
#             tensor_utils.validate_tensor(arg[0], 1)
#             tensor_utils.validate_pmf(arg[1], len(arg[0]))
#             self.lengths[arg[2]] = {
#                 'support' : arg[0],
#                 'pmf' : arg[1]
#             }

class Sequences(Dataset):
    """ 
    Creates sequences that consist of elements drawn from two sets. May expand 
    an arbitrary number of sets in the future.
    """

    def __init__(
            self, 
            num_seq, 
            len_distr, 
            elem_distr,
            num_vars, 
            seq_order='permute',
            transform=None,
            dtype=torch.int8
        ):
        """ 
        Set attributes and intialize torch.distributions.Categorical sampling
        objects.

        Parameters
        ----------
        num_seq : int 
            Number of sequences that will be sampled.
        len_distr : dict 
            Dict with keys 'pos' and 'neg'. Corresponding values are dicts
            stored as values of the 'pos' and 'neg' keys in the `lengths`
            attribute of a SeqLengths object.
        elem_distr : dict 
            Dict with keys 'pos' and 'neg'. Corresponding values are dicts
            containing keys 'support' and 'pmf' whose values contain the
            elements of that class (1d or 2d tensor with elements corresponding
            to rows) and the corresponding pmf.
        num_vars : int 
            Number of variables for each sequence.
        seq_order : string 
            {'pos_first' | 'neg_first' | 'permute'}. If 'pos_first', the
            positive class elements will be drawn first with the negative class
            elements drawn next and concatenated to the end. If 'neg_first',
            the negative class elements come first. If 'permute' the positive
            class and negative class elements are drawn separately,
            concatenated, and randomly permuted.
        transform : function, callable class 
            Transform to be applied within the __getitem__ method.
        dtype : torch.dtype
            torch datatype of the sequence values. Default is torch.int8.
        """
        self.num_seq = num_seq
        self.len_distr = len_distr
        self.elem_distr = elem_distr
        self.num_vars = num_vars # TODO: Either eliminate num_vars (enforce that elem must have 2d tensor and use 2nd dim, or at least check against elem)
        self.seq_order = seq_order
        self.transform = transform
        self.dtype = dtype

        self.data = None
        self.labels = None
        self._sampler_initialized_flag = False
        self._strings_sampled_flag = False
        self._pad_eos_overridden_flag = False

        self._initialize_samplers()
        self.sample()
        self.add_special_tokens()
        
    def _initialize_samplers(self):
        """ 
        """
        for class_ in ['pos', 'neg']:
            # Samplers for sequence lenghths. Change syntax to update?
            self.len_distr[class_]['sampler'] = torch.distributions.Categorical(
                probs=self.len_distr[class_]['pmf']
            )
            
            # Samplers for vertices.
            self.elem_distr[class_]['sampler'] = torch.distributions.Categorical(
                probs=self.elem_distr[class_]['pmf']
            )

        self._sampler_initialized_flag = True
    
    def sample(self):
        """ 
        """
        def get_row_label(A, B):
            """ 
            A and B are tensors of shape (m, n) and (p, n), respectively. 
            Returned is a 1d tensor of length m whose k_th element is the row
            index of the row of B matching the k_th row of A.
            """
            tensor_utils.validate_tensor(A, 2)
            tensor_utils.validate_tensor(B, 2)
            if A.shape[1] != B.shape[1]: 
                raise ValueError(
                    "Second dimension sizes of `A` and `B` must match but got " 
                    f"{A.shape[1]} and {B.shape[1]}."
                )
            
            matches = (A[:, None, :] == B).all(dim=-1) # Shape (m, p)
            return torch.where(matches)[1]

        if not self._sampler_initialized_flag:
            raise RuntimeError(
                "Samplers for sequence length and vertices have not been instantiated."
            )
        
        # Sample sequence lengths.
        self.seq_lens = {
            class_ : self.len_distr[class_]['support'][ 
                self.len_distr[class_]['sampler'].sample((self.num_seq,))
            ]
            for class_ in ['pos', 'neg']
        } 

        # Determine order of pos vs neg class within sequence. If permutation
        # desired, will be performed after sampling.
        if self.seq_order == 'pos_first' or self.seq_order == 'permute':
            class_1 = 'pos'
            class_2 = 'neg' 
        elif self.seq_order == 'neg_first' :
            class_1 = 'neg'
            class_2 = 'pos'
        else:
            raise ValueError(
                f"Unrecognized value '{self.seq_order}' for attribute `seq_order`."
            )
        
        # Sample elements from each class, for each sequence.
        self.seq_elems = {
            class_ : [
                self.elem_distr[class_]['support'][
                    self.elem_distr[class_]['sampler'].sample(
                        (self.seq_lens[class_][i_seq],)
                    )
                ] 
                if self.seq_lens[class_][i_seq] > 0 else torch.tensor([]) # Can't sample 0 items directly
                for i_seq in range(self.num_seq)
            ]
            for class_ in [class_1, class_2]
        }

        # Get within class (positive vs negative) index for each element.
        self.within_class_labels = {
            class_ : [
                get_row_label(seq, self.elem_distr[class_]['support'])
                if self.seq_lens[class_][i_seq] > 0 else torch.tensor([])
                for i_seq, seq in enumerate(self.seq_elems[class_])
            ]
            for class_ in [class_1, class_2]
        }

        # Add four to the positive class labels because labels 0-3 will be used
        # for special tokens. Add 1 to negative class and multiply by negative
        # 1 so that negative class indices begin at negative 1 and decrement.
        self.within_class_labels['pos'] = [
            labels + 4 for labels in self.within_class_labels['pos']
        ]
        self.within_class_labels['neg'] = [
            labels + 1 for labels in self.within_class_labels['neg']
        ]
        self.within_class_labels['neg'] = [
            labels * -1 for labels in self.within_class_labels['neg']
        ]

        # Combine positive and negative class for each sequence.
        seq_pairs = zip(self.seq_elems[class_1], self.seq_elems[class_2])
        self.data = [torch.cat(pair, dim=0) for pair in seq_pairs]
        label_pairs = zip(self.within_class_labels[class_1], self.within_class_labels[class_2])
        self.labels = [torch.cat(pair, dim=0) for pair in label_pairs]

        # Optionally permute sequence elements (will intermix positive and negative elements).
        if self.seq_order == 'permute':
            perm_ind = [
                torch.randperm(self.seq_lens['pos'][i_seq] + self.seq_lens['neg'][i_seq])
                for i_seq in range(self.num_seq)
            ]
            for i_seq in range(self.num_seq):
                if perm_ind[i_seq].numel() > 0: # Will be empty if 0 elements drawn from both positive and negative class
                    self.data[i_seq] = self.data[i_seq][perm_ind[i_seq], :]
                    self.labels[i_seq] = self.labels[i_seq][perm_ind[i_seq]]

        self._strings_sampled_flag = True

    def add_special_tokens(self, pad_eos=None):
        """ 
        Add beginning of sequence, switch, count, and end of sequence tokens.
        """
        if not self._strings_sampled_flag: 
            raise RuntimeError(
                "Strings must be sampled and labeled before calling this method. "
            )

        # Define special (non-demonstration) tokens.
        truth_table = utils.get_lexicographic_ordering(
            2, torch.tensor([0, 1], dtype=self.dtype)
        )
        special_tokens = {
            token : {
                'token' : torch.cat(
                    (
                        torch.zeros((self.num_vars,), dtype=self.dtype), 
                        torch.tensor([1], dtype=self.dtype), # This bit is 1 for special tokens, zero otherwise 
                        truth_table[i_token,:]
                    )
                ),
                'label' : torch.tensor([i_token], dtype=torch.int64)
            }
            for i_token, token in enumerate(['count', 'eos', 'bos', 'switch'])
        }
        
        self.final_seq_lens = torch.full((self.num_seq,), 0, dtype=torch.int64)
        self.resp_phase_mask = []
        for i_seq, seq in enumerate(self.data):
            # Add three extra dimensions of zeros to demonstration tokens, if 
            # sequence is not empty (0 elements drawn from both pos and neg class).
            if self.data[i_seq].numel() > 0:
                seq = torch.cat(
                    (seq, torch.zeros((self.data[i_seq].shape[0], 3), dtype=self.dtype)), 
                    dim=1
                )
            else:
                # Above case creates new tensor so staying consistent here.
                seq = self.data[i_seq].clone() 

            # Add beginning of sequence token and label.
            seq = torch.cat(
                (special_tokens['bos']['token'].unsqueeze(0), seq), dim=0 # seq_len dimension
            )
            labels_ = torch.cat(
                (special_tokens['bos']['label'], self.labels[i_seq])
            )

            # Add switch token to signal switch from demonstration to response phase.
            switch_idx = len(labels_) # Used for creating mask for response phase
            seq = torch.cat(
                (seq, special_tokens['switch']['token'].unsqueeze(0)), dim=0
            )
            labels_ = torch.cat((labels_, special_tokens['switch']['label']))

            # Add count tokens.
            pos_count = self.seq_lens['pos'][i_seq] 
            seq = torch.cat(
                (seq, special_tokens['count']['token'].repeat(pos_count, 1)), 
                dim=0
            )
            labels_ = torch.cat(
                (labels_, special_tokens['count']['label'].repeat(pos_count))
            )
            
            # Add EOS token.
            seq = torch.cat(
                (seq, special_tokens['eos']['token'].unsqueeze(0)), dim=0
            )
            labels_ = torch.cat((labels_, special_tokens['eos']['label']))

            # Create mask for response phase.
            resp_phase_mask = torch.zeros((len(labels_),), dtype=torch.int8)
            resp_phase_mask[switch_idx+1:] = 1
            resp_phase_mask = resp_phase_mask > 0
                
            self.data[i_seq] = seq
            self.labels[i_seq] = labels_
            self.final_seq_lens[i_seq] = len(labels_)
            self.resp_phase_mask.append(resp_phase_mask)

        self.special_tokens = special_tokens

    def __len__(self):
        """
        Returns:
        --------
        int 
            Number of sequences (first dimension size of sequences).
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Parameters:
        -----------
        idx : int
            Index of sequence and its corresponding labels to retrieve.
        
        Returns:
        --------
        """
        sequence = self.data[idx]
        if self.transform is not None: sequence = self.transform(sequence)
        return (
            sequence, 
            self.labels[idx], 
            self.final_seq_lens[idx], 
            self.resp_phase_mask[idx],
            idx
        )
    
    @staticmethod
    def pad_collate_fn(batch, padding_value=0):
        """ 
        Returns batch (tensor) of right 0 padded sequences, along with 
        corresponding lengths and labels.
        """
        sequences, labels, lengths, masks, seq_ind = zip(*batch)

        padded_sequences = pad_sequence(
            sequences, batch_first=True, padding_value=padding_value, padding_side='right'
        )
        padded_labels = pad_sequence(
            labels, batch_first=True, padding_value=padding_value, padding_side='right'
        )
        padded_masks = pad_sequence(
            masks, batch_first=True, padding_value=False, padding_side='right'
        )
    
        return (
            padded_sequences, 
            padded_labels, 
            torch.tensor(lengths), 
            padded_masks, 
            torch.tensor(seq_ind)
        )


class Embedder:
    """ 
    Linear transformation of data with addition of noise from arbitrary 
    additive noise distribution.
    """
    def __init__(
            self, 
            ambient_dim, 
            mean_center=False, 
            offset_1=None, 
            offset_2=None,
            method='random_rotation',
            noise_distr=None 
        ):
        self.ambient_dim = ambient_dim
        self.mean_center = mean_center
        self.offset_1 = offset_1
        self.offset_2 = offset_2
        self.method = method 
        self.noise_distr = noise_distr
        self.noise_sampler = (
            tensor_utils.make_sampler(self.noise_distr) 
            if self.noise_distr is not None else None
        )
            
        # If 'random_rotation', create a random standard normal matrix and use 
        # Q as a rotation matrix, for A = QR. The random matrix is stored as an 
        # attribute.
        if self.method == 'random_rotation':
            self.random_mat = torch.normal(0, 1, (ambient_dim, ambient_dim))
            self.lin_transform, _ = torch.linalg.qr(self.random_mat)
        # Specify an arbitrary linear transformation as an n x n matrix.
        elif isinstance(self.method, torch.Tensor):
            if len(self.method.shape) != 2: 
                raise ValueError(
                    "`method` can be passed as a 2d tensor, but a "
                    f"{len(self.method.shape)}d tensor was received."
                )
            if len(torch.unique(self.method.shape)) != 1:
                raise ValueError(
                    f"`method` can be passed as a 2d tensor, but it must be square."
                )
            self.lin_transform = self.method
        elif self.method == 'identity':
            self.lin_transform = None
        else:
            raise ValueError(f"Unrecognized value {self.method} for `method`.")
        
    def _noise(self, x): 
        """ 
        """
        return (
            x + self.noise_sampler(x.shape) 
            if self.noise_sampler is not None else x
        )

    def __call__(self, data):
        """ 
        Parameters:
        -----------
        data (2d tensor): Of shape (num_datapoints, num_vars+3).
        """
        data = data.clone()
        num_datapoints, num_dims = data.shape
    
        # E.g. this could be to center hypercube at 0 before rotation.
        if self.offset_1 is not None: data += self.offset_1

        if num_dims > self.ambient_dim:
            raise ValueError(
                f"Dimensionality of data is {num_dims}, but cannot be greater "
                f"than value of attribute `ambient_dim`({self.ambient_dim})."
            )
        else:
            extra_dims = self.ambient_dim - num_dims

        # Pad with zeros to introduce any extra dimensions. 
        data_padded = torch.cat(
            (data, torch.zeros((num_datapoints, extra_dims))), dim=1
        )

        transformed = (
            data_padded @ self.lin_transform 
            if self.lin_transform is not None else data_padded
        )
            
        transformed = self._noise(transformed)

        if self.offset_2 is not None: transformed += self.offset_2
        
        return transformed
    