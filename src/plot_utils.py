from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.colors as mcolors
import numpy as np
import torch
from torch import nn

from data import utils as data_utils


def subplot_dims(num_subplots, layout='balanced_wide', distance=2):
    """ 
    For a given number of desired subplots, returns grid shape as close to a 
    square as possible, without leaving too many grid entries blank.

    Parameters
    ----------
    num_subplots : int 
        Number of desired/actual subplots.
    layout : string
        One of {'wide' | 'tall'}. This function returns a grid shape as close
        to square as possible (see `distance`) will be returned, with dimension
        0 being the smaller one in the former case and the larger one in the
        latter. Default = 'wide'.
    distance : int >= 0 
        Max absolute value of difference between grid dimensions. E.g., set to
        0 to always get a square grid. Default = 2. 

    Returns
    -------
    dims : 2-tuple
        Row and column sizes of grid.
    """
    def get_divisors(n):
        """Returns ascending list of divisors for a given natural number n."""
        divisors = set()
        for k in range(1, int(n**0.5)+1):
            if n % k == 0: 
                divisors.add(k)
                divisors.add(n // k)
        return sorted(divisors)
    
    if not isinstance(distance, int):
        raise TypeError(
            f"Expected type int for `distance` but got {type(distance)}."
        )
    elif distance < 0:
        raise ValueError(
            f"`distance` should be a non-negative integer but got {distance}."
        )
    
    searching = True
    while searching:
        divisors = get_divisors(num_subplots)
        num_divisors = len(divisors)
        # Get two divisors closest to or including median divisor.
        central_divisors = (
            num_subplots // divisors[num_divisors//2],
            divisors[num_divisors//2]
        ) 
        # If divisors are close enough, end the search. Else, increase size 
        # of grid and continue.
        if abs(central_divisors[1] - central_divisors[0]) <= distance:
            searching = False
        else: 
            num_subplots += 1
    dims = central_divisors
    
    if layout == 'tall': 
        dims = dims[::-1]
    elif layout != 'wide':
        raise ValueError(
            f"Unrecognized value {layout} for `layout`, must be one of "
            "{'wide' | 'tall'}."
        )

    return dims

def plot_labeled_points(x, labels, labels_dict=None, ax=None, legend_loc='best'):
    """ 
    """
    DEFAULT_PARAMS = {
        'color' : (0, 0.4470, 0.7410),
        'marker' : 'o',
        'linewidths' : 1.5
    }

    # Validate labels.
    data_utils.validate_tensor(x, 2)
    labels = labels.squeeze()
    if len(labels) != x.shape[0]:
        raise ValueError(
            "Length of `labels` must match first dimension size of `x` but "
            f"got {len(labels)} and {x.shape[0]}."
        )
    labels_unique = torch.unique(labels)
    num_labels = len(labels_unique)
    
    # Validate data.
    num_dims = x.shape[1]
    if not(num_dims == 2 or num_dims == 3):
        raise ValueError(
            f"Dimension 1 of `x` must be of size 2 or 3, but got {x.shape[1]}."
        )
    
    # Validate kwargs.
    if labels_dict is None:
        # No label names provided.
        plot_legend = False
        # Generate different color for each label and use values in
        # DEFAULT_PARAMS for other plot parameters.
        cmap = plt.get_cmap('rainbow')
        colors = cmap(np.linspace(0, 1, num_labels))
        labels_dict = {
            int(label) : {'color' : color} 
            for label, color in zip(labels_unique, colors)
        }
    else:
        plot_legend = True
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d') if num_dims == 3 else fig.add_subplot()
    else:
        fig = ax.figure
    
    # Plot.
    for label in labels_unique:
        ind = torch.where(labels == label)[0]
        curr_dict = (
            {**DEFAULT_PARAMS, **labels_dict[int(label)]} 
            if int(label) in list(labels_dict.keys())
            else DEFAULT_PARAMS
        )
        if num_dims == 2:
            ax.scatter(x[ind, 0], x[ind, 1], **{**DEFAULT_PARAMS, **curr_dict})
        else:
            ax.scatter(
                x[ind, 0], x[ind, 1], x[ind, 2], **{**DEFAULT_PARAMS, **curr_dict}
            )
        
    if plot_legend: ax.legend(loc=legend_loc)

    return fig, ax

def plot_softmax(
        logits, 
        softmax_dim=-1, 
        width=0.5, 
        depth=0.8, 
        output_support=torch.tensor([0, 1]), 
        cmap=cm.viridis, 
        ax=None
    ):
    """ 
    width controls sequence dimension
    depth controls support dimension
    """
    data_utils.validate_tensor(logits, 2)
    softmax = nn.Softmax(dim=softmax_dim)
    probs = softmax(logits)
    probs = probs.detach().cpu()
    probs = probs.numpy()

    # Prepare for plotting.
    seq_len = probs.shape[0]
    x, y = torch.meshgrid(torch.arange(seq_len, dtype=output_support.dtype), output_support)
    x, y, probs = x.ravel(), y.ravel(), probs.ravel()
    bottom = torch.zeros_like(x)
    colors = cmap(probs)

    if ax is None:
        fig = plt.figure
        ax = fig.add_subplot(projection='3d')
    else:
        fig = ax.get_figure()

    ax.bar3d(x, y, bottom, width, depth, probs, color=colors)

    return fig, ax

def ensure_numpy(x):
    """ 
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    else:
        return np.asarray(x)

def plot_trajectory(
        x, 
        timepoints=None, 
        norm=None, 
        cmap=None, 
        uniform_axis_size=True,
        set_box_aspect=False,
        ax=None,
        traj_linewidth=1, 
        marker='o',
        marker_size=30,
        marker_linewidth=1.5,
        marker_edgecolor='none',
        alpha=1.0,
        depthshade=False
    ):
    """ 
    """
    # Validate input data.
    x = ensure_numpy(x)
    if len(x.shape) != 2: raise ValueError(
        f"`x` must be 2d but was passed as a {len(x.shape)}d array."
    )
    
    # Default to indices of timepoints.
    if timepoints is None: 
        timepoints = np.arange(x.shape[0])
    else:
        timepoints = ensure_numpy(timepoints)
        if len(timepoints.shape) != 1: 
            raise ValueError(
                "`timepoints` should be 1d but was passed as a " 
                f"{len(timepoints.shape)}d array."
            )
        if len(timepoints) != x.shape[0]:
            raise ValueError(
                "Length of `timepoints` must match first dimension size of x "
                f"({x.shape[0]}), but got {len(timepoints)}."
            )
        
    # Default to normalizing to min and max of timepoints.
    if norm is None: plt.Normalize(timepoints.min(), timepoints.max())

    # Default to dark to light blue colormap.
    if cmap is None:
        cmap = mcolors.LinearSegmentedColormap.from_list(
            'lightsky_to_dark_blue', 
            [(0, 'lightskyblue'), (1, 'darkblue')]
        )

    # Construct segments.
    x_expanded = x[:, np.newaxis, :]
    segments = np.concatenate((x_expanded[:-1], x_expanded[1:]), axis=1)
    lc = Line3DCollection(
        segments, cmap=cmap, norm=norm, linewidth=traj_linewidth, alpha=alpha
    )
    lc.set_array(timepoints[:-1]) # Each segment receives color based on its start time

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
    else:
        fig = ax.get_figure()

    ax.add_collection3d(lc)

    # Add markers at timepoints.
    ax.scatter(
        x[:, 0], x[:, 1], x[:, 2],
        c=timepoints, 
        cmap=cmap,
        norm=norm,
        marker=marker,
        s=marker_size,
        edgecolors=marker_edgecolor,
        linewidth=marker_linewidth,
        alpha=alpha,
        depthshade=depthshade
    )

    if set_box_aspect: ax.set_box_aspect([1, 1, 1])

    if uniform_axis_size:
        axis_limits = x.min(), x.max()
        ax.set_xlim(axis_limits)
        ax.set_ylim(axis_limits)
        ax.set_zlim(axis_limits)
        
    return fig, ax

