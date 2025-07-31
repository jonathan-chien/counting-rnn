def subplot_dims(num_subplots, layout='wide', distance=2):
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


# def ensure_numpy(x):
#     """ 
#     """
#     if isinstance(x, torch.Tensor):
#         return x.detach().cpu().numpy()
#     else:
#         return np.asarray(x)