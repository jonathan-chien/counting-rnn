import torch


def pca(data, num_comps=None, corr=False):
    """ 
    PCA via SVD. 

    Dimensions
    ----------
    M : Number of observations.
    N : Number of variables.
    P : Number of Components to be retained.

    Parameters
    ----------
    data : torch.Tensor
        Of shape (M, N). Matrix of data.
    num_comps : int or None
        P. Number of components to keep. If None, all components are retained. 
        (Default: None)
    corr : bool
        Operation is equivalent to diagonalization of covariance matrix, if
        False, or of correlation matrix, if True. (Default: False)

    Returns
    -------
    pcs : torch.Tensor
        Of shape (M, P). Matrix whose columns are principal components, or
        scores.
    lambdas : torch.Tensor 
        Of shape (P,). Eigenvalues of retained components.
    loadings : torch.Tensor
        Of shape (M, P). Columns are retained right eigenvectors of data
        cov/corr matrix, scaled by the square root of the corresponding
        eigenvalue.
    v : torch.Tensor
        Of shape (M, P). Columns are retained right eigenvectors of the data
        cov/corr matrix.
    eigenspectrum torch.Tensor
        Of shape (N,). All eigenvalues of data cov/corr matrix.
    """
    if corr:
        data_ = data / torch.norm(data, p=2, dim=0, keepdim=True)
    else:
        data_ = data - torch.mean(data, dim=0)

    u, s, vh = torch.linalg.svd(data_, full_matrices=False)
    pcs = u * s
    lambdas = s**2 / (data.shape[0] - 1)
    eigenspectrum = lambdas.clone()
    v = vh.T
    loadings = v * torch.sqrt(lambdas)

    if num_comps is None:
        return {
            'eigenspectrum': eigenspectrum,
            'lambdas': lambdas, 
            'v': v,
            'pcs': pcs, 
            'loadings': loadings, 
        }
    elif isinstance(num_comps, int):
        if not (num_comps > 0 and num_comps <= data.shape[1]):
            raise ValueError(
                f"`num_comps` must be an integer between 1 and {data.shape[1]} "
                f"but got {num_comps}."
            )
        return {
            'eigenspectrum': eigenspectrum,
            'lambdas': lambdas[:num_comps], 
            'v': v[:, :num_comps],
            'pcs': pcs[:, :num_comps], 
            'loadings': loadings[:, :num_comps], 
        }
    else:
        raise TypeError(
            "Expected the string 'all' or an int for `num_comps`, but got type " 
            f"{type(num_comps)}."
        )