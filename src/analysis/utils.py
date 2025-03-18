import torch

def pca(data, num_comps=None, corr=False):
    """ 
    PCA via SVD. 

    Parameters:
        data (2d tensor) : m x n matrix of data.
        num_comps (int, None) : Number of components to keep. If None, all 
            components are retained. (Default: None)
        corr (bool) : Equivalent to diagonalization of covariance matrix, 
            if False, or of correlation matrix, if True. (Default: False)

    Returns:
        pcs (2d tensor): m x p matrix whose columns are principal components, 
            or scores, where p is the number of components retained.
        eigenvalues (1d tensor) : Eigenvalues of retained components.
        loadings (2d tensor) : Eigenvectors of retained components, scaled by 
            the square root of the corresponding eigenvalue.
        eigenspectrum (1d tensor) : All eigenvalues of cov/corr matrix.
    """
    if corr:
        data_ = data / torch.norm(data, p=2, dim=0, keepdim=True)
    else:
        data_ = data - torch.mean(data, dim=0)

    u, s, vh = torch.linalg.svd(data_, full_matrices=False)
    pcs = u * s
    eigenvalues = s**2 / (data.shape[0] - 1)
    eigenspectrum = eigenvalues.clone()
    v = vh.T
    loadings = v * torch.sqrt(eigenvalues)

    if num_comps is None:
        return pcs, eigenvalues, loadings, v, eigenspectrum
    elif isinstance(num_comps, int):
        if not (num_comps > 0 and num_comps <= data.shape[1]):
            raise ValueError(
                f"`num_comps` must be an integer between 1 and {data.shape[1]} "
                f"but got {num_comps}."
            )
        return (
            pcs[:, :num_comps], 
            eigenvalues[:num_comps], 
            loadings[:, :num_comps], 
            v[:, :num_comps],
            eigenspectrum
        )
    else:
        raise TypeError(
            "Expected the string 'all' or an int for `num_comps`, but got type " 
            f"{type(num_comps)}."
        )