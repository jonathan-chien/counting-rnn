import torch

def set_seed(torch_seed, cuda_seed, use_deterministic_algos, cudnn_deterministic, cudnn_benchmark):
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed(cuda_seed)

    torch.use_deterministic_algorithms(use_deterministic_algos)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = cudnn_benchmark


# Could also set seed for random module and numpy if needed.