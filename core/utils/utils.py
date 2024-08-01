#----> general imports
import numpy as np
import os
import argparse
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import random

#----> torch imports 
import torch
import torch
import torch.backends.cudnn
import torch.cuda

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_model_precision(model, precision):
    """
    Sets the precision of the model to the specified precision.

    Args:
        model (torch.nn.Module): The model to set the precision for.
        precision (str): The desired precision. Can be one of 'float64', 'float32', or 'bfloat16'.

    Returns:
        tuple: A tuple containing the model with the updated precision and the corresponding torch precision.
    """
    if precision == 'float64':
        torch_precision = torch.float64
    elif precision == 'float32':
        torch_precision = torch.float32
    elif precision == 'bfloat16':
        torch_precision = torch.bfloat16
    model = model.to(torch_precision)
    
    return model, torch_precision


def set_deterministic_mode(SEED, disable_cudnn=False):
    """
    Sets the random seed for various libraries to ensure deterministic behavior.

    Args:
        SEED (int): The seed value to use for random number generation.
        disable_cudnn (bool, optional): Whether to disable cuDNN. Defaults to False.

    Notes:
        - Sets the random seed for torch, random, numpy, and torch.cuda.
        - If `disable_cudnn` is False, also sets cuDNN to use deterministic algorithms.
        - If `disable_cudnn` is True, disables cuDNN.

    """
    torch.manual_seed(SEED)  # Seed the RNG for all devices (both CPU and CUDA).
    random.seed(SEED)  # Set python seed for custom operators.
    rs = RandomState(
        MT19937(SeedSequence(SEED)))  # If any of the libraries or code rely on NumPy seed the global NumPy RNG.
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(
        SEED)  # If you are using multi-GPU. In case of one GPU, you can use # torch.cuda.manual_seed(SEED).

    if not disable_cudnn:
        torch.backends.cudnn.benchmark = False  # Causes cuDNN to deterministically select an algorithm,
        # possibly at the cost of reduced performance
        # (the algorithm itself may be nondeterministic).
        torch.backends.cudnn.deterministic = True  # Causes cuDNN to use a deterministic convolution algorithm,
        # but may slow down performance.
        # It will not guarantee that your training process is deterministic
        # if you are using other libraries that may use nondeterministic algorithms
    else:
        torch.backends.cudnn.enabled = False  # Controls whether cuDNN is enabled or not.
        # If you want to enable cuDNN, set it to True.
    

def smooth_rank_measure(embedding_matrix, eps=1e-7):
    """
    Compute the smooth rank measure of a matrix of embeddings.
    
    Args:
        embedding_matrix (torch.Tensor): Matrix of embeddings (n x m). n: number of patch embeddings, m: embedding dimension
        alpha (float): Smoothing parameter to avoid division by zero.

    Returns:
        float: Smooth rank measure.
    """
    
    # Perform SVD on the embedding matrix
    _, S, _ = torch.svd(embedding_matrix)
    
    # Compute the smooth rank measure
    p = S / torch.norm(S, p=1) + eps
    p = p[:embedding_matrix.shape[1]]
    smooth_rank = torch.exp(-torch.sum(p * torch.log(p)))
    smooth_rank = round(smooth_rank.item(), 2)
    
    return smooth_rank
