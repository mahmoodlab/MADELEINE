# general
import sys
sys.path.append('../')
sys.path.append('../../')
import os
from collections import OrderedDict
from tqdm import tqdm
import wandb # type: ignore
import pdb

# numpy
import numpy as np
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import random

# torch
import torch # type: ignore
import torch.backends.cudnn # type: ignore
import torch.cuda # type: ignore

# internal
from core.utils.file_utils import save_pkl

# global magic numbers
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HE_POSITION = 0 # HE slide is always the first one 

# move to utils
def val_loop(config, ssl_model, val_dataloader):
    """
    Perform validation loop for the SSL model.

    Args:
        config (object): Configuration object containing model settings.
        ssl_model (object): SSL model to be evaluated.
        val_dataloader (object): Dataloader for validation dataset.

    Returns:
        tuple: A tuple containing the results dictionary and the rank measure.
            - results_dict (dict): Dictionary containing the embeddings and slide IDs.
            - rank (float): Rank measure calculated from the embeddings.
    """

    # set model to eval 
    ssl_model.eval()
    ssl_model.to(DEVICE)
    ssl_model, torch_precision = set_model_precision(ssl_model, config.precision)
    
    all_embeds = []
    all_slide_ids = []
    
    # do everything without grads 
    with torch.no_grad():
        for data in tqdm(val_dataloader):
        
            # unpack data and process
            data['feats'] = data['feats'].to(DEVICE).to(torch_precision)
            
            # forward
            wsi_embed = ssl_model(data=data, device=DEVICE, train=False)
            wsi_embed = wsi_embed['HE']
            
            all_embeds.extend(wsi_embed.squeeze(dim=1).to(torch.float32).detach().cpu().numpy())
            all_slide_ids.append(data['slide_ids'][0])
            
    all_embeds = np.array(all_embeds)
    all_embeds_tensor = torch.Tensor(all_embeds)
    rank = smooth_rank_measure(all_embeds_tensor)  
    results_dict = {"embeds": all_embeds, 'slide_ids': all_slide_ids}
    
    return results_dict, rank

def extract_slide_level_embeddings(args, val_dataloaders, ssl_model):
    """
    Extracts slide-level embeddings for each dataset in val_dataloaders using the provided ssl_model.

    Args:
        args (object): The arguments object containing various configuration options.
        val_dataloaders (dict): A dictionary containing the validation dataloaders for each dataset.
        ssl_model (object): The SSL model used for extracting embeddings.

    Returns:
        None
    """
    for dataset_name in val_dataloaders:
        print(f"\n* Extracting slide-level embeddings of {dataset_name}")
        curr_loader = val_dataloaders[dataset_name]
        curr_results_dict, curr_val_rank = val_loop(args, ssl_model, curr_loader)
        print("Rank for {} = {}".format(dataset_name, curr_val_rank))
        print("\033[92mDone \033[0m")
        
        if args.log_ml:
            wandb.run.summary["{}_rank".format(dataset_name)] = curr_val_rank
            
        save_pkl(os.path.join(args.RESULS_SAVE_PATH, f"{dataset_name}.pkl"), curr_results_dict)

def load_checkpoint(args, ssl_model):
    """
    Loads a checkpoint file and updates the state of the SSL model.

    Args:
        args (Namespace): The command-line arguments.
        ssl_model (nn.Module): The SSL model to update.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        RuntimeError: If the checkpoint file is corrupted or incompatible with the model.

    """
    state_dict = torch.load(os.path.join(args.RESULS_SAVE_PATH, "model.pt"))
    try:
        ssl_model.load_state_dict(state_dict)
    except:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        ssl_model.load_state_dict(new_state_dict)
        print('Model loaded by removing module in state dict...')

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
    rs = RandomState(MT19937(SeedSequence(SEED)))  # If any of the libraries or code rely on NumPy seed the global NumPy RNG.
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # If you are using multi-GPU. In case of one GPU, you can use # torch.cuda.manual_seed(SEED).

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
