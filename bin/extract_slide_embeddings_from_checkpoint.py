"""
# Usage
python extract_slide_embeddings_from_checkpoint.py --pretrained ../results_brca/DEBUG_dfc80197ddc463b89ee1cd2a5d89f421/
"""


# general
import sys; sys.path.append("../")
import os
import json
from collections import OrderedDict
import pdb
import argparse

# torch
import torch # type: ignore

# internal
from core.utils.process_args import get_args
from core.models.Model import MADELEINE
from core.utils.utils import extract_slide_level_embeddings
from core.utils.setup_components import setup_DownstreamDatasets

import pdb

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_config(path_to_config):
    """
    Reads a JSON configuration file and returns its contents.

    Args:
        path_to_config (str): The path to the configuration file.

    Returns:
        dict: The contents of the configuration file.

    """
    with open(os.path.join(path_to_config, 'config.json')) as json_file:
        data = json.load(json_file)
        return data 
 
def restore_model(args, model, state_dict):
    """
    Restores the model's state from the given state dictionary.
    Args:
        model (nn.Module): The model to restore the state for.
        state_dict (dict): A dictionary containing the model's state.
    Returns:
        nn.Module: The model with the restored state.
    """
    print("* Loading model from {}...".format(args.pretrained), end="")
    sd = list(state_dict.keys())
    contains_module = any('module' in entry for entry in sd)
    
    if not contains_module:
        model.load_state_dict(state_dict, strict=True)
    else:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] 
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=True)
    
    print("\033[92m Done \033[0m")
        
    return model 

if __name__ == "__main__":
    
    args = get_args()
    assert args.pretrained is not None, "Must provide a path to a pretrained dir. Usage: --pretrained SOME_PATH/EXP_CODE/"
    config_from_model = read_config(args.pretrained)
    config_from_model = argparse.Namespace(**config_from_model)
    
    # set up MADELEINE model
    model = MADELEINE(
        config=config_from_model,
        stain_encoding=config_from_model.add_stain_encoding,
    ).to(DEVICE)
    
    # general info printing
    total_params = sum(p.numel() for p in model.parameters())
    print("* Total number of parameters = {}".format(total_params))
    
    # restore wsi embedder for downstream slide embedding extraction.  
    model = restore_model(args, model, torch.load(os.path.join(args.pretrained, 'model.pt')))
    
    # extract downstream slide embeddings using the freshly trained model and save
    val_dataloaders = setup_DownstreamDatasets(config_from_model)
    extract_slide_level_embeddings(config_from_model, val_dataloaders, model)
    
    print()
    print(100*"-")
    print("End of experiment, bye!")
    print(100*"-")
    print()
    