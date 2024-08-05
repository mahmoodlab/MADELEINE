
import sys 
sys.path.append("../../")
    
import os
import json
from argparse import Namespace
from typing import Dict, Optional, Union
import torch # type: ignore

from core.models.Model import create_model
import pdb


def create_model_from_pretrained(
        model_cfg: Union[str, Namespace, Dict],
        device: Union[str, torch.device] = 'cpu',
        checkpoint_path: Optional[str] = None,
        hf_auth_token: Optional[str] = None,
):
    model = create_model(
        model_cfg,
        device,
        checkpoint_path=checkpoint_path,
        hf_auth_token=hf_auth_token,
    )

    return model


# test create_model_from_pretrained
if __name__ == "__main__":
    
    
    
    path_to_model = "../../results_brca/dfc80197ddc463b89ee1cd2a5d89f421"
    ckpt_path = os.path.join(path_to_model, "model.pt")
    config_path = os.path.join(path_to_model, "config.json")
    
    # load config and change to namespace 
    
    with open(config_path) as f:
        config = json.load(f)
    
    # convert json to name space
    config = Namespace(**config)
    
    # load model 
    model = create_model_from_pretrained(
        model_cfg=config,
        device='cuda',
        checkpoint_path=ckpt_path,
    )
    
    pdb.set_trace()