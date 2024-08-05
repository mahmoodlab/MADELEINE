
import sys

from core.utils.utils import set_model_precision 
sys.path.append("../../")
    
import os
import json
from argparse import Namespace
from typing import Dict, Optional, Union
import torch # type: ignore

from core.models.Model import create_model
import pdb

from huggingface_hub import snapshot_download


# local_dir = ''
# snap_download(repo_id="hf_hub:MahmoodLab/madeleine", local_dir=local_dir)
# create_model_from_pretrained(local_dir)

def create_model_from_pretrained(
        local_dir: str,
        overwrite: bool,
        ):
    
    # make sure local_dir exists 
    os.makedirs(local_dir, exist_ok=True)
    
    # if local_dir is empty, and overwrite is False, raise error
    assert len(os.listdir(local_dir)) != 0 or overwrite, "local_dir is empty and overwrite is False. Dont know where to find model"
    
    # if overwrite, then download model from HF
    if overwrite:
        print(f"* overwrite is True. Downloading model at {local_dir}")
        snapshot_download(repo_id="MahmoodLab/madeleine", local_dir=local_dir)
    else:
        print("* overwrite is False. Using model found in local_dir")
         
    # load config and weights 
    model_cfg = json.load(open(os.path.join(local_dir, "model_config.json")))
    model_cfg = Namespace(**model_cfg)
    checkpoint_path = os.path.join(local_dir, "model.pt")
    
    model = create_model(
        model_cfg,
        device="cuda",
        checkpoint_path=checkpoint_path,
    )
    
    # get precision of model
    precision = set_model_precision(model_cfg.precision)

    return model, precision

# def create_model_from_pretrained(
#         model_cfg: Union[str, Namespace, Dict],
#         device: Union[str, torch.device] = 'cpu',
#         checkpoint_path: Optional[str] = None,
#         hf_auth_token: Optional[str] = None,
# ):
#     model = create_model(
#         model_cfg,
#         device,
#         checkpoint_path=checkpoint_path,
#         hf_auth_token=hf_auth_token,
#     )

#     return model


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