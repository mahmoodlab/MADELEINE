import os
import json
from argparse import Namespace

from madeleine.models.Model import create_model
import pdb

from huggingface_hub import snapshot_download

from madeleine.utils.utils import set_model_precision 

# local_dir = ''
# snap_download(repo_id="hf_hub:MahmoodLab/madeleine", local_dir=local_dir)
# create_model_from_pretrained(local_dir)

def create_model_from_pretrained(local_dir: str):
    
    # make sure local_dir exists 
    os.makedirs(local_dir, exist_ok=True)
        
    # Download model from HF
    print(f"* Downloading model at {local_dir}")
    snapshot_download(repo_id="MahmoodLab/madeleine", local_dir=local_dir)

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