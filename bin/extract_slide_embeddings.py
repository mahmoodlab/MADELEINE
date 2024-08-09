"""
# Usage
python extract_slide_embeddings.py --local_dir ../results/BCNB/ 
"""

# general
import sys; sys.path.append("../")

from core.datasets.dataset import SimpleDataset, simple_collate
import argparse
import os

# internal 
from core.models.factory import create_model_from_pretrained
from core.utils.utils import  run_inference
from torch.utils.data import DataLoader 
from core.utils.file_utils import save_pkl


# define downstream dataset and loader
def get_downstream_loader(path):
    """
    Returns a DataLoader object for downstream dataset.
    Returns:
        DataLoader: A DataLoader object that loads data for downstream processing.
    """
    dataset = SimpleDataset(features_path=os.path.join(path, 'patch_embeddings'))
    loader = DataLoader(dataset, num_workers=4, collate_fn=simple_collate)     
    return loader


if __name__ == "__main__":

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", type=str, default=None)
    parser.add_argument("--model_dir", type=str, default='./../models')

    args = parser.parse_args()
    local_dir = args.local_dir

    # init Madeleine model
    model, precision = create_model_from_pretrained(os.path.join(args.model_dir, 'MADELEINE'))

    # get downstream loader
    dataloader = get_downstream_loader(path=local_dir)

    # extract slide embeddings
    results_dict, rank = run_inference(model, dataloader, torch_precision=precision)
    save_pkl(os.path.join(local_dir, "madeleine_slide_embeddings.pkl"), results_dict)
