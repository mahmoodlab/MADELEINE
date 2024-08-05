"""
# Usage
python extract_slide_embeddings_from_checkpoint.py --overwrite --local_dir ../results_brca/HUB_dfc80197ddc463b89ee1cd2a5d89f421
"""

# general
import sys; sys.path.append("../")

from core.datasets.dataset import SlideDataset, collate
import argparse
import os

# internal 
from core.models.factory import create_model_from_pretrained
from core.utils.utils import  run_inference
from torch.utils.data import DataLoader # type: ignore
from core.utils.file_utils import save_pkl

# define downstream dataset and loader
def get_downstream_loader():
    """
    Returns a DataLoader object for downstream processing.
    Returns:
        DataLoader: A DataLoader object that loads data for downstream processing.
    """
    
    dataset = SlideDataset(
        dataset_name = "BCNB",
        csv_path= "../dataset_csv/BCNB/BCNB.csv",
        features_path="../data/downstream/BCNB/feats_h5",
        modalities=["HE"],
        train=False
    )

    loader = DataLoader(
            dataset, 
            batch_size=1, 
            shuffle=False, 
            collate_fn=collate,
            num_workers=4,
        )
            
    return loader

if __name__ == "__main__":

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--local_dir", type=str, default=None)

    args = parser.parse_args()
    overwrite = args.overwrite
    local_dir = args.local_dir

    # init model
    model, precision = create_model_from_pretrained(local_dir, overwrite=overwrite)

    # get downstream loader
    loader = get_downstream_loader()

    # extract slide embeddings
    results_dict, rank = run_inference(model, loader, torch_precision=precision)

    # save
    save_pkl(os.path.join(args.local_dir, "BCNB.pkl"), results_dict)





    