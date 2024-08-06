"""
# Usage
python extract_slide_embeddings_from_checkpoint.py --overwrite --local_dir ../results_brca/MADELEINE
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

DATASETS = {
    "BCNB" : {
        "csv_path" : "../dataset_csv/BCNB/BCNB.csv",
        "features_path" : "../data/downstream/BCNB/feats_h5"
    }
}


# define downstream dataset and loader
def get_downstream_loaders():
    """
    Returns a DataLoader object for downstream processing.
    Returns:
        DataLoader: A DataLoader object that loads data for downstream processing.
    """
    all_loaders = {}
    for d_name in DATASETS:
        dataset = SlideDataset(
            dataset_name=d_name,
            csv_path=DATASETS[d_name]["csv_path"],
            features_path=DATASETS[d_name]["features_path"],
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
        
        all_loaders[d_name] = loader
            
    return all_loaders

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
    all_loaders = get_downstream_loaders()

    # extract slide embeddings
    for dataset_name, loader in all_loaders.items():
        results_dict, rank = run_inference(model, loader, torch_precision=precision)
        save_pkl(os.path.join(args.local_dir, f"{dataset_name}.pkl"), results_dict)





    