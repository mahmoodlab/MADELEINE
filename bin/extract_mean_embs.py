# general
import sys; sys.path.append('../')
import os
import numpy as np
import h5py
from tqdm import tqdm
import pickle
import pdb


# datasets 
DATASETS = ["BCNB"]

if __name__ == "__main__":
    
    # path to patch feats 
    root_path_to_patches = "../data/downstream"
    
    for dataset in DATASETS:
        
        print("Extracting mean embeddings for dataset:", dataset)
        
        path_to_patches = os.path.join(root_path_to_patches, dataset, "feats_h5")
        all_files = os.listdir(path_to_patches)
        
        # iterate over all files, open each file, calculate the mean embedding, and store in numpy 
        all_embeds = []
        all_slide_ids = []
        for f in tqdm(all_files):
    
            path_to_file = os.path.join(path_to_patches, f)
            with h5py.File(path_to_file, 'r') as file:
                patch_feats = file['features'][:].squeeze(0)
            
            mean_embed = np.mean(patch_feats, axis=0)
            all_embeds.append(mean_embed)
            all_slide_ids.append(f.split(".h5")[0])
        
        # make a dictionary from embeds and slide_ids
        embed_dict = {"embeds": np.array(all_embeds), "slide_ids": all_slide_ids}
        
        # save dictionary as pickle file 
        save_path = os.path.join("../results_brca/mean", f"{dataset}.pkl")
        
        # pickle dump dictionary
        with open(save_path, 'wb') as handle:
            pickle.dump(embed_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        print("\033[92m Done \033[0m")
        print()
        