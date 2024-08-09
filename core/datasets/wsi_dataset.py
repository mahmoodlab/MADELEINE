from __future__ import print_function, division
import os
import pandas as pd
import h5py

import torch 
import torch.utils.data
from torch.utils.data import Dataset
import numpy as np

import pdb 


def load_features(h5_path):
    with h5py.File(h5_path, 'r') as hdf5_file:
        feats = hdf5_file['features'][:].squeeze()
    if isinstance(feats, np.ndarray):
        feats = torch.Tensor(feats)
    return feats

class SlideDataset(Dataset):
    def __init__(self, dataset_name, csv_path, features_path, modalities, embedding_size=None, sample=-1, train=True):
        """
        Args:
            dataset_name (string) : name of dataset for differential handling 
            csv_path (string): Path to the csv file with labels and slide_id.
            features_path (string): Directory with all the feature files.
            sample (int): Number of tokens to sample per modality. Default: no sampling. 
            modalities (string): he or all. 
        """
        self.dataset_name = dataset_name
        self.dataframe = pd.read_csv(csv_path)
        self.features_path = features_path
        self.modalities = modalities
        self.sample = sample
        self.train = train
        self.embedding_size = embedding_size

    def __len__(self):
        return len(self.dataframe)
    
    def sample_n(self, feats):
        if self.sample > -1:
            if feats.shape[0] < self.sample:
                patch_indices = torch.randint(0, feats.shape[0], (self.sample,))
                feats = feats[patch_indices]
            else:
                patch_indices = torch.randperm(feats.shape[0])[:self.sample]
                feats = feats[patch_indices]
        return feats

    def __getitem__(self, index):
        
        # common to all datasets
        slide_id = self.dataframe.iloc[index, self.dataframe.columns.get_loc('slide_id')]
        modality_labels = [self.dataframe.iloc[index, self.dataframe.columns.get_loc(modality)] for modality in self.modalities]
        
        if self.train:
            
            split_type = self.dataframe.iloc[index, self.dataframe.columns.get_loc('split')]
            special_id = "" if split_type == "train" else f"_{split_type}"
            
            all_feats = []
            for modality, modality_label in zip(self.modalities, modality_labels):
                curr_h5_path = os.path.join(self.features_path, f"{slide_id}_{modality}{special_id}.h5")
                curr_feats = load_features(curr_h5_path) if modality_label == 1 else torch.zeros([2, self.embedding_size])
                curr_feats = self.sample_n(curr_feats)
                all_feats.append(curr_feats)

        
        else:
            
            curr_h5_path = os.path.join(self.features_path, f"{slide_id}.h5")
            curr_feats = load_features(curr_h5_path)
            all_feats = [curr_feats]
            modality_labels = [1]
            
        data = {
            'feats': all_feats,
            'modality_labels': modality_labels,
            'slide_id': slide_id
        }
        
        return data

def collate(batch):
        # Create separate lists for features and labels
        slide_ids = [item['slide_id'] for item in batch]
        batch_features = [torch.stack(item['feats']) for item in batch]
        batch_labels = [torch.Tensor(item['modality_labels']) for item in batch]
        
        batch_features_stacked = torch.stack(batch_features)
        batch_labels_stacked = torch.stack(batch_labels)
        
        return {
            "feats" : batch_features_stacked,
            "modality_labels" : batch_labels_stacked,
            'slide_ids' : slide_ids
        }


class SimpleDataset(Dataset):
    def __init__(self, features_path):
        """
        Args:
            features_path (string): Directory with all the feature files.
        """
        self.features_path = features_path
        self.fnames = os.listdir(self.features_path)
        self.fnames = [fn for fn in self.fnames if fn.endswith('.h5')]

    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, index):
        curr_h5_path = os.path.join(self.features_path, self.fnames[index])
        features = load_features(curr_h5_path)
        slide_id = os.path.splitext(self.fnames[index])[0]
        return features, slide_id


def simple_collate(batch):
    features, slide_ids = zip(*batch)
    features_batch = torch.stack(features)
    return features_batch, list(slide_ids)