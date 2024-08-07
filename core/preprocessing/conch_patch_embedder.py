from tqdm import tqdm 
import numpy as np 
import h5py 
from PIL import Image 
import cv2
import torch 
from torch.utils.data import Dataset

from conch.open_clip_custom import create_model_from_pretrained

import pdb 


def build_conch_model():
	model, eval_transform = create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/conch", force_image_size=224)
	return model, eval_transform


def save_hdf5(output_fpath, 
				  asset_dict, 
				  attr_dict= None, 
				  mode='a', 
				  auto_chunk = True,
				  chunk_size = None):
	"""
	output_fpath: str, path to save h5 file
	asset_dict: dict, dictionary of key, val to save
	attr_dict: dict, dictionary of key: {k,v} to save as attributes for each key
	mode: str, mode to open h5 file
	auto_chunk: bool, whether to use auto chunking
	chunk_size: if auto_chunk is False, specify chunk size
	"""
	with h5py.File(output_fpath, mode) as f:
		for key, val in asset_dict.items():
			data_shape = val.shape
			if len(data_shape) == 1:
				val = np.expand_dims(val, axis=1)
				data_shape = val.shape

			if key not in f: # if key does not exist, create dataset
				data_type = val.dtype
				if data_type == np.object_: 
					data_type = h5py.string_dtype(encoding='utf-8')
				if auto_chunk:
					chunks = True # let h5py decide chunk size
				else:
					chunks = (chunk_size,) + data_shape[1:]
				try:
					dset = f.create_dataset(key, 
											shape=data_shape, 
											chunks=chunks,
											maxshape=(None,) + data_shape[1:],
											dtype=data_type)
					### Save attribute dictionary
					if attr_dict is not None:
						if key in attr_dict.keys():
							for attr_key, attr_val in attr_dict[key].items():
								dset.attrs[attr_key] = attr_val
					dset[:] = val
				except:
					print(f"Error encoding {key} of dtype {data_type} into hdf5")
				
			else:
				dset = f[key]
				dset.resize(len(dset) + data_shape[0], axis=0)
				assert dset.dtype == val.dtype
				dset[-data_shape[0]:] = val
	
	return output_fpath


def collate_features(batch):
	features = torch.cat([item[0] for item in batch], dim=0)
	coords = np.vstack([item[1] for item in batch])
	return features, coords


def embed_tiles(
		wsi, 
		tile_h5_path,
		embedding_save_path,
		device='cuda',
		precision=torch.float32
	):
	"""
	Extract embeddings from tiles using encoder and save to h5 file
	"""

	model, img_transforms = build_conch_model()

	dataset = WSIBagDataset(wsi.img, tile_h5_path, eval_transform=img_transforms)
	dataloader = torch.utils.data.DataLoader(
		dataset, 
		batch_size=64, 
		shuffle=False,
		num_workers=8,
		collate_fn=collate_features,
	)

	model.to(device)
	model.eval()
	for batch_idx, (imgs, coords) in tqdm(enumerate(dataloader), total=len(dataloader)):

		imgs = imgs.to(device)
		with torch.inference_mode(), torch.cuda.amp.autocast(dtype=precision):
			embeddings = model.encode_image(imgs, proj_contrast=False, normalize=False)
		if batch_idx == 0:
			mode = 'w'
		else:
			mode = 'a'
		asset_dict = {
			'features': embeddings.cpu().numpy(),
			'coords': coords,
			}
		# asset_dict.update({key: np.array(val) for key, val in batch.items() if key != 'imgs'})
		save_hdf5(
			embedding_save_path,
			mode=mode,
			asset_dict=asset_dict,
		)
	return embedding_save_path 


class WSIBagDataset(torch.utils.data.Dataset):
	def __init__(
			self, 
			wsi, 
			coords_h5_fpath, 
			eval_transform=None, 
			verbose=0
	):
		"""
		Args:
			- slide_fpath (str): Path to WSI.
			- coords_h5_fpath (string): Path to h5 file containing coordinates.
			- eval_transform (torchvision.transform): Which Transform to use.
			- thresh (float): For 4K mode patching only.
			- verbose (int): Printing attribute information in the coords_h5 file.
		"""
		self.wsi = wsi
		self.coords_h5_fpath = coords_h5_fpath
		self.eval_transform = eval_transform

		with h5py.File(self.coords_h5_fpath, "r") as f:
			self.attr_dict = {}
			for k in f.keys():
				attrs = dict(f[k].attrs)
				if len(attrs.keys()) > 0: self.attr_dict[k] = attrs
			
			self.coords = f['coords'][:]
			self.patch_level = f['coords'].attrs['patch_level']
			self.patch_size = f['coords'].attrs['patch_size']
			self.length = len(self.coords)

			self.custom_downsample = f['coords'].attrs['custom_downsample']
			if self.custom_downsample > 1:
				self.target_patch_size = self.patch_size // self.custom_downsample
			else:
				self.target_patch_size = self.patch_size

		if verbose:
			self.summary()

	def __len__(self):
		return self.length

	def summary(self):
		with h5py.File(self.coords_h5_fpath, "r") as f:
			print('Coord Attrs:')
			for k,v in f['coords'].attrs.items():
				print(f'\t{k}: {v}')
			print('Feature Extraction Settings')
			print('\tNum Coords', self.length)
			print('\tTarget Patch Size: ', self.target_patch_size)
			print('\tTransformations: ', self.eval_transform)

	def __os_read_region(self, coord):
		try:
			img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
			if self.custom_downsample > 1:
				img = img.resize((int(self.patch_size) // int(self.custom_downsample),)*2)
			return img
		except:
			print('Error reading region. Returning white image.')
			return Image.fromarray(np.ones((self.patch_size, self.patch_size, 3), dtype=np.uint8)*255)

	def __getitem__(self, idx):
		coord = self.coords[idx]
		img = self.__os_read_region(coord)
		img = self.eval_transform(img).unsqueeze(dim=0)
		return img, coord
