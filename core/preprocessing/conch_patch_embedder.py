from tqdm import tqdm 
import numpy as np 
import h5py 

import torch 
from torch.utils.data import Dataset

from conch.open_clip_custom import create_model_from_pretrained


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


class TileEmbedder:
	def __init__(self, 
				 model_name='conch_ViT-B-16',
				 model_repo='hf_hub:MahmoodLab/conch',
				 device='cuda',
				 precision=torch.float32):
		self.model_name = model_name
		self.model_repo = model_repo
		self.device = device
		self.precision = precision
		self.model, self.img_transforms = self._build_conch_model()

	def _build_conch_model(self):
		model, eval_transform = create_model_from_pretrained(self.model_name, self.model_repo, force_image_size=224)
		return model, eval_transform

	def embed_tiles(self, wsi, tile_h5_path, embedding_save_path) -> str:
		dataset = TileDataset(wsi.img, tile_h5_path, eval_transform=self.img_transforms)
		
		dataloader = torch.utils.data.DataLoader(
			dataset, 
			batch_size=64, 
			shuffle=False,
			num_workers=8,
			collate_fn=collate_features,
		)

		self.model.to(self.device)
		self.model.eval()
		
		for batch_idx, (imgs, coords) in tqdm(enumerate(dataloader), total=len(dataloader)):
			imgs = imgs.to(self.device)
			with torch.inference_mode(), torch.amp.autocast(dtype=self.precision, device_type=self.device):
				embeddings = self.model.encode_image(imgs, proj_contrast=False, normalize=False)
			mode = 'w' if batch_idx == 0 else 'a'
			asset_dict = {
				'features': embeddings.cpu().numpy(),
				'coords': coords,
			}
			save_hdf5(embedding_save_path, mode=mode, asset_dict=asset_dict)
		
		return embedding_save_path


class NEWTileDataset(Dataset):
	def __init__(self, wsi, coords_h5_fpath, eval_transform=None):
		self.wsi = wsi
		self.coords_h5_fpath = coords_h5_fpath
		self.eval_transform = eval_transform
		self._load_coords()

		fishing_rod_downsample = self.target_patch_size / self.patch_size
		self.patcher = WSIPatcher(self.target_patch_size, fishing_rod_downsample, 1, custom_coords=self.coords)

	def _load_coords(self):
		with h5py.File(self.coords_h5_fpath, "r") as f:
			self.attr_dict = {k: dict(f[k].attrs) for k in f.keys() if len(f[k].attrs) > 0}
			self.coords = f['coords'][:]
			self.patch_size = f['coords'].attrs['patch_size']
			self.custom_downsample = f['coords'].attrs['custom_downsample']
			self.target_patch_size = int(self.patch_size) // int(self.custom_downsample) if self.custom_downsample > 1 else self.patch_size

	def __len__(self):
		return len(self.patcher)

	def __getitem__(self, idx):
		img = patcher[idx]
		img = self.eval_transform(img).unsqueeze(dim=0)
		return img, coord


class TileDataset(Dataset):
	def __init__(self, wsi, coords_h5_fpath, eval_transform=None):
		self.wsi = wsi
		self.coords_h5_fpath = coords_h5_fpath
		self.eval_transform = eval_transform
		self._load_coords()

	def _load_coords(self):
		with h5py.File(self.coords_h5_fpath, "r") as f:
			self.attr_dict = {k: dict(f[k].attrs) for k in f.keys() if len(f[k].attrs) > 0}
			self.coords = f['coords'][:]
			self.patch_level = f['coords'].attrs['patch_level']
			self.patch_size = f['coords'].attrs['patch_size']
			self.length = len(self.coords)
			self.custom_downsample = f['coords'].attrs['custom_downsample']
			self.target_patch_size = int(self.patch_size) // int(self.custom_downsample) if self.custom_downsample > 1 else self.patch_size

	def __len__(self):
		return self.length

	def read_region(self, coord):
		img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
		if self.custom_downsample > 1:
			img = img.resize((self.target_patch_size,)*2)
		return img

	def __getitem__(self, idx):
		coord = self.coords[idx]
		img = self.read_region(coord)
		img = self.eval_transform(img).unsqueeze(dim=0)
		return img, coord
