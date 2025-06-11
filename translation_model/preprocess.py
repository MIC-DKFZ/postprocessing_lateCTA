import os, sys
import pandas as pd
import numpy as np
from loguru import logger
import SimpleITK as sitk
from typing import Union
import blosc2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from scipy.ndimage import center_of_mass, shift
import argparse
import time


from nndet.io import load_json, save_json


from monai.transforms import (
   Compose,
   LoadImaged,
   EnsureChannelFirstd,
   Spacingd,
   ScaleIntensityd,
   ResizeWithPadOrCropd,
   ToNumpyd,
)
from monai.data import Dataset, DataLoader
from monai.transforms import MapTransform


def build_data(folder : os.PathLike) -> list:
    """
    Derive dataframe with case IDs and corresponding 
    filenames

    Params
    ------
    folder : input folder

    Returns
    -------
    df : list with filename information
    
    """
    # Inspect raw folder
    cta_folder = os.path.join(folder, "imagesTr")
    label_folder = os.path.join(folder, "labelsTr")
    brain_folder = os.path.join(folder, "brainTr")

    # Build output list of dictionaries for each case of interest
    df = [] 
    cta_files = sorted(os.listdir(cta_folder))
    for cta_file in cta_files:
        if ".nii.gz" in cta_file:
            cid = cta_file.replace(".nii.gz", "")
            cta_file = os.path.join(cta_folder, cta_file)
            time_file = os.path.join(label_folder, 
                                     f"{cid}_time.nii.gz") 
            dist_file = os.path.join(label_folder, 
                                     f"{cid}_dist.nii.gz") 
            brain_file = os.path.join(brain_folder, 
                                     f"{cid}.nii.gz") 
            cid_dict = {"cid" : cid,
                        "cta" : cta_file,
                        "dist" : dist_file,
                        "time" : time_file,
                        "brain" : brain_file} 
            df.append(cid_dict)

    return df


def center_image_on_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
   # Compute the centroid of the mask
   centroid = center_of_mass(mask.astype(bool))


   # Get the image center
   image_center = np.array(image.shape) / 2


   # Calculate the shift needed to move the centroid to the center
   shift_vector = image_center - centroid


   # Shift the image
   centered_image = shift(
       image, shift=shift_vector, order=1, mode="constant", cval=-1024.0
   )


   return centered_image




class CropEmptyAxialSlicesd(MapTransform):
   # Consider only axial slices with brain information, plus center image in the brain mask
   def __init__(self, keys, margin=2):
       super().__init__(keys)
       self.margin = margin


   def __call__(self, data):
       d = dict(data)


       # 🔍 Access the image filename
       mask = sitk.GetArrayFromImage(sitk.ReadImage(d["mask"]))
       ind = np.where(np.sum(mask, (1, 2)) > 0)[0]
       z_min, z_max = int(ind.min()) - self.margin, int(ind.max()) + self.margin


       # Step 1: Convert mask to PyTorch tensor
       mask_torch = torch.from_numpy(mask).bool()  # shape: (Z, Y, X)


       # Step 2: Expand dimensions to match the image shape
       mask_torch = mask_torch.unsqueeze(0)  # shape: (1, Z, Y, X)


       # Step 3: Ensure the mask is on the same device as the image
       mask_torch = mask_torch.to(d["image"].device)


       # Step 4: Set image values to -1024 where mask is False (0)
       d["image"][~mask_torch] = -1024


       for key in self.keys:
           if key in d:
               if d[key].ndim == 3:
                   d[key] = d[key][z_min : z_max + 1]
               elif d[key].ndim == 4:
                   d[key] = d[key][:, z_min : z_max + 1]


       # Center image on mask centroid
       mask_trimmed = mask[z_min : (z_max + 1)]
       image_centered = center_image_on_mask(
           image=d["image"][0].numpy(), mask=mask_trimmed
       )
       d["image"][0] = torch.tensor(image_centered, device=d["image"].device)


       return d




class GlobalNormalize(MapTransform):
   # MONAI transform for global normalization
   def __init__(self, keys, mean, std):
       super().__init__(keys)
       self.mean = mean
       self.std = std


   def __call__(self, data):
       d = dict(data)
       for key in self.keys:
           d[key] = (d[key] - self.mean) / self.std
       return d




class SaveAsBlosc2d(MapTransform):
   """
   Save selected keys of the dictionary as a .b2nd-compressed file.
   """


   def __init__(self, keys, output_dir):
       super().__init__(keys)
       self.output_dir = output_dir
       os.makedirs(self.output_dir, exist_ok=True)


   def __call__(self, data):
       d = dict(data)
       save_path = os.path.join(self.output_dir, f"{d['cid']}.b2nd")


       # Save just the array for the "image" key
       arr = d[self.keys[0]]
       if hasattr(arr, "numpy"):
           arr = arr.numpy()
       arr = np.ascontiguousarray(arr.squeeze())


       """
       plt.figure()
       plt.subplot(311)
       plt.imshow(arr[arr.shape[0] // 2], cmap="gray")
       plt.colorbar()
       plt.subplot(312)
       plt.imshow(arr[:, arr.shape[1] // 2], cmap="gray")
       plt.colorbar()
       plt.subplot(313)
       plt.imshow(arr[:, :, arr.shape[2] // 2], cmap="gray")
       plt.colorbar()
       plt.show()
       """


       # Convert to a Blosc2 NDArray
       blosc2.asarray(arr, urlpath=save_path)


       logger.info(f"Saved to: {save_path}")
       return d  # downstream transforms can continue using uncompressed data




def compute_median_spacing_globalvals(
   data_list: list,
   keys : list,
) -> Union[tuple, dict, dict]:
   """
   Obtain median spacing, global mean and global std
   (focusing on the brain only) from dataset


   Params
   ------
   data_list : filenames for dataset
   keys : keys of interest to be loaded


   Returns
   -------
   median_spacing : derived median spacing, calibrated with the downsampling factor
   global_mean : global mean of images, considering only brain
   global_std : global standard deviation of images, considering only brain


   """
   loader = LoadImaged(keys=keys, image_only=False)

   spacings = []

   # Derive keys to analyze: all but brain keys
   ind_brain = keys.index("brain")
   keys_analyze = keys[:ind_brain] + keys[(ind_brain+1):]  

   # elements for global mean and standard deviation computation
   n_voxels, total_sum, total_sq_sum, global_mean, global_std = {}, {}, {}, {}, {}
   for key in keys_analyze: # Initialization 
       n_voxels[key], total_sum[key], total_sq_sum[key] = 0, 0.0, 0.0

   
   # Iterate through case IDs
   spacings = [] # Store spacings
   for item in data_list:
       loaded = loader(item)
       logger.info(f"Planning {loaded['cid']}")

       # Load brain mask
       mask =  np.swapaxes(loaded["brain"], 0, -1)

       # Determine brain limits
       lims = extract_brain_limits(brain_seg=mask,
                                    dist_map=np.swapaxes(loaded["dist"], 0, -1))
       
       # Derive spacing 
       spacing = np.flip(np.array(loaded[f"{keys[0]}_meta_dict"]["pixdim"][1:4]))  # (Z, Y, X)
       spacings.append(spacing) 
       
       # Iterate through keys of the same case ID, accumulate values 
       # for global mean and global std computation for normalization
       for key in keys_analyze:
           img = np.swapaxes(loaded[key], 0, -1)
           # Restrict image and brain mask to previously derived limits
           trimmed_img = img[lims[0]:(lims[1]+1)] 
           trimmed_mask = mask[lims[0]:(lims[1]+1)] 
           n_voxels[key] += (trimmed_mask > 0).sum() 
           total_sum[key] += trimmed_img[trimmed_mask > 0].sum()
           total_sq_sum[key] += (trimmed_img[trimmed_mask > 0]**2).sum()  

           print(trimmed_img[trimmed_mask > 0].mean(),
                 trimmed_img[trimmed_mask > 0].std(),
                 spacing,
                 trimmed_img.shape,
                 trimmed_mask.shape) 

   # Median spacing derivation
   spacings_np = np.array(spacings).astype(float)
   median_spacing = np.median(spacings_np, axis=0).astype(float)


   # Global mean and standard deviation derivation
   for key in keys_analyze:
       global_mean[key]  = float(total_sum[key] / (n_voxels[key] + np.finfo(float).eps))
       global_std[key]  = float(np.sqrt(total_sq_sum[key] / (n_voxels[key]  + np.finfo(float).eps) - global_mean[key]**2))


   return tuple(median_spacing), global_mean, global_std




class EnsureZYXShapeD(MapTransform):
   """
   Custom MONAI transform to ensure the spatial dimensions are (Z, Y, X).
   Assumes input shape is (C, D, H, W) or (D, H, W), and channel is not confused with spatial axes.
   """


   def __init__(self, keys):
       super().__init__(keys)


   def __call__(self, data):
       d = dict(data)
       d["image"] = torch.swapaxes(d["image"], 1, -1)
       return d




def get_preprocessing_transforms(
   global_mean: float,
   global_std: float,
   preprocessed_folder: os.PathLike,
   downsample: float = 1.0,
   key: str = "Tr",
   patch_size: tuple = (96, 96, 96),
   median_spacing: tuple = (1.0, 1.0, 1.0),
):


   out_folder = os.path.join(preprocessed_folder, f"images{key}")
   if not (os.path.exists(out_folder)):
       os.makedirs(out_folder)


   spacing = [sp * downsample for sp in median_spacing]
   transforms = [
       LoadImaged(keys=["image"], meta_keys=["image"]),
       EnsureChannelFirstd(keys=["image"]),
       EnsureZYXShapeD(keys=["image"]),
       CropEmptyAxialSlicesd(
           keys=["image"],
       ),
       Spacingd(keys=["image"], pixdim=spacing, mode="bilinear"),
       ScaleIntensityd(keys=["image"]),
       ResizeWithPadOrCropd(keys=["image"], spatial_size=patch_size),
       GlobalNormalize(keys=["image"], mean=global_mean, std=global_std),
       ToNumpyd(keys=["image"]),
       SaveAsBlosc2d(keys=["image"], output_dir=out_folder),
   ]


   return Compose(transforms)



def skip_processed(data: list, preprocessed: os.PathLike, key: str = "Tr") -> list:
   """
   Skip file if already processed


   Params
   ------
   data : data objects
   preprocessed : preprocessed folder


   Returns
   -------
   data_out : filtered dataset, only with non-processed cases


   """
   data_out = []
   for d in data:
       outfile = os.path.join(preprocessed, f"images{key}", f"{d['cid']}.b2nd")
       if not os.path.exists(outfile):
           data_out.append(d)


   return data_out




def apply_window(image: np.ndarray, window_center: float, window_width: float):
   """
   Apply window to image


   Params
   ------
   image : input image
   window_center : window center
   window_width : window width


   Returns
   -------
   windowed : windowed image


   """
   lower = window_center - window_width / 2
   upper = window_center + window_width / 2
   windowed = np.clip(image, lower, upper)
   windowed = (windowed - lower) / window_width  # Normalize to [0, 1]
   return windowed


def extract_brain_limits(brain_seg : np.ndarray, dist_map : np.ndarray) -> list:
    """
    Locate first and last axial slices where the brain spans
    and there is distance information, since the 
    preprocessing will only focus on those slices

    Params
    ------
    brain_seg : input brain segmentation
    dist_map : distance map

    Returns
    -------
    lims : brain limits
    
    """
    # Locate limits in brain segmentation
    brain_axial_coords = np.where(brain_seg > 0)[0]
    brain_lims = [brain_axial_coords.min(), 
                  brain_axial_coords.max()] 
    
    # Locate limits in distance maps
    distance_axial_coords = np.where(dist_map < dist_map.max())[0] 
    dist_lims = [distance_axial_coords.min(),
                 distance_axial_coords.max()]

    # Combine limit definitions
    lims = [max(brain_lims[0] , dist_lims[0]),
            min(brain_lims[-1] , dist_lims[-1])] 

    return lims   



def preprocess_dataset():

   # Load configuration
   cfg_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cfg_translation.json")

   assert (
       os.path.exists(cfg_file) and ".json" in cfg_file
   ), f"Configuration file '{cfg_file}' does not exist or is not .json"
   cfg = load_json(cfg_file)
   cfg_keys = list(cfg.keys())
   #required_keys = ["patch_size", "workers", "downsample"]

   #for k in required_keys:
       # Check that all required keys are present
       #assert k in cfg_keys, f"'{k}' not in configuration"  


   # Determine folders
   raw_splitted = os.path.join(os.getenv("data_folder"), "raw_splitted")
   preprocessed = os.path.join(os.getenv("data_folder"), "preprocessed")


   if not (os.path.exists(preprocessed)):
       os.makedirs(preprocessed)


   # Set up logger
   logger.remove()
   logger.add(
       sys.stdout,
       format="<level>{level}</level>: {message}",
       level="INFO",
       colorize=True,
   )
   log_file = os.path.join(preprocessed, "preprocessing.log")
   logger.add(log_file, level="INFO")

   # Set up plan file, and load plan if it already exists
   plan_file = os.path.join(preprocessed, "plan.json")
   plan = {}
   keys_analysis = ["cta", "time", "dist"]
   if os.path.exists(plan_file):
       # Load precomputed values from previous runs
       plan = load_json(plan_file)
       median_spacing = plan["spacing"]
       global_mean, global_std = {}, {}
       for keys in keys_analysis:
          global_mean[keys]  = plan[f"stats_{keys}"]["mean"]
          global_std[keys]  = plan[f"stats_{keys}"]["std"]

   # Data preparation
   logger.info("Preparing data...")
   data = build_data(folder=raw_splitted)

   # Planning
   logger.info(
       "🔍 Computing median spacing, global mean, and global std from training data..."
   )  

   if not (os.path.exists(plan_file)):
       keys = ["cta", "brain", "time", "dist"] 
       median_spacing, global_mean, global_std = compute_median_spacing_globalvals(data_list = data,
                                                                                   keys = keys)
       
       # Store planning information
       plan["spacing"] = list(median_spacing)
       
       keys_analysis = list(global_mean.keys())
       for key in keys_analysis:
          stats_key = {}
          stats_key["mean"], stats_key["std"] = global_mean[key], global_std[key]    
          plan[f"stats_{key}"] = stats_key

       # Save plan file
       save_json(data=plan, path=plan_file)


   logger.info(f"✅ Median spacing (Z, Y, X): {median_spacing}")
   for keys in keys_analysis:
      logger.info(f"✅ Global mean, global std: {global_mean[keys]}, {global_std[keys]}")

   sys.exit()
   transforms = get_preprocessing_transforms(
       patch_size=cfg["patch_size"],
       downsample=cfg["downsample"],
       preprocessed_folder=preprocessed,
       median_spacing=median_spacing,
       global_mean=global_mean,
       global_std=global_std,
   )
   data = skip_processed(data=data, preprocessed=preprocessed)
   dataset = Dataset(data=data, transform=transforms)
   dataloader = DataLoader(dataset, batch_size=1, num_workers=cfg["workers"])


   for batch in dataloader:
       # Skip files that have already been preprocessed
       outfile = os.path.join(preprocessed, "imagesTr", f"{batch['cid'][0]}.b2nd")
       if not (os.path.exists(outfile)):
           images = batch["image"]
           labels = batch["label"]
           logger.info(f"Processed batch shape: {images.shape}, labels: {labels}")




if __name__ == "__main__":
    t1 = time.time()
    preprocess_dataset()
    print(f"Time ellapsed: {time.time()-t1}sec")
