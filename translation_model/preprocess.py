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
   key_image: str = "image",
   key_mask: str = "mask",
   margin: int = 2,
) -> Union[tuple, float, float]:
   """
   Obtain median spacing, global mean and global std
   (focusing on the brain only) from dataset


   Params
   ------
   data_list : filename and labels for dataset
   downsample : downsampling factor
   key_image : MONAI key for image (image filename)
   key_mask : MONAI key for mask (mask filename)
   margin : margin for image cropping


   Returns
   -------
   median_spacing : derived median spacing, calibrated with the downsampling factor
   global_mean : global mean of images, considering only brain
   global_std : global standard deviation of images, considering only brain


   """
   loader = LoadImaged(keys=[key_image, key_mask], meta_keys=["image", "mask"])
   spacings = []


   # elements for global mean and standard deviation computation
   n_voxels = 0
   total_sum = 0.0
   total_sq_sum = 0.0
   for item in data_list:
       loaded = loader(item)
       logger.info(f"Planning {loaded['cid']}")
       spacing = np.flip(np.array(loaded[key_image].meta["pixdim"][1:4]))  # (Z, Y, X)
       img, mask = np.swapaxes(loaded[key_image].numpy(), 0, -1), np.swapaxes(
           loaded[key_mask].numpy(), 0, -1
       )


       # 🧠 Perform brain mask-based cropping
       z_nonzero = np.any(mask > 0, axis=(1, 2))


       # Mask out with brain
       inv_mask = (1 - mask).astype(bool)
       img[inv_mask] = -1024
       mask = mask.astype(bool)


       if mask.sum() > 0:
           # Restrict image to axial slices where brain mask is present, with a margin of two slices
           z_min, z_max = np.where(z_nonzero)[0][[0, -1]]
           z_min = max(0, z_min - margin)
           z_max = min(mask.shape[0] - 1, z_max + margin)
           img = img[z_min : z_max + 1]
           mask = mask[z_min : z_max + 1]


       # Update parameters for image size
       # n_voxels += img.size
       # total_sum += img.sum()
       # total_sq_sum += (img**2).sum()
       # Compute HU stats only on brain voxels
       n_voxels += mask.sum()
       total_sum += img[mask].sum()
       total_sq_sum += (img[mask] ** 2).sum()
       print(
           img.sum() / img.size,
           img.std(),
           img[mask].sum() / mask.sum(),
           img[mask].std(),
       )


       spacings.append(spacing)


   # Median spacing derivation
   spacings_np = np.array(spacings).astype(float)
   median_spacing = np.median(spacings_np, axis=0)


   # Global mean and standard deviation derivation
   global_mean = total_sum / n_voxels
   global_std = np.sqrt(total_sq_sum / n_voxels - global_mean**2)


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




def load_dataset(
   csv_path: os.PathLike, brain_path: os.PathLike, key: str = "Tr"
) -> list:
   """
   Load dataset from CSV label file


   Params
   ------
   csv_path : CSV file
   brain_path : brain segmentation file path
   key : key for dataset ("Tr" for training, "Ts" for testing)


   Returns
   -------
   data : output paths and labels


   """
   df = pd.read_csv(csv_path)
   data = []
   for i in range(df.shape[0]):
       img_file = os.path.join(
           os.getenv("data_folder"),
           "raw_splitted",
           f"images{key}",
           f"{str(df.iloc[i,0])}_0000.nii.gz",
       )
       brain_file = os.path.join(brain_path, f"{str(df.iloc[i,0])}.nii.gz")
       if os.path.exists(img_file) and os.path.exists(brain_file):
           d = {
               "image": os.path.join(
                   os.getenv("data_folder"),
                   "raw_splitted",
                   f"images{key}",
                   f"{str(df.iloc[i,0])}_0000.nii.gz",
               ),
               "cid": str(df.iloc[i, 0]),
               "mask": os.path.join(brain_path, f"{str(df.iloc[i,0])}.nii.gz"),
               "label": int(df.iloc[i, 1]),
           }
           data.append(d)


   return data




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




def preprocess_dataset():

   # Load configuration
   cfg_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cfg_translation.json")

   assert (
       os.path.exists(cfg_file) and ".json" in cfg_file
   ), f"Configuration file '{cfg_file}' does not exist or is not .json"
   cfg = load_json(cfg_file)
   cfg_keys = list(cfg.keys())
   required_keys = ["patch_size", "workers", "brain_path", "downsample"]

   for k in required_keys:
       # Check that all required keys are present
       assert k in cfg_keys, f"'{k}' not in configuration"  


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


   # Set up label file
   csv_path = os.path.join(raw_splitted, "labelsTr.csv")
   data = load_dataset(csv_path, brain_path=cfg["brain_path"])


   # Set up plan file, and load plan if it already exists
   plan_file = os.path.join(preprocessed, "plan.json")
   plan = {}
   if os.path.exists(plan_file):
       # Load precomputed values from previous runs
       plan = load_json(plan_file)
       median_spacing = plan["spacing"]
       global_mean = plan["mean"]
       global_std = plan["std"]


   # Planning
   logger.info(
       "🔍 Computing median spacing, global mean, and global std from training data..."
   )
   if not (os.path.exists(plan_file)):
       median_spacing, global_mean, global_std = compute_median_spacing_globalvals(
           data
       )
       plan["spacing"] = list(median_spacing)
       plan["mean"], plan["std"] = float(global_mean), float(global_std)


       # Save plan file
       print(plan["spacing"], plan["mean"], plan["std"])
       save_json(data=plan, path=plan_file)


   logger.info(f"✅ Median spacing (Z, Y, X): {median_spacing}")
   logger.info(f"✅ Global mean, global std: {global_mean}, {global_std}")


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
