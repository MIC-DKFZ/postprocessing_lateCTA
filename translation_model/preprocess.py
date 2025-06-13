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
from joblib import Parallel, delayed
from scipy.ndimage import center_of_mass, shift
import argparse
import time

from monai.apps.nnunet import nnUNetV2Runner
from monai.transforms import (
   Compose,
   LoadImaged,
   EnsureChannelFirstd,
   Spacingd,
   ToNumpyd,
)
from monai.data import Dataset, DataLoader
from monai.transforms import MapTransform

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(script_dir))
from utils.load_save import write_data, load_data


def build_data_crop_parallel(folder : os.PathLike, cta_file : os.PathLike, task_id : str, key : str ="Tr") -> dict:
    """
    Prepare dataset for preprocessing and apply cropping in parallel

    Params
    ------
    folder : folder with raw data
    cta_file : CTA file of interest
    task_id : task folder
    key : whether to build data and crop for the training set ("Tr") or for the test set ("Ts")

    Returns
    -------
    cid_dict : relevant information for case ID of interest
    
    """

    # Inspect raw folder
    cta_folder = os.path.join(folder, f"images{key}")
    label_folder = os.path.join(folder, f"labels{key}")
    brain_folder = os.path.join(folder, f"brain{key}")

    # Provide cropped folder
    crop_folder = os.path.join(os.path.dirname(os.path.dirname(folder)), "raw_cropped")
    cta_crop_folder = os.path.join(crop_folder, task_id, f"images{key}")
    label_crop_folder = os.path.join(crop_folder, task_id, f"labels{key}") 
    brain_crop_folder = os.path.join(crop_folder, task_id, f"brain{key}") 

    if not(os.path.exists(cta_crop_folder)):
        os.makedirs(cta_crop_folder)
    if not(os.path.exists(label_crop_folder)):
        os.makedirs(label_crop_folder)
    if not(os.path.exists(brain_crop_folder)):
        os.makedirs(brain_crop_folder)

    if ".nii.gz" in cta_file:
        # Fill in data information
        cid = cta_file.replace(".nii.gz", "") 

        cta_crop_file = os.path.join(cta_crop_folder, cta_file)
        cta_crop_file = cta_crop_file.replace(".nii.gz", "_0000.nii.gz")
        time_crop_file = os.path.join(label_crop_folder, 
                                    f"{cid}_time.nii.gz") 
        dist_crop_file = os.path.join(label_crop_folder, 
                                    f"{cid}_dist.nii.gz") 
        brain_crop_file = os.path.join(brain_crop_folder, 
                                    f"{cid}.nii.gz") 
        bin_crop_file = os.path.join(label_crop_folder, 
                                    f"{cid}.nii.gz")
        
        cid_dict = {"cid" : cid,
                    "cta" : cta_crop_file,
                    "dist" : dist_crop_file,
                    "time" : time_crop_file,
                    "brain" : brain_crop_file} 

        # Crop data
        if not(os.path.exists(cta_crop_file)) or not(os.path.exists(dist_crop_file)) or not(os.path.exists(brain_crop_file)) or not(os.path.exists(bin_crop_file)):
            cta_file = os.path.join(cta_folder, cta_file)
            time_file = os.path.join(label_folder, 
                                    f"{cid}_time.nii.gz") 
            dist_file = os.path.join(label_folder, 
                                    f"{cid}_dist.nii.gz") 
            brain_file = os.path.join(brain_folder, 
                                    f"{cid}.nii.gz")
            
        
            cta_image = sitk.ReadImage(cta_file)
            cta_img = sitk.GetArrayFromImage(cta_image)
            
            time_image = sitk.ReadImage(time_file)
            time_img = sitk.GetArrayFromImage(time_image)
            bin_img = (time_img > 0).astype(np.uint8)

            dist_image = sitk.ReadImage(dist_file)
            dist_img = sitk.GetArrayFromImage(dist_image)

            brain_image = sitk.ReadImage(brain_file)
            brain_img = sitk.GetArrayFromImage(brain_image)

            lims = extract_brain_limits(brain_seg=brain_img,
                                        cta = cta_img)
            
            # Restrict CTA image only to the brain
            cta_img_copy = cta_img.copy()
            cta_img_copy[dist_img == dist_img.max()] = -1024
            
            imgs = [cta_img_copy, time_img, 
                    dist_img, brain_img, bin_img]
            outfiles = [cta_crop_file, time_crop_file, 
                        dist_crop_file, brain_crop_file, bin_crop_file] 
            
            print(f"Cropping {cid}...")
            for img, outfile in zip(imgs, outfiles):
                cropping(img = img, lims = lims, 
                            image=time_image, outfile=outfile)
                    
    return cid_dict


def create_datalist(folder : os.PathLike) -> dict:
    """
    Create data list for MONAI patch size and architecture estimation

    Params
    ------
    folder : input data folder

    Returns
    -------
    datalist : resulting data list object
    
    """

    # Determine training data
    train_folder = os.path.join(folder, "imagesTr")
    train_label_folder = os.path.join(folder, "labelsTr")
    train_files = sorted(os.listdir(train_folder))

    assert os.path.exists(train_folder) and os.path.exists(train_label_folder), f"Train folder '{train_folder}' or train label folder '{train_label_folder}' do not exist"

    # Determine testing data
    test_folder = os.path.join(folder, "imagesTs")
    test_label_folder = os.path.join(folder, "labelsTs")

    # Save training information
    train_info = []
    for f in train_files:
        if ".nii.gz" in f:
            cid = f.replace(".nii.gz", "")
            full_file = os.path.join(train_folder, f)
            label_file = os.path.join(train_label_folder, f"{cid}_time.nii.gz")

            assert os.path.exists(full_file) and os.path.exists(label_file), f"CTA file '{full_file}' or label file '{label_file}' do not exist"

            d = {"image" : [full_file],
                 "label" : label_file}  
            train_info.append(d)


    # Save testing information
    test_info = []
    if os.path.exists(test_folder):
        test_files = sorted(os.listdir(test_folder))
        for f in test_files:
            if ".nii.gz" in f:
                cid = f.replace(".nii.gz", "")
                full_file = os.path.join(test_folder, f)
                label_file = os.path.join(test_label_folder, f"{cid}_time.nii.gz")

                assert os.path.exists(full_file) and os.path.exists(label_file), f"CTA file '{full_file}' or label file '{label_file}' do not exist"

                d = {"image" : [full_file],
                    "label" : label_file}  
                test_info.append(d)

    # Provide final datalist information
    datalist = {"training" : train_info,
                "validation" : [],
                "test" : test_info}
    
    return datalist



def create_configyaml(task_id : str):
    """
    Create config.yaml file for derivation of patch size and 
    architecture parameters

    Params
    ------
    task_id : task ID
    
    """
    config = {"name" : task_id,
              "modality" : "CT",
              "dataroot" : os.path.join(os.getenv("data_folder")),
              "datalist" : os.path.join(os.getenv("data_folder"), "preprocessed", task_id, "datalist.json"),
              "nnunet_raw" : os.path.join(os.getenv("data_folder"), "raw_cropped"),
              "nnunet_preprocessed" : os.path.join(os.getenv("data_folder"), "preprocessed"),
              "nnunet_results" : os.path.join(os.getenv("model_folder")),
              #"dataset_name_or_id" : int(task_id[(-3):]),
              "preprocessing": {
                    "planner": {
                        "device": "cuda",  # or "cpu"
                        "allowed_memory_mb": 11000,  # limit to 11GB GPU memory
                        # Optionally add spacing override or more planner fields:
                        # "target_spacing": [1.0, 1.0, 1.0],
                    }}
                }
    
    return config


def create_datasetjson(task_id : str, num_training : int, outfile : os.PathLike):
    """
    Create dataset.json file for patch size and architecture
    optimization

    Params
    ------
    task_id : task information
    num_training : number of training cases
    outfile : output file

    Returns
    -------
    Saved dataset.json file
    
    """
    dataset = {"channel_names" : {"0" : "CT"},
               "labels" : {"background" : 0,
                           "lesion" : 1},
               "numTraining" : num_training,
               "file_ending" : ".nii.gz",
               "name" : task_id,
               "reference" : "",
               "release" : "",
               "description" : "",
               "overwrite_image_reader_writer" : "NibabelIOWithReorient"}
    
    write_data(data=dataset, 
               filename=outfile)



def build_data_crop(folder : os.PathLike, task_id : str, workers : int = 6, key : str = "Tr") -> list:
    """
    Derive dataframe with case IDs and corresponding 
    filenames
    At the same time, crop the data with the corresponding brain 
    and distance map information

    Params
    ------
    folder : input folder
    task_id : task folder
    workers : parallel workers (default: 6)
    key : whether to build data and crop for the training set ("Tr") or for the test set ("Ts")

    Returns
    -------
    df : list with filename information
    
    """
    # Inspect raw folder
    cta_folder = os.path.join(folder, f"images{key}")
    label_folder = os.path.join(folder, f"labels{key}")
    brain_folder = os.path.join(folder, f"brain{key}")

    # Provide cropped folder
    crop_folder = os.path.join(os.path.dirname(folder), "raw_cropped", task_id)
    cta_crop_folder = os.path.join(crop_folder, f"images{key}")
    label_crop_folder = os.path.join(crop_folder, f"labels{key}") 
    brain_crop_folder = os.path.join(crop_folder, f"brain{key}") 

    # Create folder if it does not exist
    folders = [cta_crop_folder, label_crop_folder, brain_crop_folder]
    for f in folders: 
        if not(os.path.exists(f)):
            os.makedirs(f)

    # Build output list of dictionaries for each case of interest
    df = [] 
    cta_files = sorted(os.listdir(cta_folder))
    df = Parallel(n_jobs=workers)(delayed(build_data_crop_parallel)(folder, cta_file, task_id, key) for cta_file in cta_files)
        
    return df


def cropping(img : np.ndarray, lims : list, image, outfile : os.PathLike):
    """
    Apply image cropping

    Params
    ------
    img : input image
    lims : croppping limits
    image : sitk reference image
    outfile : output file
    
    Returns
    -------
    Saved cropped image in output file
    
    """
    crop_img = img[lims[0]:(lims[1]+1)]
    crop_image = sitk.GetImageFromArray(crop_img)
    crop_image.SetOrigin(image.GetOrigin())
    crop_image.SetSpacing(image.GetSpacing())
    crop_image.SetDirection(image.GetDirection())

    sitk.WriteImage(crop_image, outfile)


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
       self.keys = keys


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
       mask_torch = mask_torch.to(d[self.keys[0]].device)


       # Step 4: Set image values to -1024 where mask is False (0)
       d[self.keys[0]][~mask_torch] = -1024


       for key in self.keys:
           if key in d:
               if d[key].ndim == 3:
                   d[key] = d[key][z_min : z_max + 1]
               elif d[key].ndim == 4:
                   d[key] = d[key][:, z_min : z_max + 1]


       # Center image on mask centroid
       mask_trimmed = mask[z_min : (z_max + 1)]
       image_centered = center_image_on_mask(
           image=d[self.keys[0]][0].numpy(), mask=mask_trimmed
       )
       d[self.keys[0]][0] = torch.tensor(image_centered, 
                                         device=d[self.keys[0]].device)


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
       self.key = keys[0]
       self.output_dir = output_dir
       os.makedirs(self.output_dir, exist_ok=True)


   def __call__(self, data):
       d = dict(data)
       # Determine filename suffix to store data
       suffix = "0000"
       if self.key.lower().strip() == "time":
           suffix = "time"
       elif self.key.lower().strip() == "dist":
           suffix = "dist" 
       save_path = os.path.join(self.output_dir, 
                                f"{d['cid']}_{suffix}.b2nd")

       # Save just the array for the "image" key
       arr = d[self.keys[0]]
       if hasattr(arr, "numpy"):
           arr = arr.numpy()
       arr = np.ascontiguousarray(arr.squeeze())
       
       #plt.figure()
       #plt.subplot(311)
       #plt.imshow(arr[arr.shape[0] // 2], cmap="gray")
       #plt.colorbar()
       #plt.subplot(312)
       #plt.imshow(arr[:, arr.shape[1] // 2], cmap="gray")
       #plt.colorbar()
       #plt.subplot(313)
       #plt.imshow(arr[:, :, arr.shape[2] // 2], cmap="gray")
       #plt.colorbar()
       #plt.show()

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
       #lims = extract_brain_limits(brain_seg=mask,
       #                             dist_map=np.swapaxes(loaded["dist"], 0, -1))
       
       # Derive spacing 
       spacing = np.flip(np.array(loaded[f"{keys[0]}_meta_dict"]["pixdim"][1:4]))  # (Z, Y, X)
       spacings.append(spacing) 
       
       # Iterate through keys of the same case ID, accumulate values 
       # for global mean and global std computation for normalization
       for key in keys_analyze:
           img = np.swapaxes(loaded[key], 0, -1)
           # Restrict image and brain mask to previously derived limits
           #trimmed_img = img[lims[0]:(lims[1]+1)] 
           #trimmed_mask = mask[lims[0]:(lims[1]+1)] 
           n_voxels[key] += (mask > 0).sum() 
           total_sum[key] += img[mask > 0].sum()
           total_sq_sum[key] += (img[mask > 0]**2).sum()  

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
       self.keys = keys
       super().__init__(keys)


   def __call__(self, data):
       d = dict(data)
       d[self.keys[0]] = torch.swapaxes(d[self.keys[0]], 1, -1)
       return d




def get_preprocessing_transforms(
   global_mean: float,
   global_std: float,
   out_folder : os.PathLike,
   key: str,
   median_spacing: tuple = (1.0, 1.0, 1.0),
):
   """
   Obtain preprocessing transforms

   Params
   ------
   global_mean : global mean value for normalization
   global_std : global standard deviation value for normalization
   out_folder : output folder
   key : image to be processed ("cta", "time", "dist")
   median_spacing : median spacing

   Returns
   -------
   transforms : preprocessing transforms
   
   """

   transforms = [
       LoadImaged(keys=[key]), # Load image
       EnsureChannelFirstd(keys=[key]), # Ensure axes are Z, Y, X
       EnsureZYXShapeD(keys=[key]), # Remove channel dimension
       GlobalNormalize(keys=[key], 
                       mean=global_mean, 
                       std=global_std), # Normalization
       Spacingd(keys=[key], 
                pixdim=median_spacing, 
                mode="bilinear"), # Image resampling
       ToNumpyd(keys=[key]), # Conversion to numpy array
       SaveAsBlosc2d(keys=[key], output_dir=out_folder), # Storage as .b2nd file
   ]


   return Compose(transforms)



def skip_processed(data: list, out_folder: os.PathLike, key : str = "cta") -> list:
   """
   Skip file if already processed


   Params
   ------
   data : data objects
   out_folder : output folder
   key : data type being analyzed ("cta", "time", "dist")

   Returns
   -------
   data_out : filtered dataset, only with non-processed cases


   """
   data_out = []
   suffix = "0000"

   if key.lower().strip() == "time":
       suffix = "time"
   elif key.lower().strip() == "dist":
       suffix = "dist"

   for d in data:
       outfile = os.path.join(out_folder, 
                              f"{d['cid']}_{suffix}.b2nd")
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


def extract_brain_limits(brain_seg : np.ndarray, cta : np.ndarray) -> list:
    """
    Locate first and last axial slices where the brain spans
    and there is distance information, since the 
    preprocessing will only focus on those slices

    Params
    ------
    brain_seg : input brain segmentation
    cta : CTA image

    Returns
    -------
    lims : brain limits
    
    """
    # Locate limits in brain segmentation
    brain_axial_coords = np.where(brain_seg > 0)[0]
    brain_lims = [brain_axial_coords.min(), 
                  brain_axial_coords.max()] 
    
    # Locate limits in distance maps
    cta_mean_axial = np.mean(cta, axis=(1,2))
    cta_axial_coords = np.where(cta_mean_axial > -1024)[0] 
    cta_lims = [cta_axial_coords.min(),
                 cta_axial_coords.max()]

    # Combine limit definitions
    lims = [max(brain_lims[0] , cta_lims[0]),
            min(brain_lims[-1] , cta_lims[-1])] 

    return lims  


def plan_architecture(c : dict, plan : dict, folder : os.PathLike, plan_file : os.PathLike, target_mem : float = 11.0) -> dict:
    """
    Plan patch size and architecture

    Params
    ------
    c : configuration for nnUNet v2
    plan : original preprocessing plan
    folder : preprocessing folder containing nnU-Net plan
    plan_file : filename for original plan

    Returns
    -------
    final_plan : updated plan with architecture information
    
    """
    # Set up runner
    runner = nnUNetV2Runner(c)

    # Run only the planning
    runner.plan_experiments(gpu_memory_target=target_mem)

    # Include patch size, batch size, 
    # and architectural details into main plan
    plan_nnunet_file = os.path.join(folder, "nnUNetPlans.json")
    assert os.path.exists(plan_nnunet_file), f"nnUNet plan file '{plan_nnunet_file}' does not exist"
    plan_nnunet = load_data(filename=plan_nnunet_file)
    patch_size = plan_nnunet["configurations"]["3d_fullres"]["patch_size"]
    batch_size = plan_nnunet["configurations"]["3d_fullres"]["batch_size"]
    arch = plan_nnunet["configurations"]["3d_fullres"]["architecture"]

    # Flip patch size
    patch_size[0], patch_size[1] = patch_size[1], patch_size[0]

    plan["patch_size"] = patch_size
    plan["batch_size"] = batch_size
    plan["architecture"] = arch

    write_data(data=plan, filename=plan_file)



def processing(global_mean : float, global_std : float, out_folder : os.PathLike, key : str, median_spacing : list, cfg : dict, data : dict):
   """
   Apply preprocessing with parameters

   Params
   ------
   global_mean : global mean for normalization
   global_std : global standard deviation for normalization
   out_folder : output folder
   key : image to be processed ('cta' for CTA, 'time' for temporal image, 
   'dist' for distance map)
   median_spacing : median spacing for image resampling
   cfg : final configuration
   data : dataset to be processed
   
   """

   # Derive transforms for each type of image 
   transforms = get_preprocessing_transforms(global_mean=global_mean,
                                             global_std=global_std,
                                             out_folder=out_folder,
                                             key=key,
                                             median_spacing=median_spacing)
   data = skip_processed(data=data, 
                         out_folder=out_folder,
                         key=key)
   dataset = Dataset(data=data, 
                     transform=transforms)
   dataloader = DataLoader(dataset, 
                           batch_size=1, 
                           num_workers=cfg["workers"])

   suffix = "0000"
   if key.lower().strip() == "time":
       suffix = "time"
   elif key.lower().strip() == "dist":
       suffix = "dist" 

   for batch in dataloader:
       # Skip files that have already been preprocessed
       outfile = os.path.join(out_folder, 
                              f"{batch['cid'][0]}_{suffix}.b2nd")
       if not (os.path.exists(outfile)):
           images = batch[key]
           logger.info(f"{batch['cid']} : Processed batch shape: {images.shape[2:]}, key: {key}")




def preprocess_dataset(args):

   # Load task argument
   task_id = args.task

   # Load configuration
   cfg_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cfg_translation.json")

   assert (
       os.path.exists(cfg_file) and ".json" in cfg_file
   ), f"Configuration file '{cfg_file}' does not exist or is not .json"
   cfg = load_data(cfg_file)


   # Determine folders
   raw_splitted = os.path.join(os.getenv("data_folder"), 
                               "raw_splitted", task_id)
   raw_cropped = os.path.join(os.getenv("data_folder"), 
                              "raw_cropped", task_id)
   preprocessed = os.path.join(os.getenv("data_folder"), 
                               "preprocessed", task_id)


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

   # Data cropping

   # Set up plan file, and load plan if it already exists
   plan_file = os.path.join(preprocessed, "plan_preprocess.json")
   plan = {}
   keys_analysis = ["cta", "time", "dist"]
   if os.path.exists(plan_file):
       # Load precomputed values from previous runs
       plan = load_data(plan_file)
       median_spacing = plan["spacing"]
       global_mean, global_std = {}, {}
       for keys in keys_analysis:
          global_mean[keys]  = plan[f"stats_{keys}"]["mean"]
          global_std[keys]  = plan[f"stats_{keys}"]["std"]

   # Create datalist
   datalist_file = os.path.join(preprocessed, "datalist.json") 
   if not(os.path.exists(datalist_file)):
      datalist = create_datalist(folder = raw_cropped)
      write_data(data=datalist, filename=datalist_file)

   # Create config.yaml
   config = create_configyaml(task_id = task_id)
   
   # Data preparation
   logger.info("Preparing data and cropping...")
   data = build_data_crop(folder=raw_splitted, 
                          task_id=task_id,
                          workers=cfg["workers"], 
                          key="Tr")
   data_test = build_data_crop(folder=raw_splitted, 
                               task_id=task_id,
                                workers=cfg["workers"], 
                                key="Ts")

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
       write_data(data=plan, path=plan_file)

   plan_keys = list(plan.keys())
   
   logger.info(f"✅ Median spacing (Z, Y, X): {median_spacing}")
   for keys in keys_analysis:
      logger.info(f"✅ Global mean {keys}, global std {keys}: {global_mean[keys]}, {global_std[keys]}")

   # Create dataset.json
   datasetjson = os.path.join(raw_cropped, 
                              "dataset.json")
   if not(os.path.exists(datasetjson)):
       create_datasetjson(task_id=task_id,
                          num_training=len(data),
                          outfile=datasetjson)  

   # Architecture planning (later for training)
   if ("batch_size" not in plan_keys) or ("patch_size" not in plan_keys) or ("architecture" not in plan_keys):
       plan_architecture(c=config, 
                        plan = plan, 
                        folder=preprocessed, 
                        plan_file=plan_file,
                        target_mem=cfg["gpu_mem"]) 
       
   # Apply processing
   for key in keys_analysis:
       out_folder = os.path.join(preprocessed, "imagesTr") if key.lower().strip() == "cta" else os.path.join(preprocessed, "labelsTr")
       processing(global_mean=global_mean[key],
                  global_std=global_std[key],
                  out_folder=out_folder,
                  key=key,
                  median_spacing=median_spacing,
                  cfg=cfg,
                  data=data) 


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", help="Task folder", type=str)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    t1 = time.time()
    preprocess_dataset(get_args())
    print(f"Time ellapsed: {time.time()-t1}sec")
