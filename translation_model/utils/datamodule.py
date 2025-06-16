import os,sys
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import pytorch_lightning as pl
import blosc2
from batchgenerators.transforms import (
    Compose, SpatialTransform, GaussianNoiseTransform, BrightnessMultiplicativeTransform,
    GammaTransform, MirrorTransform, ContrastAugmentationTransform, SimulateLowResolutionTransform
)
from batchgenerators.dataloading import MultiThreadedAugmenter
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase


def open_b2nd(file : os.PathLike):
    """
    Open preprocessed b2nd file

    Params
    ------
    file : b2nd file to be read

    Returns
    -------
    b2nd : loaded array
    
    """
    assert os.path.exists(file) and ".b2nd" in file, f".b2nd file '{file}' does not exist or is not b2nd"
    return blosc2.open(urlpath=file, mode="r", mmap_mode="r").astype(np.float32)


class CachedGuidedPatchDataset(Dataset):
    def __init__(
        self,
        image_paths : list,
        dist_paths : list,
        time_paths : list,
        cfg : dict, 
        patch_size : list
    ):
        """
        Args:
        image_paths : image filepaths
        dist_paths : distance filepaths
        time_paths : time filepaths
        cfg : hyperparameter configuration
        patch_size : patch size

        """
        self.image_paths = image_paths
        self.dist_paths = dist_paths
        self.time_paths = time_paths
        self.cfg = cfg
        self.patch_size = patch_size

        # Precompute coordinates for sampling
        self.vessel_coords, self.brain_coords = self.cache_vessel_coords()


    def cache_vessel_coords(self):
        """
        Obtain vessel and brain coordinates for guided patch sampling
        
        """
        vessel_coords, brain_coords = [], []
        for time_path, dist_path in zip(self.time_paths, self.dist_paths):
            
            # Load images
            time_img = open_b2nd(file = time_path)
            dist_img = open_b2nd(file = dist_path)

            ind_vessels = np.where(time_img > 0)
            ind_brain = np.where(dist_img < dist_img.max())

            # Prepare vessel indexes
            if ind_vessels.shape[0] > 0:
                ind_vessels = np.array(ind_vessels).T
            else:
                ind_vessels = np.array([])
            vessel_coords.append(ind_vessels)


            # Prepare brain indexes
            if ind_brain.shape[0] > 0:
                ind_brain = np.array(ind_brain).T
            else:
                ind_brain = np.array([])
            brain_coords.append(ind_brain)

        return vessel_coords, brain_coords

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # Path sampling
        image_path = self.image_paths[index]
        time_path = self.time_paths[index]
        dist_path = self.dist_paths[index]

        # Device 
        dev = "cuda" if torch.cuda.is_available() else "cpu"

        # Loading
        img = torch.tensor(open_b2nd(file = image_path), 
                           device=dev)
        t = torch.tensor(open_b2nd(file=time_path), 
                           device=dev)
        dist = torch.tensor(open_b2nd(file=dist_path), 
                           device=dev)
        
        # Obtain vessel and brain masks
        vessel_mask = t > 0
        brain_mask = dist < dist.max()

        # Extract patch center
        center = self._derive_center(index=index)

        # Extract patches
        img_patch = self._extract_patch(img, center)
        t_patch = self._extract_patch(t, center)
        dist_patch = self._extract_patch(dist, center)

        patch = np.stack([img_patch.cpu().numpy(),
                          dist_patch.cpu().numpy(),
                          t_patch.cpu().numpy(),
                          vessel_mask.cpu().numpy(),
                          brain_mask.cpu().numpy()])

        return {"data": patch}


    def _derive_center(self, index):

        # Guided sampling decision
        if np.random.rand() < self.cfg["foreground_ratio"]:
            # Vessel centered sampling
            center = self.vessel_coords[index].flatten()
        else:
            # Brain centered sampling
            center = self.brain_coords[index].flatten()

        return center

    def _extract_patch(self, volume, center):

        start = center - self.patch_size // 2
        end = start + self.patch_size

        # Clamp to valid range
        start = torch.clamp(start, min=0)
        end = torch.clamp(end, max=torch.tensor(volume.shape))

        slices = tuple(slice(int(s), int(e)) for s, e in zip(start, end))
        patch = volume[slices]

        # Padding if needed
        pad_needed = [max(0, ps - patch.shape[i]) for i, ps in enumerate(self.patch_size)]
        if any(pad_needed):
            padding = [(p // 2, p - p // 2) for p in pad_needed]
            patch = F.pad(patch, [x for p in reversed(padding) for x in p])

        return patch


class TripletDataLoader(SlimDataLoaderBase):
    def __init__(self, dataset, batch_size):
        super().__init__(dataset, batch_size, False, None)
        self.dataset = dataset

    def generate_train_batch(self):
        indices = np.random.choice(len(self.dataset), self.batch_size, replace=True)
        data = [self.dataset[i]["data"] for i in indices]
        batch = np.stack(data, axis=0)  # [B, C=5, D, H, W]
        return {"data": batch}
    


def get_batchgenerators_augmentation(cfg : dict, patch_size : tuple = (96, 192, 192)):
    """
    Obtain batchgenerators augmentations

    Params
    ------
    cfg : augmentatation hyperparameters
    patch_size : patch size
    
    """
    tr_transforms = Compose([
        SpatialTransform(
            patch_size,
            patch_center_dist_from_border=None,
            do_elastic_deform=bool(cfg["el_deform"]), 
            alpha=cfg["alpha_deform"], 
            sigma=cfg["sigma_deform"],
            do_rotation=bool(cfg["rot"]), 
            angle_x=(-cfg["rx"], cfg["rx"]), 
            angle_y=(-cfg["ry"], cfg["ry"]), 
            angle_z=(-cfg["rz"], cfg["rz"]),
            do_scale=bool(cfg["scale"]), 
            scale=cfg["s"], 
            border_mode_data='constant',
            border_cval_data=cfg["border_data"], 
            order_data=cfg["order_interp"], 
            random_crop=bool(cfg["random_crop"])
        ),
        MirrorTransform(axes=cfg["mirror_axes"]),
        #GaussianNoiseTransform(p_per_sample=cfg["p_noise"]),
        #BrightnessMultiplicativeTransform(multiplier_range=cfg["multiplier_bright"], 
        #                                  p_per_sample=cfg["p_bright"]),
        #ContrastAugmentationTransform(p_per_sample=cfg["p_contrast"]),
        #GammaTransform(gamma_range=cfg["gamma_range"], 
        #               p_per_sample=cfg["p_gamma"], 
        #               retain_stats=True),
        SimulateLowResolutionTransform(zoom_range=(cfg["zoom_lowres"], 1), 
                                       per_channel=True, 
                                       p_per_channel=cfg["p_lowres"], 
                                       order_downsample=0, 
                                       order_upsample=3, 
                                       p_per_sample=cfg["p_lowres_sample"])
    ])

    val_transforms = None  # No augmentation for validation
    return tr_transforms, val_transforms


class CTAPatchDataModule(pl.LightningDataModule):
    def __init__(self, train_cases : np.ndarray, val_cases : np.ndarray, cfg : dict, patch_size : tuple =(96, 192, 192), batch_size : int = 2):
        """
        Datamodule for CTA -> dist + time translation model

        Params
        ------
        train_cases : train image + distance + time files
        val_cases : validation image + distance + time files
        cfg : hyperparameter configuration
        patch_size : patch size

        """
        super().__init__()
        self.train_cases = train_cases
        self.val_cases = val_cases
        self.cfg = cfg
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.num_workers = cfg["workers"]


    def setup(self, stage=None):
        self.train_dataset = CachedGuidedPatchDataset(image_paths = self.train_cases[:,0],
                                                      dist_paths = self.train_cases[:,1],
                                                      time_paths = self.train_cases[:,2],
                                                      cfg = self.cfg,
                                                      patch_size = self.patch_size)
        self.val_dataset = CachedGuidedPatchDataset(image_paths = self.val_cases[:,0],
                                                    dist_paths = self.val_cases[:,1],
                                                    time_paths = self.val_cases[:,2],
                                                    cfg = self.cfg,
                                                    patch_size = self.patch_size)

        # Train loader wrapped with BatchGenerators
        base_loader = TripletDataLoader(self.train_dataset, batch_size=self.batch_size)
        transforms = get_batchgenerators_augmentation(cfg=self.cfg,
                                                      patch_size=self.patch_size)
        self.train_loader = MultiThreadedAugmenter(base_loader, 
                                                   transforms, 
                                                   num_processes=self.num_workers)


    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

