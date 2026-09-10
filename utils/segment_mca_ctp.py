# SPDX-FileCopyrightText: Copyright 2026 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: Apache-2.0

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import torch
import numpy as np
import SimpleITK as sitk
import os




class predictionAlgorithm:
   def __init__(self, train_dir : os.PathLike, device = torch.device("cuda", 0), folds : tuple = (0,1,2,3,4), tile_step_size : float = 0.5, use_gaussian : bool = True, use_mirroring : bool = True):
       self.predictor = nnUNetPredictor(
           tile_step_size=tile_step_size,
           use_gaussian=use_gaussian,
           use_mirroring=use_mirroring,
           perform_everything_on_device=True,
           device=device,
           verbose=False,
           verbose_preprocessing=False,
           allow_tqdm=True,
       )
       self.predictor.initialize_from_trained_model_folder(
           train_dir,
           use_folds=folds,
           checkpoint_name="checkpoint_final.pth",
       )


   def predict(self, image_ct) -> np.ndarray:
       input_array = sitk.GetArrayFromImage(image_ct)

       spacing = image_ct.GetSpacing()
       # 3d, as in original nnunet
       input_array = input_array[None]
       spacing_for_nnunet = list(spacing)[::-1]
       props = {
           # the spacing is inverted with [::-1] because sitk returns the spacing in the wrong order lol. Image arrays
           # are returned x,y,z but spacing is returned z,y,x. Duh.
           "spacing": spacing_for_nnunet
       }


       ret, probs = self.predictor.predict_single_npy_array(
           input_image=input_array,
           image_properties=props,
           segmentation_previous_stage=None,
           output_file_truncated=None,
           save_or_return_probabilities=True,
       )


       return ret, probs


