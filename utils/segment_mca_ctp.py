from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import torch
import numpy as np
import SimpleITK as sitk
import os




class predictionAlgorithm:
   def __init__(self, train_dir : os.PathLike):
       self.predictor = nnUNetPredictor(
           tile_step_size=0.5,
           use_gaussian=True,
           use_mirroring=True,
           perform_everything_on_device=True,
           device=torch.device("cuda", 0),
           verbose=False,
           verbose_preprocessing=False,
           allow_tqdm=True,
       )
       self.predictor.initialize_from_trained_model_folder(
           train_dir,
           use_folds=(0, 1, 2, 3, 4),
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




if __name__ == "__main__":
   train_dir = "/scratch/amartinezmora/det_models/Dataset870_TopCoW24/nnUNetTrainerSkeletonRecallNoMirroring__nnUNetPlans__3d_fullres"
   img_file = "/scratch/amartinezmora/preprocessed/ctp/mrclean_late_30002_sum.nii.gz"
   segm_file = "/scratch/amartinezmora/preprocessed/ctp/mrclean_late_30002_mca.nii.gz"
   #prob_file = "/home/a870a/0024_0000_prob_topcow"


   img = sitk.ReadImage(img_file)
   img_array = sitk.GetArrayFromImage(img)
   out, probs = predictionAlgorithm(train_dir=train_dir).predict(image_ct=img)
   #print(np.unique(out))
   out1 = (out > 4).astype(float)
   out2 = (out < 8).astype(float)
   out_mca = (out1*out2).astype(float)
   outimg = sitk.GetImageFromArray(out_mca.astype(float))
   outimg.CopyInformation(img)
   sitk.WriteImage(outimg, segm_file)


