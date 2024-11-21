import SimpleITK as sitk
import os,sys
import numpy as np


def npy2nii(npy : np.ndarray, image : sitk.Image, out : os.PathLike):
    """
    Transform npy file into nii.gz

    Params
    ------
    npy : array to save as nii.gz
    image : sitk image used as reference for spacing and origin
    out : filename of 4D NIFTI file

    Returns
    -------
    Saved info from npy file as output file
    
    """
    frames = []
    for i in range(npy.shape[0]):   
        frame_image = sitk.GetImageFromArray(npy[i])
        frame_image.CopyInformation(image)
        frames.append(frame_image)
    series = sitk.JoinSeries(frames)

    
    sitk.WriteImage(series, out)

if __name__ == "__main__":
    file = "/scratch/amartinezmora/code/smoothed.npy"
    image_file = "/scratch/amartinezmora/raw_data/ctp/mrclean_late_30002/mrclean_late_30002_t_00.nii.gz"
    out = "/scratch/amartinezmora/code/smoothed.nii.gz"

    image = sitk.ReadImage(image_file)

    img = np.load(file)
    npy2nii(npy=img, image=image, out=out)
    

