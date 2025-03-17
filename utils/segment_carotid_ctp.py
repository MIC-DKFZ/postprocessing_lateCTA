import os,sys
import numpy as np 
from totalsegmentator.python_api import totalsegmentator
import SimpleITK as sitk
import nibabel as nib
import xmltodict
from typing import Union

def sum_ctp(folder: os.PathLike, outfile : os.PathLike):
    """
    Sum all the CTP frames in a folder and save the file as .nii.gz

    Params
    ------
    folder : folder with CTP information

    """
    assert os.path.exists(folder), f"Folder '{folder}' does not exist"
    assert os.path.exists(os.path.dirname(outfile)), f"Parent folder of '{outfile}' does not exist"
    assert ".nii.gz" in outfile, f"Outfile '{outfile}' is not .nii.gz"

    # Load and iterate through each frame file
    files = os.listdir(folder)
    cont_img = 0
    for enum_file, file in enumerate(files):
        if ".nii.gz" in file:
            full_file = os.path.join(folder, file)
            cont_img += 1
            if enum_file == 0:
                image = sitk.ReadImage(full_file)
                img = sitk.GetArrayFromImage(image)
                sum_img = img.copy()
            else:
                sum_img += sitk.GetArrayFromImage(sitk.ReadImage(full_file))

    sum_img = (sum_img / float(cont_img)).astype(np.int16)
    sum_image = sitk.GetImageFromArray(sum_img)
    sum_image.CopyInformation(image)
    sitk.WriteImage(sum_image, outfile)


def segment_ica(img) -> np.ndarray:
    """
    Code to segment the Internal Carotid Artery with TotalSegmentator

    Params
    ------
    img : input image to be segmented

    Returns
    -------
    ica_segm : internal carotid artery segmentation

    """

    # Conduct segmentation
    output_img = totalsegmentator(img, task="headneck_bones_vessels", nr_thr_resamp=6, nr_thr_saving=6)
    segm = output_img.get_fdata()
    segm = np.swapaxes(segm, 0, -1)

    # Determine labels
    ext_header = output_img.header.extensions[0].get_content()
    ext_header = xmltodict.parse(ext_header)
    ext_header = ext_header["CaretExtension"]["VolumeInformation"]["LabelTable"][
        "Label"
    ]

    # If only one label, ext_header is a dict instead of a list (because of xmltodict.parse()) -> convert to list
    if isinstance(ext_header, dict):
        ext_header = [ext_header]

    labels = {e["#text"]: int(e["@Key"]) for e in ext_header}

    # Set up segmentations
    ica_right_segm = (segm == labels["internal_carotid_artery_right"]).astype(float)
    ica_left_segm = (segm == labels["internal_carotid_artery_left"]).astype(float)

    ica_segm = ((ica_left_segm + ica_right_segm) > 0).astype(float)
    return ica_segm



def segment_brain(img : np.ndarray) -> Union[np.ndarray, np.ndarray, np.ndarray]:
    """
    Segment brain, skull and common carotid artery from image with TotalSegmentator


    Params
    ------
    img : image to be segmented

    Returns
    -------
    brain_segm : brain segmentation
    skull_segm : skull segmentation
    cca_segm : common carotid artery segmentation
    

    """
    # Read image also in SimpleITK
    output_img = totalsegmentator(img)
    segm = output_img.get_fdata()
    
    segm = np.swapaxes(segm, 0, -1)
    brain_segm = (segm == 90).astype(np.int32)
    skull_segm = (segm == 91).astype(np.int32)
    cca_left_segm = (segm == 57).astype(np.int32)
    cca_right_segm = (segm == 58).astype(np.int32)
    cca_segm = (cca_left_segm + cca_right_segm).astype(np.int32)

    return brain_segm, skull_segm, cca_segm






folder = "/scratch/amartinezmora/raw_data/ctp/mrclean_late_30002"
#outfile = "/scratch/amartinezmora/preprocessed/ctp/mrclean_late_30002_sum.nii.gz"
outfile = "/scratch/amartinezmora/raw_data/ctp/mrclean_late_30002/mrclean_late_30002_t_20.nii.gz"
outfile_ica = "/scratch/amartinezmora/preprocessed/ctp/mrclean_late_30002_ica.nii.gz"
#sum_ctp(folder=folder, outfile=outfile)

if __name__ == '__main__':
    segment_ica(input_file=outfile, outfile = outfile_ica)