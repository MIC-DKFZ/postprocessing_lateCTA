import SimpleITK as sitk
import numpy as np
import os,sys
from registration.setParameters import set_parameters
from registration.registration_utils import get_transformation_matrix,apply_transformation
import shutil



if __name__ == "__main__":
    ctp_folder = "/scratch/amartinezmora/raw_data/ctp/mrclean_late_30002"
    atlas_file = "/scratch/amartinezmora/raw_data/cta/mrclean_late_30002.nii.gz"
    out_folder = "/scratch/amartinezmora/raw_data/cta_registered_to_ctp"

    assert os.path.exists(ctp_folder) and os.path.exists(atlas_file), f"CTP folder '{ctp_folder}' or atlas file '{atlas_file}' do not exist"

    if not(os.path.exists(out_folder)):
        os.makedirs(out_folder)

    #cid = os.path.basename(ctp_folder)
    #out_folder = os.path.join(out_folder, cid)
    #if os.path.exists(out_folder):
    #    shutil.rmtree(out_folder)

    # Copy folder into destination, to do all the registration stuff in there
    #shutil.copytree(ctp_folder, out_folder)
    #outfile = os.path.join(out_folder, os.path.basename(atlas_file))
    #shutil.copy(atlas_file, out_folder)

    # Load first frame of CTP and atlas
    ctp_frame_files = sorted(os.listdir(ctp_folder))
    ctp_frame_files = [os.path.join(ctp_folder, f) for f in ctp_frame_files if ".nii.gz" in f]
    t0_file = ctp_frame_files[0]
    t0_image = sitk.ReadImage(t0_file)
    atlas_image = sitk.ReadImage(atlas_file)

    # Set parameters
    p = set_parameters(method='euler', DefaultPixelValue=-1024, metric='AdvancedNormalizedCorrelation')
    # Derive transformation matrix
    transform_matrix = get_transformation_matrix(fixed=t0_image, moving=atlas_image, clipvalue=[None, None], parameters=p)

    atlas_image = apply_transformation(transform_matrix, moving=atlas_image, segmentation=False, default_pixel=-1024)
    outfile = os.path.join(out_folder, os.path.basename(atlas_file))
    sitk.WriteImage(atlas_image, outfile)

    # For each CTP frame, register it to the atlas
    """
    for frame in ctp_frame_files:
        frame_image = sitk.ReadImage(frame)
        frame_image = apply_transformation(transform_matrix, moving=frame_image, segmentation=False, default_pixel=-1024)
        outfile = os.path.join(out_folder, frame)
        sitk.WriteImage(frame_image, outfile)
    """
