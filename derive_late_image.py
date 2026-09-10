# SPDX-FileCopyrightText: Copyright 2026 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: Apache-2.0

import os,sys
import numpy as np
import SimpleITK as sitk
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.spatial import cKDTree
from scipy.ndimage import binary_dilation
from typing import Union
import math
import argparse
import time


def extract_late_vessels(time_img : np.ndarray) -> Union[np.ndarray, np.ndarray]:
    """
    Extract late vessels from time file

    Params
    ------
    time_img : time image 

    Returns
    -------
    late : late vessel segmentation
    
    """
    # Artery derivation: around 30th percentile of vessels 
    # with earliest time of arrival

    # Obtain vessel coordinates with a time of arrival > 0
    time_vals = time_img[time_img > 0]  
    arterial_time = np.percentile(time_vals, 30) # Threshold used to obtain the arterial tree

    artery_img = ((time_img > 0) & (time_img < arterial_time)).astype(np.float32)

    # Distinguish between veins and unusually lately enhanced vessels with a two-centroid clustering
    late_vals = time_img[time_img > arterial_time] 
    coords = np.vstack(np.where(time_img > arterial_time))
    features = np.vstack([late_vals.reshape(1, -1), coords]).T

    # Feature standardization 
    features_std = StandardScaler().fit_transform(features)  

    # Run clustering and predict
    k = 2
    gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=0)
    labels = gmm.fit_predict(features_std)
    
    gmm.fit(features_std)

    # Cluster assignment: the cluster with the highest 95th percentile value presents abnormally high vessels 
    cluster_img = np.zeros(time_img.shape)
    cluster_img[features[:,1].astype(int), features[:,2].astype(int), features[:,3].astype(int)] = labels + 1

    # Identify the label with lately enhanced vessels (the one with the highest arrival)
    max_values = [] # Get 95 percentile of the time values in each centroid 
    for i in range(k):
        label_mask = cluster_img == i + 1
        # perc = np.percentile(time_img[label_mask].flatten(), 95) 
        perc = time_img[label_mask].max()
        max_values.append(perc)

    label_late = np.argmax(np.array(max_values)) + 1
    late_img = (cluster_img == label_late).astype(np.float32) 

    return artery_img, late_img

def late2thrombus_distance(late_img : os.PathLike, thrombus_img : os.PathLike, spacing : tuple) -> Union[float, float]:
    """
    Derive late enhanced to thrombus distance,
    both in mm and voxels

    Params
    ------
    late_img : lately enhanced vessels
    thrombus_img : thrombus segmentation
    spacing : image spacing

    Returns
    -------
    spatial : spatial distance
    voxel : voxel distance
    
    """

    # Set an image of the late enhanced section together with thrombus information 
    late_thrombus_img = late_img + 2*thrombus_img
    late_coords, thrombus_coords = np.argwhere(late_thrombus_img == 1), np.argwhere(late_thrombus_img == 2)

    tree = cKDTree(late_coords)
    distances, _ = tree.query(thrombus_coords)

    # Minimum distance between label 1 and label 2
    voxel = distances.min()

    # Transform voxel distance into spatial distance with spacing information
    spatial = voxel*np.linalg.norm(np.array(spacing))

    return voxel, spatial



def generate_late_areas(late_segm : np.ndarray, spacing : tuple, dist : float) -> np.ndarray:
    """
    Generate area of influence of lately enhanced vessels

    Params
    ------
    late_segm : area of influence of lately enhanced vessels
    spacing : image spacing
    dist : distance of reference to be taken

    Returns
    -------
    influence_segm : segmentation with area of influence

    """
    # Obtain voxel diagonal size in space
    diag = np.linalg.norm(np.array(spacing))

    # Obtain number of dilation iterations to expand late segmentation area
    iter = int(math.ceil(dist / (diag + np.finfo(float).eps)))

    # Set up dilation structure
    dilate_struct = np.array([[[0,0,0],[0,1,0],[0,0,0]],
                              [[0,1,0],[1,1,1],[0,1,0]],
                              [[0,0,0],[0,1,0],[0,0,0]]]) 

    # Derive area of influence by dilating late vessel segmentation
    influence_segm = binary_dilation(late_segm.astype(bool), 
                                     structure=dilate_struct, 
                                     iterations=iter).astype(np.float32)

    return influence_segm
 

def main(args):
    # Derive area with late enhancement influence 
    tta_folder = args.tta
    thr_folder = args.thr
    out_folder = args.out

    assert os.path.exists(tta_folder), f"Time folder '{tta_folder}' does not exist"
    assert os.path.exists(thr_folder), f"Thrombus segmentation folder '{thr_folder}' does not exist"
    assert os.path.exists(os.path.dirname(out_folder)), f"Parent of output folder '{out_folder}' does not exist"

    if not(os.path.exists(out_folder)):
        os.makedirs(out_folder)

    # Iterate through case IDs 
    thrombus_files = sorted(os.listdir(thr_folder))
    dists = [] # Store distance values 
    late_masks = [] # Store late masks
    cids = [] # Store case IDs 
    spacings = [] # Store spacings       

    for f in thrombus_files:
        if ".nii.gz" in f:
            cid = f.replace(".nii.gz", "")
            file = os.path.join(tta_folder, f"{cid}.nii.gz") # Time file 
            thrombus_file = os.path.join(thr_folder, f) # Thrombus file  

            # Load time file
            time_image = sitk.ReadImage(file)
            time_img = sitk.GetArrayFromImage(time_image)
            spacing = np.array(time_image.GetSpacing())

            # Load thrombus image
            thrombus_image = sitk.ReadImage(thrombus_file)
            thrombus_img = sitk.GetArrayFromImage(thrombus_image) 

            # Derive lately enhanced vessels
            artery_img, late_img = extract_late_vessels(time_img = time_img) 

            # Save late segmentation
            # late_image = sitk.GetImageFromArray(late_img)
            # late_image.CopyInformation(time_image)
            # sitk.WriteImage(late_image, os.path.join(out_folder, f"{cid}.nii.gz"))

            # Load thrombus to late section distance 
            dist_voxel, dist_spatial = late2thrombus_distance(late_img=late_img, 
                                          thrombus_img=thrombus_img, 
                                          spacing=spacing)
            print(cid, dist_spatial)
            
            dists.append(dist_spatial)
            cids.append(cid)
            late_masks.append(late_img)
            spacings.append(spacing)


    # Set up 95 percentile of distances of thrombi to late vessel segmentations
    p95 = np.percentile(np.array(dists), 95)
    print(p95)
    for cid, late_mask, spacing in zip(cids, late_masks, spacings):
        # Load time file
        file = os.path.join(tta_folder, f"{cid}.nii.gz") # Time file 
        time_image = sitk.ReadImage(file)

        # Generate area of influence of lately enhanced vessels
        outfile = os.path.join(out_folder, f"{cid}.nii.gz")
        influence_segm = generate_late_areas(late_segm=late_mask, spacing = spacing, dist=p95)
        influence_segm_image = sitk.GetImageFromArray(influence_segm)
        influence_segm_image.CopyInformation(time_image)
        sitk.WriteImage(influence_segm_image, outfile)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tta", help="Folder with preprocessed TTA data", type=str)
    parser.add_argument("--thr", help="Folder with preprocessed thrombus data", type=str)
    parser.add_argument("--out", help="Output folder with lately enhanced areas", type=str)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    t1 = time.time()
    main(get_args())
    print(f"Time ellapsed: {time.time()-t1}sec")