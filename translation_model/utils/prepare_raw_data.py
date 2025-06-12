import os,sys
import shutil
import numpy as np
import SimpleITK as sitk
import argparse
import time
from scipy.stats import rankdata



def time2rank(time_img : np.ndarray) -> np.ndarray:
    """
    Convert time to rank image

    Params
    ------
    time_img : time vessel image

    Returns
    -------
    rank : rank image
    
    """
    mask = time_img > 0
    foreground_vals = time_img[mask].flatten()
    ranks = rankdata(foreground_vals, 
                     method='average')
    percentiles = (ranks - 1) / (len(foreground_vals) - 1)
    rank = np.zeros(time_img.shape, 
                    dtype=np.float32)
    rank[mask] = percentiles

    return rank
    

def main(args):
    cta_folder = args.c
    dist_folder = args.d
    time_folder = args.t
    brain_folder = args.b
    out_folder = args.o
    split = args.s

    assert os.path.exists(cta_folder), f"CTA folder '{cta_folder}' does not exist"
    assert os.path.exists(dist_folder), f"Distance folder '{dist_folder}' does not exist"
    assert os.path.exists(time_folder), f"Time folder '{time_folder}' does not exist"
    assert os.path.exists(os.path.dirname(out_folder)), f"Parent output folder '{os.path.dirname(out_folder)}' does not exist"

    if brain_folder is not None:
        assert os.path.exists(brain_folder), f"Brain folder '{brain_folder}' does not exist"
        # Create output brain folder
        brain_out_folder = os.path.join(out_folder, "raw_splitted", "brainTr")
        brain_out_test_folder = os.path.join(out_folder, "raw_splitted", "brainTs")
        if not(os.path.exists(brain_out_folder)):
            os.makedirs(brain_out_folder)
        if not(os.path.exists(brain_out_test_folder)):
            os.makedirs(brain_out_test_folder)

    # Obtain CIDs
    cta_cids = np.array([f.replace(".nii.gz", "") for f in sorted(os.listdir(cta_folder))], dtype=str) 
    dist_cids = np.array([f.replace(".nii.gz", "") for f in sorted(os.listdir(dist_folder))], dtype=str) 
    time_cids = np.array([f.replace(".nii.gz", "") for f in sorted(os.listdir(time_folder))], dtype=str) 
    common_cids = np.intersect1d(cta_cids, dist_cids)
    common_cids = np.intersect1d(common_cids, time_cids)

    # Set up output folders
    train_folder = os.path.join(out_folder, "raw_splitted", "imagesTr") 
    label_folder = os.path.join(out_folder, "raw_splitted", "labelsTr") 
    test_folder = os.path.join(out_folder, "raw_splitted", "imagesTs") 
    label_test_folder = os.path.join(out_folder, "raw_splitted", "labelsTs") 

    if not(os.path.exists(train_folder)):
        os.makedirs(train_folder)

    if not(os.path.exists(label_folder)):
        os.makedirs(label_folder)

    if not(os.path.exists(test_folder)):
        os.makedirs(test_folder)

    if not(os.path.exists(label_test_folder)):
        os.makedirs(label_test_folder)

    # By default: apply a random train-test split
    # TODO: stratify for phase 
    sample_size = int(split*common_cids.shape[0])
    np.random.seed(42)
    test_cids = np.random.choice(common_cids, 
                                 size=sample_size, 
                                 replace=False)

    # Iterate through IDs
    for i in common_cids:
        # Retrieve files
        # CTA file
        cta_file = os.path.join(cta_folder, f"{i}.nii.gz")

        # Brain file
        brain_file = os.path.join(brain_folder, f"{i}_brain.nii.gz") if brain_folder is not None else None

        # Set up output filenames
        out_dist_file = os.path.join(label_folder, f"{i}_dist.nii.gz")
        out_cta_file = os.path.join(train_folder, f"{i}.nii.gz")  
        out_time_file = os.path.join(label_folder, f"{i}_time.nii.gz")
        if os.path.exists(brain_file):
            out_brain_file = os.path.join(brain_out_folder, f"{i}.nii.gz")
        if i in test_cids:
            print(i, "test")
            out_dist_file = os.path.join(label_test_folder, f"{i}_dist.nii.gz")
            out_cta_file = os.path.join(test_folder, f"{i}.nii.gz")  
            out_time_file = os.path.join(label_test_folder, f"{i}_time.nii.gz")
            out_brain_file = os.path.join(brain_out_test_folder, f"{i}.nii.gz")
        else:
            print(i, "train") 

        # Identify inferior and superior rows where to apply trimming for distance map
        # Avoid having distance information in slices where there is no CTA information 
        cta_image = sitk.ReadImage(cta_file)
        cta_img = sitk.GetArrayFromImage(cta_image)
        cta_img[cta_img == -1024] = 0
        sum_rows = np.sum(cta_img, axis=(1,2))
        ind_rows = np.where(sum_rows == 0)[0]
        low_lim, high_lim = 0, cta_img.shape[0]-1
        if ind_rows.shape[0] > 0:  
            lower_rows = ind_rows[ind_rows < cta_img.shape[0]//2]  
            upper_rows = ind_rows[ind_rows > cta_img.shape[0]//2] 
            low_lim, high_lim = lower_rows.max(), upper_rows.min()


        # Time file
        time_file = os.path.join(time_folder, f"{i}.nii.gz")

        # Distance file: trim it with CTA information
        if not(os.path.exists(out_dist_file)):
            dist_file = os.path.join(dist_folder, f"{i}.nii.gz") 
            dist_image = sitk.ReadImage(dist_file)
            dist_img = sitk.GetArrayFromImage(dist_image)
            dist_img[:low_lim] = dist_img.max()
            dist_img[high_lim:] = dist_img.max()
            dist_image_trim = sitk.GetImageFromArray(dist_img)
            dist_image_trim.CopyInformation(dist_image)
            sitk.WriteImage(dist_image_trim, out_dist_file)  

        # Copy CTA file 
        if not(os.path.exists(out_cta_file)): 
            shutil.copyfile(cta_file, out_cta_file)

        # Copy brain file, if it exists
        if (brain_file is not None):
            if not(os.path.exists(out_brain_file)):
                shutil.copyfile(brain_file, out_brain_file)

        # Convert time image into rank image
        if not(os.path.exists(out_time_file)):
            time_image = sitk.ReadImage(time_file)
            time_img = sitk.GetArrayFromImage(time_image) 
            rank_img = time2rank(time_img=time_img)
            rank_image = sitk.GetImageFromArray(rank_img.astype(np.float32))
            rank_image.CopyInformation(time_image)
            sitk.WriteImage(rank_image, out_time_file)


def get_args():
    # Remove predictions outside of lately enhanced regions 
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", help="CTA folder", required=True, type=str)
    parser.add_argument("--t", help="Time map folder", required=True, type=str)
    parser.add_argument("--d", help="Distance map folder", required=True, type=str)
    parser.add_argument("--b", help="Brain segmentation folder", default=None, type=str)
    parser.add_argument("--o", help="Output folder", required=True, type=str)
    parser.add_argument("--s", help="Train-test split ratio", default=0.2, type=float)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    t1 = time.time()
    main(get_args())
    print(f"Time ellapsed: {time.time()-t1}sec")