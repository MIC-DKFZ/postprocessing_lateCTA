import os,sys
import numpy as np
import SimpleITK as sitk
import argparse
from typing import Union



def analyze_case(cta_file : os.PathLike, ctp_folder : os.PathLike) -> Union[float, float, int, int]:
    """
    Analyze case of interest, extract lowest correlation of CTP frame vs CTA, 
    and also highest, and their indexes

    Params
    ------
    cta_file : CTA scan file to analyze
    ctp_folder : folder with CTP scan data

    Returns
    -------
    r_max : maximum correlation found
    r_min : minimum correlation found
    ctp_file_max : CTP frame with maximum index of correlation
    ctp_file_min : CTP frame with minimum index of correlation
    
    """
    cta_scan = sitk.GetArrayFromImage(sitk.ReadImage(cta_file))
    # Locate corresponding CTP frames
    cid = os.path.basename(cta_file).replace(".nii.gz", "") 
    ctp_subfolder = os.path.join(ctp_folder, cid)
    ctp_files = np.array(sorted(os.listdir(ctp_subfolder)), dtype=str)
    rs = [] 
    for ctp_file in ctp_files:
        if ".nii.gz" in ctp_file:
            full_ctp_file = os.path.join(ctp_subfolder, ctp_file)
            ctp_scan = sitk.GetArrayFromImage(sitk.ReadImage(full_ctp_file))
            r = compare_files(cta_scan = cta_scan, ctp_scan=ctp_scan)
            rs.append(r)

    rs = np.array(rs)

    ind_max, ind_min = np.argmax(rs),np.argmin(rs)
    r_max, r_min = rs[ind_max], rs[ind_min] 
    frame_max, frame_min = ctp_files[ind_max],ctp_files[ind_min]
    frame_max = int(frame_max.split(".")[0].split("_")[-1]) 
    frame_min = int(frame_min.split(".")[0].split("_")[-1])   

    return r_max, r_min, frame_max, frame_min


def compare_files(cta_scan : np.ndarray, ctp_scan : np.ndarray):
    """
    Compare registered CTP frame to reference CTA scan, by means
    of correlation

    Params
    ------
    cta_scan : CTA scan
    ctp_scan : frame of CTP scan

    Returns
    -------
    r : correlation coefficient
    
    """
    # Consider only non-air and non-NaN CTA and CTP voxels
    cta_scan = np.nan_to_num(cta_scan, nan=-1024)
    ctp_scan = np.nan_to_num(ctp_scan, nan=-1024)
    mask_cta = (cta_scan > -1024)
    mask_ctp = (ctp_scan > -1024) 
    mask = mask_ctp*mask_cta

    cta_vals, ctp_vals = cta_scan[mask],ctp_scan[mask] 
    r = np.corrcoef(cta_vals, ctp_vals)[1,0] 
    return r



def main(args):
    # Load arguments
    ctp_folder = args.ctp
    cta_folder = args.cta
    out = args.out

    assert os.path.exists(ctp_folder) and os.path.exists(cta_folder), f"Folder with CTA data '{cta_folder}' or with CTP data '{ctp_folder}' does not exist"
    assert os.path.exists(os.path.dirname(out)), f"Parent folder of output file '{out}' does not exist"

    # Iterate through cases in CTA folder
    cta_files = sorted(os.listdir(cta_folder))
    for cta_file in cta_files:
        if ".nii.gz" in cta_file:
            full_cta_file = os.path.join(cta_folder, cta_file)
            cid = cta_file.replace(".nii.gz", "") 
            ctp_subfolder = os.path.join(ctp_folder, cid)
            if os.path.exists(ctp_subfolder):
                r_max, r_min, ind_max, ind_min = analyze_case(cta_file=full_cta_file, ctp_folder=ctp_folder)
                with open(out, "a") as f:
                    f.write(f"{cid},{r_max},{r_min},{ind_max},{ind_min}\n")
                f.close()

    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctp", help="Folder with CTP data", required=True, type=str)
    parser.add_argument("--cta", help="Folder with CTA data", required=True, type=str)
    parser.add_argument("--out", help="Output file", required=True, type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(get_args())