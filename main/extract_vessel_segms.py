import os, sys
import numpy as np
import SimpleITK as sitk
import time
import argparse
from typing import Union
from joblib import Parallel, delayed


def extreme_slices(segmentation: np.ndarray) -> Union[int, int]:
    """
    Derive extreme slices from brain segmentation

    Params
    ------
    segmentation : brain segmentation

    Returns
    -------
    start_slice : low axial limit
    high_lim : high axial limit

    """
    # Find the indices along the z-axis that have non-zero values
    non_empty_slices = np.any(segmentation, axis=(1, 2))

    # Get the first and last slice index where there is brain tissue
    start_slice = np.argmax(non_empty_slices)
    end_slice = len(non_empty_slices) - 1 - np.argmax(non_empty_slices[::-1])

    return start_slice, end_slice


def iterate(infolder, segfolder, outfolder, brainfolder, file):
    cid = file.replace("_0000.nii.gz", "")
    full_file = os.path.join(infolder, file)
    segfile = os.path.join(segfolder, f"{cid}_0001.nii.gz")
    brainfile = os.path.join(brainfolder, f"{cid}.nii.gz")
    outfile = os.path.join(outfolder, f"{cid}_0001.nii.gz")
    try:
        if not (os.path.exists(outfile)):
            # Load brain segmentation and inferior and superior extremes
            brain_image = sitk.ReadImage(brainfile)
            brainsegm = sitk.GetArrayFromImage(brain_image)
            extremes = extreme_slices(segmentation=brainsegm)

            # Load vessel segmentation
            vessel_image = sitk.ReadImage(segfile)
            vesselsegm = sitk.GetArrayFromImage(vessel_image)
            vesselsegm_bin = (vesselsegm > 0).astype(np.uint8)
            vesselsegm_bin = vesselsegm_bin[extremes[0] : extremes[-1]]

            # Load main image
            image = sitk.ReadImage(full_file)
            outimage = sitk.GetImageFromArray(vesselsegm_bin)
            outimage.CopyInformation(image)

            print(cid)
            sitk.WriteImage(outimage, outfile)

    except:
        print(f"Vessel segmentation file '{segfile}' not found")
        pass


def main(args):
    infolder = args.i
    segfolder = args.s
    brain_folder = args.b
    outfolder = args.o
    workers = args.np

    assert os.path.exists(infolder), f"Input folder '{infolder}' does not exist"
    assert os.path.exists(
        segfolder
    ), f"Segmentation folder '{segfolder}' does not exist"
    assert os.path.exists(brain_folder), f"Brain folder '{brain_folder}' does not exist"
    assert os.path.exists(
        os.path.dirname(outfolder)
    ), f"Parent output directory '{os.path.dirname(outfolder)}' does not exist"

    files = sorted(os.listdir(infolder))
    Parallel(n_jobs=workers)(
        delayed(iterate)(infolder, segfolder, outfolder, brain_folder, file)
        for file in files
        if file.endswith(".nii.gz")
    )


def get_args():
    parser = argparse.ArgumentParser(
        description="Gather vessel segmentations for time-vessel map experiment baselines"
    )
    parser.add_argument("--i", help="Input folder", required=True, type=str)
    parser.add_argument("--s", help="Segmentation folder", required=True, type=str)
    parser.add_argument("--b", help="Brain folder", required=True, type=str)
    parser.add_argument(
        "--o", help="Output folder", required=True, default=None, type=str
    )
    parser.add_argument("--np", help="Workers", required=False, default=4, type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    t1 = time.time()
    main(get_args())
    print(f"Time ellapsed: {time.time()-t1}sec")
