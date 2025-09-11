import os, sys
import numpy as np
import SimpleITK as sitk
import argparse
from joblib import Parallel, delayed
import time


def process_id(infile: os.PathLike, infolder: os.PathLike, outfolder: os.PathLike):
    cid = infile.replace("_0000.nii.gz", "")
    outfile = os.path.join(outfolder, f"{cid}.nii.gz")

    if not (os.path.exists(outfile)):
        print(cid)
        full_file = os.path.join(infolder, infile)

        image = sitk.ReadImage(full_file)
        img = sitk.GetArrayFromImage(image)

        mask = (img > -1024).astype(np.uint8)

        mask_image = sitk.GetImageFromArray(mask)
        mask_image.CopyInformation(image)

        sitk.WriteImage(mask_image, outfile)


def main(args):
    infolder = args.i
    outfolder = args.o
    workers = args.np

    assert os.path.exists(infolder), f"Input folder '{infolder}' does not exist"
    assert os.path.exists(
        os.path.dirname(outfolder)
    ), f"Output folder '{os.path.dirname(outfolder)}' does not exist"
    assert workers > 0, "Number of parallel workers is zero or negative"

    if not (os.path.exists(outfolder)):
        os.makedirs(outfolder)

    # Iterate through input files
    infiles = sorted(os.listdir(infolder))

    Parallel(n_jobs=workers)(
        delayed(process_id)(infile, infolder, outfolder)
        for infile in infiles
        if ".nii.gz" in infile
    )


def get_args():
    # Get brain cluster from images
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", help="Input data folder", required=True, type=str)
    parser.add_argument("--o", help="Output folder", required=True, type=str)
    parser.add_argument("--np", help="Parallel workers", default=4, type=int)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    t1 = time.time()
    main(get_args())
    print("Time ellapsed (seconds): ", time.time() - t1)
