import os, sys
import numpy as np
import SimpleITK as sitk
from joblib import Parallel, delayed
import pandas as pd
from batchgenerators.utilities.file_and_folder_operations import load_json
import argparse
import time
import random


def process_file(file, folder, key, df=None):

    cid = file.replace(".nii.gz", "")
    str_cid = cid.replace(f"{key}_", "")
    num_cid = int(str_cid)

    full_file = os.path.join(folder, file)

    # Extract spacing and matrix size information
    image = sitk.ReadImage(full_file)
    img = sitk.GetArrayFromImage(image)
    spacing = np.flip(np.array(image.GetSpacing()))
    matrix = np.array(img.shape, dtype=int)

    # Extract sex and age information
    if key != "ukb":
        manufacturer = "siemens"

        # Derive record to analyze
        df_info = df.loc[num_cid]

        age = int(df_info["age_calculated"])
        sex = str(df_info["sex"]).lower()
        sex = "w" if sex == "f" else "m"

        if key == "STROKE":
            loc = "Heidelberg"
            kernel = ""
        else:
            loc = str(df_info["Location"])
            kernel = str(df_info["Kernel"])

    else:
        manufacturer = "Philips"
        age = 77
        rnd = random.randrange(0, 1)
        sex = "m" if rnd <= 0.54 else "w"
        loc = "Bonn"
        kernel = "B"

    # Prepare final dictionary with information
    print(cid)
    return {
        cid: {
            "manufacturer": manufacturer,
            "sex": sex,
            "age": age,
            "location": loc,
            "kernel": kernel,
            "spacing_x": spacing[0],
            "spacing_y": spacing[1],
            "spacing_z": spacing[2],
            "matrix_x": matrix[0],
            "matrix_y": matrix[1],
            "matrix_z": matrix[2],
        }
    }


def main(args):
    infile = args.i
    outfile = args.o
    workers = args.np

    assert os.path.exists(infile) and infile.endswith(
        ".json"
    ), f"Input file '{infile}' does not exist or is not .json"
    assert os.path.exists(os.path.dirname(outfile)) and outfile.endswith(
        ".csv"
    ), f"Parent directory of output file '{os.path.dirname(outfile)}' does not exist or is not .csv"
    assert workers > 0, "Zero or negative number of parallel workers"

    # Load information
    info = load_json(infile)
    keys = list(info.keys())

    all_metas = []

    for key in keys:
        folder = info[key]["img_folder"]
        ref_folder = info[key]["ref_folder"]
        files = sorted(os.listdir(folder))
        metafile = info[key]["metafile"]
        df = None
        if os.path.exists(metafile):
            df = pd.read_excel(metafile)
            df = df.set_index("record_id")
        metas = Parallel(n_jobs=workers)(
            delayed(process_file)(file, ref_folder, key, df)
            for file in files
            if key in file and file.endswith(".nii.gz")
        )
        all_metas += metas

    out = {}
    for meta in all_metas:
        meta_key = list(meta.keys())[0]
        out[meta_key] = meta[meta_key]

    # Build up final CSV file
    out = pd.DataFrame.from_dict(out, orient="index")
    out.to_csv(outfile)


def get_args():
    # Gather metadata
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", help="Input file", type=str)
    parser.add_argument("--o", help="Output file", type=str, default="")
    parser.add_argument("--np", help="Workers", type=int, default=4)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    t1 = time.time()
    main(get_args())
    print(f"Time ellapsed: {time.time()-t1}sec")
