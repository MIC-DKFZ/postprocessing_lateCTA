# SPDX-FileCopyrightText: Copyright 2026 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: Apache-2.0

import os, sys
import numpy as np
import time
import argparse
import SimpleITK as sitk
from typing import Union
import shutil
import pandas as pd
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


def save_image(arr: np.ndarray, ref_image, outfile: os.PathLike):
    """
    Save image

    Params
    ------
    arr : array to be saved
    ref_image : reference SimpleITK image
    outfile : output file

    """
    out_image = sitk.GetImageFromArray(arr)
    out_image.SetSpacing(ref_image.GetSpacing())
    out_image.SetDirection(ref_image.GetDirection())
    out_image.SetOrigin(ref_image.GetOrigin())
    sitk.WriteImage(out_image, outfile)


def process_id(
    i: str,
    img_folder: os.PathLike,
    label_folder: os.PathLike,
    out_img_folder: os.PathLike,
    out_label_folder: os.PathLike,
    out_brain_folder: os.PathLike,
    brain_folder: os.PathLike,
    df_label: pd.DataFrame,
    df_phase: pd.DataFrame,
    tag: str,
    cohort: str,
):
    """
    Process ID by loading original image and label
    - Ignore IDs with extracranial labels
    - Constrain image to brain
    - Ignore arterial IDs from external cohorts

    Params
    ------
    i : input file
    img_folder : input image folder
    label_folder : input label folder
    out_img_folder : output image folder
    out_label_folder : output label folder
    out_brain_folder : output brain folder
    brain_folder : TotalSegmentator brain folder
    df_label : label information
    df_phase : phase information
    tag : tag to be added to each case ID
    cohort : cohort under analysis

    """
    # Derive case ID
    cid = i.replace("_0000.nii.gz", "")
    out_cid = f"{tag}{cid}"
    cid_zeros = ("0000" + cid.replace("STROKE_", ""))[(-4):]
    cid_num = str(int(cid_zeros))

    # derive output file
    outfile_img = os.path.join(out_img_folder, f"{cid}_0000.nii.gz")
    outfile_label = os.path.join(out_label_folder, f"{cid}.nii.gz")
    outfile_brain = os.path.join(out_brain_folder, f"{cid}.nii.gz")

    label_ind = df_label.index.values.astype(str).tolist()
    phase_ind = df_phase.index.values.astype(str).tolist()

    if (
        not (os.path.exists(outfile_img))
        or not (os.path.exists(outfile_label))
        or not (os.path.exists(outfile_brain))
    ):
        # Skip already processed cases
        # print(out_cid)
        # Check if case contains extracranial label
        extracranial = [
            "Carotis_T_R",
            "Carotis_T_L",
            "ACC_R",
            "ACC_L",
            "ACI_BIFURK_R",
            "ACI_BIFURK_L",
            "ACI_R",
            "ACI_L",
            "V13_R",
            "V13_L",
            "V4_R",
            "V4_L",
        ]

        if cid_num in label_ind:
            df_extracranial = df_label.loc[int(cid_num)]
            df_extracranial = df_extracranial[extracranial].values
            extracranial_label = df_extracranial.sum()

            if not (extracranial_label):
                skip = False
                if (cohort == "ukb") or (cohort == "fast"):
                    # Determine phase
                    if cid_num in phase_ind:
                        phase_info = df_phase.loc[int(cid_num)]
                        phase = phase_info["Phase"]
                        # if "arterial" in phase:
                        #    skip = True
                        #    print(
                        #        f"External CID in arterial phase: {out_cid}, skipping..."
                        #    )
                    else:
                        skip = True

                if not (skip):
                    # Derive image and label files
                    input_file = os.path.join(img_folder, i)
                    label_file = os.path.join(label_folder, f"{cid}.nii.gz")
                    brain_file = os.path.join(brain_folder, f"{out_cid}.nii.gz")

                    if not (os.path.exists(brain_file)):
                        brain_file = os.path.join(brain_folder, f"{cid}.nii.gz")

                    # Derive images
                    input_image = sitk.ReadImage(input_file)
                    input_img = sitk.GetArrayFromImage(input_image)
                    label_img = sitk.GetArrayFromImage(sitk.ReadImage(label_file))

                    if os.path.exists(brain_file):
                        brain_img = sitk.GetArrayFromImage(sitk.ReadImage(brain_file))
                        if brain_img.sum() > 0:
                            low_lim, high_lim = extreme_slices(segmentation=brain_img)
                            # Clip volumes to brain slices
                            out_img = input_img[low_lim:high_lim]
                            out_label = label_img[low_lim:high_lim]
                            out_brain = brain_img[low_lim:high_lim]
                        else:
                            out_img = input_img.copy()
                            out_label = label_img.copy()
                            out_brain = brain_img.copy()
                    else:
                        print(f"Non-existing brain segmentation for {out_cid}")
                        out_img = input_img.copy()
                        out_label = label_img.copy()
                        out_brain = brain_img.copy()

                    # Save output images and labels
                    save_image(arr=out_img, ref_image=input_image, outfile=outfile_img)
                    save_image(
                        arr=out_label, ref_image=input_image, outfile=outfile_label
                    )
                    save_image(
                        arr=out_brain, ref_image=input_image, outfile=outfile_brain
                    )
                    # Save .json label files
                    json_file = os.path.join(label_folder, f"{cid}.json")
                    out_json_file = os.path.join(out_label_folder, f"{cid}.json")
                    shutil.copyfile(json_file, out_json_file)

            else:
                print(f"{cid} presents an extracranial label, skipping...")


def process_cohort(
    cohort: str,
    df_info: pd.DataFrame,
    brain_folder: os.PathLike,
    df_phase: pd.DataFrame,
    outfolder: os.PathLike,
    workers: int = 4,
):
    """
    Process cohort files

    Params
    ------
    cohort : cohort under study
    df_info : information on label file, images, and label files for the given cohort
    brain_folder : brain folder
    df_phase : phase information file
    outfolder : output folder

    """
    if cohort == "STROKE":
        # Save results in imagesTr and labelsTr
        out_img_folder = os.path.join(outfolder, "raw_splitted", "imagesTr")
        out_label_folder = os.path.join(outfolder, "raw_splitted", "labelsTr")
        out_brain_folder = os.path.join(outfolder, "raw_splitted", "brainTr")
        tag = ""  # Tag to add to each case processed
    else:
        # Save results in imagesTs and labelsTs
        out_img_folder = os.path.join(outfolder, "raw_splitted", "imagesTs")
        out_label_folder = os.path.join(outfolder, "raw_splitted", "labelsTs")
        out_brain_folder = os.path.join(outfolder, "raw_splitted", "brainTs")
        tag = "" + cohort + "_"  # Tag to add to each case processed

    if not (os.path.exists(out_img_folder)):
        os.makedirs(out_img_folder)

    if not (os.path.exists(out_label_folder)):
        os.makedirs(out_label_folder)

    if not (os.path.exists(out_brain_folder)):
        os.makedirs(out_brain_folder)

    # Load label file
    label_df = pd.read_excel(df_info["label_file"])
    label_df = label_df.set_index("record_id")
    label_df = label_df.fillna(0)

    # Load image and label folders
    img_folder = df_info["img_folder"]
    label_folder = df_info["label_folder"]

    if ("." in img_folder) and ("." in label_folder):
        # There are several folders
        img_folder = img_folder.split(".")
        label_folder = label_folder.split(".")
    else:
        img_folder, label_folder = [img_folder], [label_folder]

    # Filter for cohort
    df_phase = df_phase[df_phase["Cohort"] == cohort]
    df_phase = df_phase.set_index("record_id")

    for enum_i, i in enumerate(img_folder):
        files = sorted(os.listdir(i))

        Parallel(n_jobs=workers)(
            delayed(process_id)(
                file,
                i,
                label_folder[enum_i],
                out_img_folder,
                out_label_folder,
                out_brain_folder,
                brain_folder,
                label_df,
                df_phase,
                tag,
                cohort,
            )
            for file in files
        )


def main(args):
    info_file = args.i
    phase_file = args.p
    brain_folder = args.b
    outfolder = args.o
    workers = args.np

    assert os.path.exists(info_file), f"Information file '{info_file}' does not exist"
    assert os.path.exists(phase_file), f"Phase file '{phase_file}' does not exist"
    assert os.path.exists(
        brain_folder
    ), f"Brain segmentation folder '{brain_folder}' does not exist"
    assert os.path.exists(
        os.path.dirname(outfolder)
    ), f"Parent output folder '{os.path.dirname(outfolder)}' does not exist"
    assert workers > 0, "Zero or negative parallel workers"

    # Read information file
    df_info = pd.read_csv(info_file)
    df_info = df_info.set_index("cohort")

    # Read phase file
    df_phase = pd.read_excel(phase_file)

    if not (os.path.exists(outfolder)):
        # Create output folder if it does not exist
        os.makedirs(outfolder)

    # Iterate through cohorts
    cohorts = df_info.index.astype(str).tolist()

    for cohort in cohorts:
        process_cohort(
            cohort=cohort,
            df_info=df_info.loc[cohort],
            brain_folder=brain_folder,
            df_phase=df_phase,
            outfolder=outfolder,
            workers=workers,
        )


def get_args():
    # Prepare data for Amsterdam project application in Germany
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--i", help="Information file for cohorts evaluated", required=True, type=str
    )
    parser.add_argument("--p", help="Phase file", required=True, type=str)
    parser.add_argument("--b", help="Brain segm folder", required=True, type=str)
    parser.add_argument("--o", help="Output folder", required=True, type=str)
    parser.add_argument("--np", help="Parallel workers", default=4, type=int)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    t1 = time.time()
    main(get_args())
    print("Time ellapsed (seconds): ", time.time() - t1)
