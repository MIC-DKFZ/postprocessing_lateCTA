# SPDX-FileCopyrightText: Copyright 2026 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: Apache-2.0

import os, sys
import SimpleITK as sitk
import numpy as np
from scipy.ndimage import binary_erosion
import time
import skfmm
import argparse
import nibabel as nib
import multiprocessing
from joblib import Parallel, delayed
from typing import Union
import matplotlib.pyplot as plt

from compute_phase import apply_window
from utils.load_save import load_data, write_data


def distance_map(img: np.ndarray, val: float, spacing: np.ndarray):
    """
    Obtain map of distances for a given mask and its
    respective spacing

    Params
    ------
    img : input image with time information
    val : time value of interest
    mask : input mask
    spacing : respective spacing


    Returns
    -------
    inv_dist : map with inverse distances

    """
    # Obtain mask from time image, only for value of interest
    mask = (img == val).astype(int)

    # Derive minimum spatial distance in map (voxel diagonal)
    voxel_diagonal = np.linalg.norm(spacing)

    # Convert mask to -1 in foreground, and +1 in background
    phi = -2 * mask + 1
    dist = skfmm.distance(phi, dx=spacing) + voxel_diagonal
    inv_dist = 1 / dist

    return inv_dist


def distance_map_full(img: np.ndarray, spacing: np.ndarray):
    """
    Obtain map of distances for a given mask and its
    respective spacing, for the full vasculature

    Params
    ------
    img : input image with time information
    val : time value of interest
    mask : input mask
    spacing : respective spacing


    Returns
    -------
    inv_dist : map with inverse distances

    """
    # Obtain mask from time image, only for value of interest
    mask = img > 0

    # Convert mask to -1 in foreground, and +1 in background
    phi = -2 * mask + 1
    dist = skfmm.distance(phi, dx=spacing)

    return dist


def bin_times(img: np.ndarray, bins: int = 4):
    """
    Bin values from time map

    Params
    ------
    img : input time map
    bins : bins for time data


    Returns
    -------
    unique_values : resulting time values to analyze

    """
    bin_mask = img != 0.0  # Obtain binary vasculature image
    # Erode image with two iterations, to remove time values with little influence and save computation time
    eroded_mask = binary_erosion(bin_mask, iterations=2)

    # Resulting time values
    time_values = img[eroded_mask].flatten()
    unique_values = np.unique(time_values)

    # Determine percentiles where to apply binning
    # ps = np.linspace(0, 1, num=bins)

    return unique_values


def weigh_distance_maps(
    dist_maps: list,
    vals: np.ndarray,
    out_file: os.PathLike,
    image,
    brain: np.ndarray = None,
):
    """
    Weigh and combine distance maps for different times

    Params
    ------
    dist_maps : set of distance maps computed for different times
    vals : different time values
    out_file : output file
    image : original time map image
    brain : brain segmentation (default: None)

    Returns
    -------
    final : final combination of time and distance

    """
    # convert maps to stack
    dist_maps = np.stack(dist_maps)

    # Normalize weight maps
    weight_maps = dist_maps / (np.sum(dist_maps, 0) + np.finfo(float).eps)

    # Obtain final map by weighting with time values
    final = np.sum(vals[:, None, None, None] * weight_maps, axis=0)

    # Remove values outside the brain
    if brain is not None:
        final[brain == 0] = final.min()

    # Save final result
    final_map_image = sitk.GetImageFromArray(final)
    final_map_image.CopyInformation(image)
    sitk.WriteImage(final_map_image, out_file)

    return final


def signed_time_image(img: np.ndarray):
    """
    Provide a sign to the time image. Giving negative values to venous areas,
    and positive values to arterial areas

    Use a percentile of 30 as separation between arteries and veins,
    according to medical literature

    Params
    ------
    img : input time image


    Returns
    -------
    signed_img : signed time image

    """
    time_vals = img[img > 0].flatten()
    p30 = np.percentile(time_vals, 30)
    signed_img = np.zeros(img.shape)
    signed_img[img > 0] = p30 + 0.01 - img[img > 0]

    return signed_img


def process_file(
    time_file: os.PathLike,
    outfile: os.PathLike,
    norm: bool = False,
    norm_stats: dict = None,
) -> Union[float, float, float]:
    """
    Obtain Eikonal distance map from time file
    Alternatively, obtain also statistics for distance map normalization

    Params
    ------
    time_file : file with time vessel information
    outfile : output file
    norm : whether to obtain normalization statistics or not
    norm_stats : statistics for normalization

    Returns
    -------
    Saved distance map and QA file

    Alternatively:

    n : number of voxels in distance map (ignoring voxels below 0.5 percentile or over 99.5 percentile)
    s : sum of voxels in distance map (ignoring voxels below 0.5 percentile or over 99.5 percentile)
    s_squares : sum of squared voxels in distance map (ignoring voxels below 0.5 percentile or over 99.5 percentile)

    """

    if not (os.path.exists(outfile)):
        image = sitk.ReadImage(time_file)
        img = sitk.GetArrayFromImage(image)
        spacing = np.array(image.GetSpacing())

        # Obtain distance map
        final = distance_map_full(img=img, spacing=spacing)

        if not (norm) and norm_stats is None:
            # Save final result
            print(f"Processing {os.path.basename(outfile)}")
            final_map_image = sitk.GetImageFromArray(final.astype(np.float32))
            final_map_image.CopyInformation(image)
            sitk.WriteImage(final_map_image, outfile)
        elif norm_stats is not None:
            final = final / (norm_stats["std"] + np.finfo(float).eps)

            # Save final result
            print(f"Processing {os.path.basename(outfile)}")
            final_map_image = sitk.GetImageFromArray(final.astype(np.float32))
            final_map_image.CopyInformation(image)
            sitk.WriteImage(final_map_image, outfile)

        if norm:
            # Derive statistics for normalization
            flattened_map = final.flatten()
            p05, p995 = np.percentile(flattened_map, 0.5), np.percentile(
                flattened_map, 99.5
            )
            mask = (flattened_map > p05) & (flattened_map < p995)
            n = mask.sum()
            s = flattened_map[mask].sum()
            s_squares = (flattened_map[mask] ** 2).sum()

            return n, s, s_squares

        else:

            # Save plots for QA
            ww_img = np.percentile(img[img > 0], 99)
            ww_final = np.percentile(final, 99) - np.percentile(final, 1)
            min_final = np.percentile(final, 1) + ww_final / 2
            final_w = apply_window(
                image=final, window_center=min_final, window_width=ww_final
            )
            img_w = apply_window(
                image=img, window_center=ww_img / 2, window_width=ww_img
            )
            images = [final_w, img_w]

            plt.figure()
            for enum_i, i in enumerate(images):
                plt.subplot(2, 3, enum_i * 3 + 1)
                plt.imshow(i[i.shape[0] // 2])
                plt.xticks([])
                plt.yticks([])
                plt.colorbar()
                plt.subplot(2, 3, enum_i * 3 + 2)
                plt.imshow(i[:, i.shape[1] // 2])
                plt.xticks([])
                plt.yticks([])
                plt.colorbar()
                plt.subplot(2, 3, enum_i * 3 + 3)
                plt.imshow(i[:, :, i.shape[2] // 2])
                plt.xticks([])
                plt.yticks([])
                plt.colorbar()

            plt.savefig(outfile.replace(".nii.gz", ".png"))
            plt.close()


def main(args):
    time_folder = args.t
    brain_folder = args.b
    workers = args.w
    out_folder = args.o
    bins = args.bin
    norm = bool(args.norm)

    assert os.path.exists(time_folder), f"Time folder '{time_folder}' does not exist"
    # assert os.path.exists(brain_folder), f"Brain folder '{brain_folder}' does not exist"

    if not (os.path.exists(out_folder)):
        # Create output folder if it does not exist
        os.makedirs(out_folder)

    # Derive case IDs to compute
    files = sorted(os.listdir(time_folder))
    # tag = "_norm.nii.gz"  # File tag to look for
    tag = ".nii.gz"  # File tag to look for

    norm_stats = None

    # Apply normalization, if necessary
    if norm:
        params_file = os.path.join(out_folder, "params.json")
        if os.path.exists(params_file):
            # Load normalization statistics, if they already exist
            norm_stats = load_data(filename=params_file)
        else:
            params = Parallel(n_jobs=workers)(
                delayed(process_file)(
                    os.path.join(time_folder, file),
                    os.path.join(out_folder, file.replace(tag, "") + ".nii.gz"),
                    norm,
                )
                for file in files
                if tag in file
            )
            params = np.array(params)

            # Aggregate stats across case IDs
            sum_params = np.sum(params, 0)

            mean = float(sum_params[1] / (sum_params[0] + np.finfo(float).eps))
            std = float(
                np.sqrt(
                    (sum_params[-1] / (sum_params[0] + np.finfo(float).eps)) - mean**2
                )
            )
            norm_stats = {"mean": mean, "std": std}
            # Save normalization statistics
            write_data(data=norm_stats, filename=params_file)

        # Set up norm parameter to False
        norm = False

    Parallel(n_jobs=workers)(
        delayed(process_file)(
            os.path.join(time_folder, file),
            os.path.join(out_folder, file.replace(tag, "") + ".nii.gz"),
            norm,
            norm_stats,
        )
        for file in files
        if tag in file
    )


def get_args():
    # Derive time-distance maps for translation model development
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--t", help="Folder with time information", required=True, type=str
    )
    parser.add_argument(
        "--b", help="Folder with brain information", required=True, type=str
    )
    parser.add_argument("--w", help="Number of parallel workers", default=4, type=int)
    parser.add_argument(
        "--bin",
        help="Number of bins to structure time information",
        default=4,
        type=int,
    )
    parser.add_argument(
        "--norm", help="Apply distance-based normalization", default=0, type=int
    )
    parser.add_argument("--o", help="Output folder", required=True, type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    t1 = time.time()
    main(get_args())
    print("Time ellapsed (seconds): ", time.time() - t1)
