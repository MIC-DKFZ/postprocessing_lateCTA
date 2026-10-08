import os, sys
import SimpleITK as sitk
import numpy as np
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor_regression
import argparse
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    load_json,
    save_json,
)
import torch
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
import pandas as pd
import time


def dice_coefficient(mask1, mask2):
    """Dice = 2|A∩B| / (|A| + |B|)"""
    intersection = np.logical_and(mask1, mask2).sum()
    return 2.0 * intersection / (mask1.sum() + mask2.sum() + 1e-8)


def postprocess_preds(
    pred: np.ndarray,
    params: dict,
    image,
    brain: np.ndarray = None,
    cid: str = None,
    cfg: dict = None,
) -> np.ndarray:
    """
    Postprocess predictions with postprocessing
    parameters

    Params
    ------
    pred : prediction from original translation model
    params : postprocessing parameters (cutoff for distance, time values)
    brain : brain segmentation for case of interest

    Returns
    -------
    out : postprocessed output prediction

    """
    # Threshold distance map
    dist_thr = (pred[0] <= float(params["dist"])).astype(np.uint8)

    # Threshold segmentation map
    segm_thr = (pred[-1] >= float(params["logit"])).astype(np.uint8)

    # Union map
    out = ((dist_thr + segm_thr) > 0).astype(np.uint8)

    out = (out * pred[1]).astype(
        np.float32
    )  # Map with only time information in vessels

    # Clip map to avoid having times > 1 and negative times
    out = np.clip(out, a_min=0.0, a_max=1.0)

    # Restrict map to brain area, if brain segmentation is available
    if brain is not None:
        out *= brain

    return out.astype(np.float32), pred[-1].astype(np.float32)


def load_label_brain(label_file: os.PathLike) -> np.ndarray:
    """
    Load label or brain file

    Params
    ------
    label_file : input label file

    Returns
    -------
    out : output label information

    """
    out = sitk.GetArrayFromImage(sitk.ReadImage(label_file))
    return out


def main(args):
    data_folder = args.d
    model_folder = args.m
    out_folder = args.o
    brain_folder = args.b
    chunks = args.chunks
    part = args.part
    mode = args.mode
    cfg_file = args.cfg
    workers = args.np

    assert os.path.exists(data_folder), f"Data folder '{data_folder}' does not exist"
    assert os.path.exists(model_folder), f"Model folder '{model_folder}' does not exist"
    assert os.path.exists(brain_folder), f"Brain folder '{brain_folder}' does not exist"
    assert os.path.exists(
        os.path.dirname(out_folder)
    ), f"Parent output folder '{os.path.dirname(out_folder)}' does not exist"
    assert mode.lower().strip() in ["final", "best"]
    assert os.path.exists(cfg_file), f"Configuration file '{cfg_file}' does not exist"
    assert workers > 0, "Zero or negative number of parallel workers"

    # Create output folder if it does not exist
    if not (os.path.exists(out_folder)):
        os.makedirs(out_folder)

    # Obtain label and brain folders
    label_folder = os.path.join(os.path.dirname(data_folder), "labelsTs")

    # Load configuration
    cfg = load_json(cfg_file)

    # Load postprocessing parameters
    params_file = os.path.join(model_folder, "best_dist_logit_cutoff.json")
    assert os.path.exists(
        params_file
    ), f"Post processing parameter file '{params_file}' does not exist"
    params = load_json(params_file)

    # Initialize predictor
    predictor = nnUNetPredictor_regression(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device("cuda", 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=("all",),
        checkpoint_name=f"checkpoint_{mode.lower().strip()}.pth",
    )

    # Sample files
    files = np.array(sorted(os.listdir(data_folder)), dtype=str)

    # Split files if chunk splitting is to be allowed
    # Parallel inference across several GPUs
    if chunks > 0:
        assert (part > -1) & (
            part < chunks
        ), f"Partition index should be between 0 and below the number of chunks (now, part={part})"
        file_chunks = np.array_split(files, chunks)
        files = file_chunks[part]

    # Save all resulting metrics in dict
    metrics = {}

    # iterate through all files
    for file in files:
        cid = os.path.basename(file).replace("_0000.nii.gz", "")
        outfile = os.path.join(out_folder, f"{cid}.nii.gz")
        logit_file = os.path.join(out_folder, f"{cid}_logit.nii.gz")
        # brain_file = os.path.join(brain_folder, f"{cid}_brain.nii.gz")
        brain_file = os.path.join(brain_folder, f"{cid}.nii.gz")
        full_file = os.path.join(data_folder, file)

        # Load brain information
        brain_file = os.path.join(brain_folder, f"{cid}.nii.gz")
        brain = load_label_brain(label_file=brain_file)

        if not (os.path.exists(outfile)):
            print(f"cid : {cid}")
            img, props = SimpleITKIO().read_images([full_file])

            # Load brain information
            brain = load_label_brain(label_file=brain_file)
            ind = np.where(brain == 0)
            img[0, ind[0], ind[1], ind[2]] = -1024.0
            image = sitk.ReadImage(full_file)
            iterator = predictor.get_data_iterator_from_raw_npy_data(
                [img], None, [props], None, workers
            )
            r = predictor.predict_from_data_iterator(iterator, False, 1)

            # Postprocess prediction with distance, time, and segmentation heads

            out, logit_map = postprocess_preds(
                pred=r[0], params=params, brain=brain, image=image, cid=cid
            )

            # Store output postprocessed prediction
            image = sitk.ReadImage(full_file)
            out_image = sitk.GetImageFromArray(out)
            out_image.CopyInformation(image)
            sitk.WriteImage(out_image, outfile)

            # out_logit_image = sitk.GetImageFromArray(logit_map)
            # out_logit_image.CopyInformation(image)
            # sitk.WriteImage(out_logit_image, logit_file)

            # Load label and brain files, if they exist, and conduct eval
            """
            if os.path.exists(label_folder):
                label_file = os.path.join(label_folder, f"{cid}.nii.gz")

                if os.path.exists(label_file) and brain is not None:
                    label = load_label_brain(label_file=label_file)

                    # Derive vessel mask for ground-truth and prediction
                    vessel_gt = label[1] > 0
                    vessel_pred = out > 0

                    # Conduct evaluation

                    # MAE distance: multiply times std to obtain distance error in mm
                    mae_dist = (
                        np.mean(np.abs(label[0][brain > 0] - r[0][0][brain > 0]))
                        * cfg["std"]
                    )

                    # MAE time
                    mae_time = np.mean(
                        np.abs(label[1][vessel_gt > 0] - out[vessel_gt > 0])
                    )

                    # Segmentation Dice coefficient
                    dice = dice_coefficient(vessel_pred, vessel_gt)

                    print(
                        f"{cid}: Dice: {dice}, MAE distance: {mae_dist}, MAE time: {mae_time}"
                    )
                    metrics[cid] = {
                        "MAE_dist": float(mae_dist),
                        "MAE_time": float(mae_time),
                        "Dice": float(dice),
                    }

                    # Save case metrics as .json file
                    metrics_file = os.path.join(out_folder, f"{cid}.json")
                    save_json(metrics[cid], metrics_file)
            """

    # Save metrics, if any evaluation has been completed
    """
    outfiles = sorted(os.listdir(out_folder))
    all_metrics = {}
    for outfile in outfiles:
        if ".json" in outfile:
            cid = outfile.replace(".json", "")
            outfile = os.path.join(out_folder, outfile)
            info = load_json(outfile)
            all_metrics[cid] = info

    # Aggregate metrics
    agg_metrics = {}  # Dict with final aggregated metrics
    metrics_df = pd.DataFrame.from_dict(all_metrics, orient="index")

    # Iterate through metrics
    metric_names = list(metrics[cid].keys())

    for metric_name in metric_names:
        agg_metrics[metric_name] = float(metrics_df[metric_name].values.mean())
        print(f"Aggregated {metric_name} : {agg_metrics[metric_name]}")
    agg_metrics_file = os.path.join(out_folder, "metrics_aggregated.json")
    save_json(agg_metrics, agg_metrics_file)
    """


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", help="Data folder", required=True, type=str)
    parser.add_argument("--m", help="Model folder", required=True, type=str)
    parser.add_argument("--o", help="Output folder", required=True, type=str)
    parser.add_argument(
        "--b", help="Brain folder", required=False, type=str, default=None
    )
    parser.add_argument(
        "--mode", help="Checkpoint to evaluate", default="final", type=str
    )
    parser.add_argument(
        "--cfg", help="Inference configuration file", required=True, type=str
    )
    parser.add_argument(
        "--chunks", help="Number of chunks to partition data", default=-1, type=int
    )
    parser.add_argument("--part", help="Chunk index", default=-1, type=int)
    parser.add_argument("--np", help="Number of parallel workers", default=4, type=int)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(get_args())
