# SPDX-FileCopyrightText: Copyright 2026 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: Apache-2.0

import torch
import os, sys
import numpy as np
import time
import argparse
from batchgenerators.utilities.file_and_folder_operations import load_json


def main(args):
    cpt_file = args.c
    model_folder = args.o

    assert os.path.exists(cpt_file), f"Checkpoint file '{cpt_file}' does not exist"
    assert os.path.exists(
        os.path.dirname(model_folder)
    ), f"Output parent folder '{os.path.dirname(model_folder)}' does not exist"

    if not (os.path.exists(model_folder)):
        os.makedirs(model_folder)

    # Load json
    cpts = load_json(cpt_file)
    weights = np.array(list(cpts.values()))
    weights /= weights.sum() + np.finfo(float).eps
    cpt_files = list(cpts.keys())

    avg_state_dict = None
    for cpt, weight in zip(cpt_files, weights):
        assert os.path.exists(cpt), f"Checkpoint file '{cpt}' does not exist"
        ckpt = torch.load(cpt, map_location="cpu")
        state_dict = ckpt["state_dict"]

        if avg_state_dict is None:
            avg_state_dict = {k: v.clone() * weight for k, v in state_dict.items()}
        else:
            for k in avg_state_dict:
                avg_state_dict[k] += state_dict[k] * weight

    for i in range(5):
        fold_folder = os.path.join(model_folder, f"fold{i}")
        if not (os.path.exists(fold_folder)):
            os.makedirs(fold_folder)
        final_ckpt_file = os.path.join(fold_folder, "model_transfer.ckpt")

        # use first checkpoint as template
        final_ckpt = torch.load(cpt_files[0], map_location="cpu")
        final_ckpt["state_dict"] = avg_state_dict

        torch.save(final_ckpt, final_ckpt_file)


def get_args():
    parser = argparse.ArgumentParser(
        description="Average several checkpoints and store the results in model directory ready to train for nnDetection"
    )
    parser.add_argument("--c", help="Checkpoint file", required=True, type=str)
    parser.add_argument("--o", help="Model folder", required=True, type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    t1 = time.time()
    main(get_args())
    print(f"Time ellapsed: {time.time()-t1}sec")
