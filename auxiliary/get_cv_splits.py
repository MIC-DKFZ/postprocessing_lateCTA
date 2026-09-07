import os, sys
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import load_json, save_json
import time
import argparse
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


def main(args):
    infolder = args.i
    late_file = args.l
    outfile = args.o
    cv_folds = args.cv

    assert os.path.exists(infolder), f"Input folder '{infolder}' does not exist"
    assert os.path.exists(late_file), f"Late phase file '{late_file}' does not exist"
    assert os.path.exists(os.path.dirname(outfile)) and outfile.endswith(
        ".json"
    ), f"Parent directory '{os.path.dirname(outfile)}' does not exist or output file is not .json"

    # Obtain late phase IDs
    late_ids = np.loadtxt(late_file, dtype=str).tolist()

    # Obtain all IDs
    cids = {}
    files = sorted(os.listdir(infolder))
    for file in files:
        if file.endswith(".json"):
            full_file = os.path.join(infolder, file)
            cid = file.replace(".json", "")
            instances = load_json(full_file)["instances"]
            label_list = []
            occlusion_label = 1 if len(list(instances.keys())) > 0 else 0
            late_label = 1 if cid in late_ids else 0
            label_list.append(
                [occlusion_label, late_label, int(occlusion_label * late_label > 0)]
            )
            cids[cid] = label_list

    all_cids = np.array(list(cids.keys()), dtype=str)
    vals = np.squeeze(np.array(list(cids.values())))

    print(vals.shape)

    # Initialize multi-label stratified K-fold
    mskf = MultilabelStratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    out = []
    for fold, (train_idx, test_idx) in enumerate(mskf.split(all_cids, vals), 1):
        train_ids = all_cids[train_idx].tolist()
        test_ids = all_cids[test_idx].tolist()
        fold_info = {"train": train_ids, "val": test_ids}
        out.append(fold_info)
        print(np.sum(vals[train_idx], 0), np.sum(vals[test_idx], 0))

    save_json(out, outfile)


def get_args():
    parser = argparse.ArgumentParser(
        description="Create stratified CV fold for venous cases and positive occlusion cases"
    )
    parser.add_argument("--i", help="Input folder", required=True, type=str)
    parser.add_argument("--l", help="Late phase file", required=True, type=str)
    parser.add_argument("--o", help="Output file", required=True, type=str)
    parser.add_argument("--cv", help="Number of CV folds", default=5, type=int)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    t1 = time.time()
    main(get_args())
    print("Time ellapsed (seconds): ", time.time() - t1)
