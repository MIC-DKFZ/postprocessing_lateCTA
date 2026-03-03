import os, sys
import argparse
from joblib import Parallel, delayed
from batchgenerators.utilities.file_and_folder_operations import load_json
import time
import numpy as np


def process_case(file, before, after):
    before_file = os.path.join(before, file)
    after_file = os.path.join(after, file)

    b = load_json(before_file)
    a = load_json(after_file)

    b_keys = len(list(b.keys()))
    a_keys = len(list(a.keys()))
    diff = b_keys - a_keys

    b_boxes, a_boxes = [], []
    if b_keys > 0:
        for k in list(b.keys()):
            box = np.array(b[k]["box"])
            b_boxes.append(box)
    b_boxes = np.array(b_boxes)

    if a_keys > 0:
        for k in list(a.keys()):
            box = np.array(a[k]["box"])
            a_boxes.append(box)
    a_boxes = np.array(a_boxes)

    dists = []
    if b_boxes.shape[0] > 0 and a_boxes.shape[0] > 0:
        for i in range(b_boxes.shape[0]):
            for j in range(a_boxes.shape[0]):
                dist = np.linalg.norm(b_boxes[i] - a_boxes[j])
                dists.append(dist)

    if len(dists) > 0:
        print(file, diff, max(dists))


def main(args):
    before = args.b
    after = args.a
    workers = args.np

    assert os.path.exists(
        before
    ), f"Folder with information before FPR '{before}' does not exist"
    assert os.path.exists(
        after
    ), f"Folder with information after FPR '{after}' does not exist"
    assert workers > 0, "Zero or negative number of parallel workers"

    files = sorted(os.listdir(before))
    Parallel(n_jobs=workers)(
        delayed(process_case)(file, before, after)
        for file in files
        if file.endswith(".json")
    )


def get_args():
    # Check which cases present the greatest difference before and after applying FPR
    parser = argparse.ArgumentParser()
    parser.add_argument("--b", help="Predictions before FPR", required=True, type=str)
    parser.add_argument("--a", help="Predictions after FPR", required=True, type=str)
    parser.add_argument("--np", help="Parallel workers", default=4, type=int)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    t1 = time.time()
    main(get_args())
    print("Time ellapsed (seconds): ", time.time() - t1)
