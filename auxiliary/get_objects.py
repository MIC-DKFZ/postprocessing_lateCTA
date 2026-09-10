# SPDX-FileCopyrightText: Copyright 2026 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: Apache-2.0

import os, sys
import time
import argparse
from joblib import Parallel, delayed
import numpy as np
from nndet.io.load import load_json


def process_id(infolder, file):

    full_file = os.path.join(infolder, file)
    info = load_json(full_file)["instances"]
    objects = len(list(info.keys()))
    return objects, int(objects > 0)


def main(args):
    infolder = args.i
    workers = args.np

    files = sorted(os.listdir(infolder))
    info = Parallel(n_jobs=workers)(
        delayed(process_id)(infolder, file) for file in files if file.endswith(".json")
    )

    info = np.array(info)
    s = np.sum(info, 0)
    print(s)


def get_args():
    # Gather number of objects and positive cases in external cohort
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", help="Input folder", type=str)
    parser.add_argument("--np", help="Number of workers", type=int)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    t1 = time.time()
    main(get_args())
    print(f"Time ellapsed: {time.time()-t1}sec")
