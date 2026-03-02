import os, sys
import numpy as np
import argparse
import time


def main(args):
    infolder = args.f
    outfile = args.o

    assert os.path.exists(infolder), f"Input folder '{infolder}' does not exist"
    assert os.path.exists(os.path.dirname(outfile)) and outfile.endswith(
        ".txt"
    ), f"Output parent directory '{os.path.dirname(outfile)}' does not exist or is not .txt"

    files = sorted(os.listdir(infolder))
    ids = []
    for file in files:
        if file.endswith(".json"):
            ids.append(file.replace(".json", ""))

    ids = np.array(ids, dtype=str)
    np.savetxt(outfile, ids, fmt="%s")


def get_args():
    parser = argparse.ArgumentParser(
        description="Obtain late phase case IDs from folder"
    )
    parser.add_argument("--f", help="Input folder", required=True, type=str)
    parser.add_argument("--o", help="Output file", required=True, type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    t1 = time.time()
    main(get_args())
    print(f"Time ellapsed: {time.time()-t1}sec")
