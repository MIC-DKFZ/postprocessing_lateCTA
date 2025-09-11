import os, sys
import time
import argparse
from batchgenerators.utilities.file_and_folder_operations import save_json
import pandas as pd
from joblib import Parallel, delayed


def process_id(file: os.PathLike, phase_df: pd.DataFrame) -> str:
    """
    Check if ID is in arterial or venous phase,
    based on that, set it to training or validation

    Params
    ------
    file : input file
    phase_df : phase information

    Returns
    -------
    out : output purpose ('train' or 'val')

    """
    # Derive ID
    cid = file.replace("_0000.nii.gz", "")
    num_cid = cid.replace("STROKE_", "")

    print(cid)

    # Derive phase
    out = "train"
    phase_ind = phase_df.index.values.astype(str).tolist()
    if num_cid in phase_ind:
        phase_info = phase_df.loc[int(num_cid)]
        phase = phase_info["Phase"]

        if not ("arterial" in phase):
            out = "val"

    return {cid: out}


def main(args):
    infolder = args.i
    phasefile = args.p
    outfile = args.o
    workers = args.np

    assert os.path.exists(infolder), f"Input folder '{infolder}' does not exist"
    assert (
        os.path.exists(phasefile) and ".xlsx" in phasefile
    ), f"Phase file '{phasefile}' does not exist or is not .xlsx"
    assert os.path.exists(os.path.dirname(outfile)), f"Parent of output directory"
    assert ".json" in outfile, f"Output file '{outfile}' is not .json"
    assert workers > 0, "Zero or negative number of parallel workers"

    # Read phase file
    phase_df = pd.read_excel(phasefile)
    phase_df = phase_df.set_index("record_id")

    # We are only analyzing the "STROKE" cohort, so we just keep that
    phase_df = phase_df[phase_df["Cohort"] == "STROKE"]

    # Iterate through IDs
    outs = Parallel(n_jobs=workers)(
        delayed(process_id)(file=file, phase_df=phase_df)
        for file in sorted(os.listdir(infolder))
        if "_0000.nii.gz" in file
    )

    train_ids, val_ids = [], []
    for out in outs:
        ks = list(out.keys())
        if out[ks[0]] == "train":
            train_ids.append(ks[0])
        elif out[ks[0]] == "val":
            val_ids.append(ks[0])

    # Provide final dict
    out_list = [{"train": train_ids, "val": val_ids}]

    save_json(out_list, outfile)


def get_args():
    # Prepare data split for Amsterdam project application in Germany
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", help="Input folder", required=True, type=str)
    parser.add_argument("--p", help="Phase file", required=True, type=str)
    parser.add_argument("--o", help="Output file", required=True, type=str)
    parser.add_argument("--np", help="Parallel workers", default=4, type=int)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    t1 = time.time()
    main(get_args())
    print("Time ellapsed (seconds): ", time.time() - t1)
