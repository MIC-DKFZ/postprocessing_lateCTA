import zipfile
import os,sys
import pydicom
import io
import numpy as np
from joblib import Parallel, delayed
import time
import pandas as pd
import argparse
import glob


def safe_get(ds, key):
    return getattr(ds, key, "Unknown")

def load_dicom_zip(path : os.PathLike):
    """
    Load DICOM information from ZIP file

    Params
    ------
    path : DICOM filepath

    Returns
    -------
    dcm_data : DICOM data headers
    
    """

    with open(path, "rb") as f:
        zip_bytes = f.read()

    zip_buffer = io.BytesIO(zip_bytes)

    with zipfile.ZipFile(zip_buffer, "r") as zf:
        files = np.array(zf.namelist(), dtype=str)
        mask = np.char.endswith(files, ".dcm")
        dcms = files[mask]

        if len(dcms) > 0:
            with zf.open(dcms[0]) as dcm_file:
                dcm_bytes = dcm_file.read()
                dcm_data = pydicom.dcmread(io.BytesIO(dcm_bytes))
                dcm_file.close()
                zf.close()
                return dcm_data
            
            
    return dcm_data



def retrieve_metadata(data) -> dict:
    """
    Retrieve specific fields from DICOM metadata

    Params
    ------
    data : DICOM metadata

    Returns
    -------
    fields : retrieved fields

    """
    sex = safe_get(data, "PatientSex")
    age = safe_get(data, "PatientAge")
    manufacturer = safe_get(data, "Manufacturer")
    model = safe_get(data, "ManufacturerModelName")
    spacing = safe_get(data, "PixelSpacing")
    kernel = safe_get(data, "ConvolutionKernel")

    fields = {"sex" : sex,
              "age" : age,
              "manufacturer" : manufacturer, 
              "model" : model,
              "spacing" : spacing,
              "kernel" : kernel}
    
    return fields


def process_file(file, folder):
    cid = file.replace(".zip", "")
    print(cid)
    full_file = os.path.join(folder, file)
    data = load_dicom_zip(path=full_file)
    fields = retrieve_metadata(data=data)
    return {cid : fields}


def process_folder(subfolder, folder):
    print(subfolder)
    full_folder = os.path.join(folder, subfolder)
    files = sorted(glob.glob(f"{full_folder}/*.zip"))
    file = files[-1]
    data = load_dicom_zip(path=file)
    fields = retrieve_metadata(data=data)
    return {subfolder : fields}


def main(args):
    dcm_folder = args.d
    outfile = args.o
    zips = bool(args.z)
    workers = args.np

    assert os.path.exists(dcm_folder), f"DICOM folder '{dcm_folder}' does not exist"
    assert os.path.exists(os.path.dirname(outfile)), f"Parent output folder '{os.path.dirname(outfile)}' does not exist or file is not .csv"
    assert workers > 0, "Zero or negative number of workers"

    files = sorted(os.listdir(dcm_folder))
    if zips:
        all_fields = Parallel(n_jobs=workers)(delayed(process_file)(file, dcm_folder) for file in files if file.endswith(".zip"))
    else:
        all_fields = Parallel(n_jobs=workers)(delayed(process_folder)(file, dcm_folder) for file in files)

    out = {}
    for f in all_fields:
        out.update(f)

    # Save information
    df = pd.DataFrame.from_dict(out, orient="index")
    df.to_csv(outfile)

def get_args():
    # Obtain DICOM metadata
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", help="DICOM folder", required=True, type=str)
    parser.add_argument("--o", help="Output file", required=True, type=str)
    parser.add_argument("--z", help="Folder structure composed of zip files", default=0, type=int)
    parser.add_argument("--np", help="Number of parallel workers", default=4, type=int)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    t1 = time.time()
    main(get_args())
    print(f"Time ellapsed: {time.time()-t1}sec")