import SimpleITK as sitk
import numpy as np
import os,sys
from registration.setParameters import set_parameters
from registration.registration_utils import get_transformation_matrix,apply_transformation
import time
import argparse
from loguru import logger
from utils.load_save import load_data


def register_frame(cta_image : sitk.Image, ctp_file : os.PathLike, out_folder : os.PathLike, p : dict, cfg : dict) -> sitk.Image:
    """
    Register CTP file (moving) to CTA file (fixed)
    and store the results in an output folder

    Params
    ------
    cta_image : fixed CTA image
    ctp_file : moving CTP file
    out_folder : folder with output results
    p : registration parameters
    cfg : registration configuration
    

    Returns
    -------
    Saved registration result in output folder
    
    """
    assert "default_val" in list(cfg.keys()), "Key 'default_val' absent in registration configuration"
    logger.info(f"Registering CTP frame {ctp_file} to CTA image")
    # Load moving image
    ctp_image = sitk.ReadImage(ctp_file)

    # Set up output file
    outfile = os.path.join(out_folder, os.path.basename(ctp_file))


    # Execute registration, based on parameters and registration
    new_cta = None # New registered CTA image in case that there are issues with the old CTA image
    try:
        transform_matrix = get_transformation_matrix(fixed=cta_image, 
                                                    moving=ctp_image, 
                                                    clipvalue=[None, None], 
                                                    parameters=p)
    except:
        logger.info("There was an error, registering first CTA to CTP...")

        transform_matrix = get_transformation_matrix(fixed=ctp_image, 
                                                    moving=cta_image, 
                                                    clipvalue=[None, None], 
                                                    parameters=p)
        
        new_cta = apply_transformation(transform_matrix, 
                                             moving=cta_image, 
                                             segmentation=False, 
                                             default_pixel=cfg["default_val"])
        new_cta.CopyInformation(ctp_image)
        

        transform_matrix = get_transformation_matrix(fixed=new_cta, 
                                                    moving=ctp_image, 
                                                    clipvalue=[None, None], 
                                                    parameters=p)
        
    transformed_image = apply_transformation(transform_matrix, 
                                             moving=ctp_image, 
                                             segmentation=False, 
                                             default_pixel=cfg["default_val"])
    
    if new_cta is None:
        transformed_image.CopyInformation(cta_image)
    else:
        transformed_image.CopyInformation(new_cta)

    # Save result
    sitk.WriteImage(transformed_image, outfile)

    return new_cta



def extract_files_to_register(in_folder : os.PathLike, out_folder : os.PathLike, cfg : dict) -> list:
    """
    Extract files to register

    Params
    ------
    in_folder : folder with input CTP frame files
    out_folder : folder with output CTP frame files
    cfg : registration configuration

    Returns
    -------
    files_register : files to be registered
    
    """
    cfg_keys = list(cfg.keys())

    assert "overwrite" in cfg_keys, "'overwrite' key not in configuration"
    overwrite = cfg["overwrite"]

    files_register = []
    
    if os.path.exists(in_folder):
        files_register = [os.path.join(in_folder,file) for file in sorted(os.listdir(in_folder))]
        if (overwrite.lower().strip() == "n") and (os.path.exists(out_folder)):
            # Check if registration files already exist, and skip them
            input_ctp_files = np.array(sorted(os.listdir(in_folder)), dtype=str)
            output_ctp_files = np.array(sorted(os.listdir(out_folder)), dtype=str)
            files_register = []
            if not(np.array_equal(input_ctp_files, output_ctp_files)):
                # The case has been issued, but not all the frames have been registered
                remaining_files = np.setdiff1d(input_ctp_files, output_ctp_files)
                files_register = [os.path.join(in_folder,remaining_file) for remaining_file in remaining_files]

    return files_register


def execute_registration(cta_file : os.PathLike, ctp_folder : os.PathLike, out_folder : os.PathLike, cfg: dict, p : dict):
    """
    Register CTA file to corresponding case ID in CTP folder

    Params
    ------
    cta_file : CTA file to be used as fixed image in the registration
    ctp_folder : folder with CTP data
    out_folder : folder where to store the registered CTP data
    cfg : configuration for registration
    p : registration parameters

    Returns
    -------
    Saved CTP files in output folder
    
    """
    # Obtain case ID
    file = os.path.basename(cta_file)
    cid = file.split(".")[0]

    # Set up input and output folders for the case ID of interest
    cid_ctp_folder = os.path.join(ctp_folder, cid)
    cid_out_folder = os.path.join(out_folder, cid)

    # Extract files to register
    files_register = extract_files_to_register(in_folder = cid_ctp_folder, 
                                               out_folder=cid_out_folder,
                                               cfg=cfg)
    
    # Iterate through the files for registration
    if len(files_register) > 0:
        logger.info(f"Processing case ID '{cid}'")
        # Load CTA image
        cta_image = sitk.ReadImage(cta_file)
        # Create output folder if it does not exist
        if not(os.path.exists(cid_out_folder)):
            os.makedirs(cid_out_folder)
        t1 = time.time()
        new_cta = None
        modified_cta = False # Whether the CTA was modified or not during the registration process, to save it later in disk
        for file in files_register:
  
            new_cta = register_frame(cta_image=cta_image, 
                        ctp_file=file, 
                        out_folder=cid_out_folder, 
                        p=p, 
                        cfg=cfg)
            
            if new_cta is not None:
                cta_image = new_cta
                modified_cta = True
                      
        if modified_cta:
            new_cta_file = cta_file.split(".")[0] + "_mod.nii.gz"
            logger.info(f"Storing the newer version of the CTA: '{new_cta_file}'")
            sitk.WriteImage(new_cta, new_cta_file)

        logger.info(f"Time ellapsed: {time.time()-t1} sec")




def main(args):
    # Load arguments
    ctp_folder = args.ctp
    cta_folder = args.cta
    out_folder = args.out

    assert os.path.exists(ctp_folder) and os.path.exists(cta_folder), f"Folder with CTA data '{cta_folder}' or with CTP data '{ctp_folder}' does not exist"
    assert os.path.exists(os.path.dirname(out_folder)), f"Parent folder of output folder '{out_folder}' does not exist"

    if not(os.path.exists(out_folder)):
        os.makedirs(out_folder)

    # Set up logfile
    logfile = os.path.join(os.path.dirname(out_folder),"register.log")
    logger.add(logfile, level="INFO")

    # Load config
    config_file = "register_config.json"
    assert os.path.exists(config_file), f"Configuration file for registration '{config_file}' does not exist"
    cfg = load_data(filename=config_file)
    cfg_keys = list(cfg.keys())

    # Set up registration parameters
    assert "method" in cfg_keys, "Key 'method' absent in configuration"
    assert "default_val" in cfg_keys, "Key 'default_val' absent in configuration"
    assert "metric" in cfg_keys, "Key 'metric' absent in configuration"
    
    p = set_parameters(method=cfg["method"], 
                       DefaultPixelValue=cfg["default_val"], 
                       metric=cfg["metric"])

    # Iterate through cases in CTA folder
    cta_files = sorted(os.listdir(cta_folder))
    for cta_file in cta_files:
        if ".nii.gz" in cta_file:
            full_cta_file = os.path.join(cta_folder, cta_file)
            execute_registration(cta_file=full_cta_file, 
                                ctp_folder=ctp_folder, 
                                out_folder=out_folder, 
                                cfg=cfg, 
                                p=p)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctp", help="Folder with CTP data", required=True, type=str)
    parser.add_argument("--cta", help="Folder with CTA data", required=True, type=str)
    parser.add_argument("--out", help="Output folder", required=True, type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(get_args())

    """
    t1 = time.time()
    ctp_folder = "/scratch/amartinezmora/raw_data/ctp/mrclean_late_30002"
    atlas_file = "/scratch/amartinezmora/raw_data/cta/mrclean_late_30002.nii.gz"
    out_folder = "/scratch/amartinezmora/raw_data/ctp_registered_to_cta"

    assert os.path.exists(ctp_folder) and os.path.exists(atlas_file), f"CTP folder '{ctp_folder}' or atlas file '{atlas_file}' do not exist"

    if not(os.path.exists(out_folder)):
        os.makedirs(out_folder)

    cid = os.path.basename(ctp_folder)
    out_folder = os.path.join(out_folder, cid)
    #if os.path.exists(out_folder):
        #shutil.rmtree(out_folder)
    if not(os.path.exists(out_folder)):
        os.makedirs(out_folder)

    # Copy folder into destination, to do all the registration stuff in there
    #shutil.copytree(ctp_folder, out_folder)
    #outfile = os.path.join(out_folder, os.path.basename(atlas_file))
    #shutil.copy(atlas_file, out_folder)

    # Load first frame of CTP and atlas
    ctp_frame_files = sorted(os.listdir(ctp_folder))

    atlas_image = sitk.ReadImage(atlas_file)

    # Set parameters
    p = set_parameters(method='euler', DefaultPixelValue=-1024, metric='AdvancedNormalizedCorrelation')
    
    for ctp_frame_file in ctp_frame_files:
        if ".nii.gz" in ctp_frame_file:
            ctp_frame_file = os.path.join(ctp_folder, ctp_frame_file)
            t_frame = time.time()
            # For each CTP frame, register it to the atlas
            frame_image = sitk.ReadImage(ctp_frame_file)
                
            # Derive transformation matrix
            transform_matrix = get_transformation_matrix(fixed=atlas_image, moving=frame_image, clipvalue=[None, None], parameters=p)
            transformed_image = apply_transformation(transform_matrix, moving=frame_image, segmentation=False, default_pixel=-1024)
            outfile = os.path.join(out_folder, os.path.basename(ctp_frame_file))
            sitk.WriteImage(transformed_image, outfile)
            print(f"Time ellapsed for {os.path.basename(ctp_frame_file)}:{time.time()-t_frame}sec")

    print(time.time()-t1)
    
    for frame in ctp_frame_files:
        frame_image = sitk.ReadImage(frame)
        frame_image = apply_transformation(transform_matrix, moving=frame_image, segmentation=False, default_pixel=-1024)
        outfile = os.path.join(out_folder, frame)
        sitk.WriteImage(frame_image, outfile)

    """
