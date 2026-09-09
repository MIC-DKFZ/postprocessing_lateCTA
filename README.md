# Anatomy- And Time-Based Post-Processing with CT Perfusion-Derived Time–Vessel Maps to Reduce Incorrect Occlusion Detections in Late-Phase CT Angiography

Background/Objectives: Computer-aided vessel occlusion detection for acute ischemic stroke is typically developed on early-phase CT angiography (CTA). In late-phase CTA, enhanced venous contrast can mimic arterial occlusions and lead to incorrect occlusion detections. To counteract this, we propose a constraint-based post-processing framework that leverages anatomical and temporal vascular information from CT perfusion (CTP) scans. Methods: Vascular anatomy and relative contrast-arrival times were encoded as time–vessel maps. Three complementary constraints worked on these maps to remove incorrect detections based on distance-to-vessel truncation points, contrast-arrival time, and spatial priors. Constraints were fixed on a 32-case late-phase development subset and applied to an external held-out cohort of late-phase CTA (N = 354). To enable application without CTP, an auxiliary CTA-to-time–vessel map generator was designed to estimate maps directly from CTA. Results: In the external cohort, the framework reduced incorrect occlusions per scan from 3.9 (95% CI 3.6–4.2) to 2.1 (95% CI 1.9–2.3), preserving occlusion-level sensitivity at 80% (95% CI 69–91). Patient-level AUROC was 83 for the baseline versus 88 with post-processing. Improvements were also observed in detectors exposed to late-phase CTA, where post-processing reduced incorrect occlusions per scan from 2.8 to 1.9. As only 51 occlusion-positive cases were present in the external cohort, confidence intervals were wide. Consequently, these estimates should be interpreted with caution. Conclusions: Anatomically and temporally informed post-processing reduced incorrect occlusion detections in late-phase CTA without measurable loss of sensitivity, potentially improving robustness against CTA phase changes in external institutions.

Please cite the following paper if you use this coder: PLACEHOLDER

## Installation

All of the code in this repository runs from **two conda environments**:

- **`ctp-postprocess`** — the main environment: PyTorch, [nnDetection](https://github.com/MIC-DKFZ/nnDetection) (occlusion detection), the `Skeleton-recall` nnU-Net v2 fork bundled in this repo (CTA-to-time–vessel map generator), SimpleElastix (registration), and other dependencies.
- **`ctp-totalseg`** — an environment containing only [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) and its own pinned `nnunetv2`. It is kept separate because TotalSegmentator's PyPI `nnunetv2` would otherwise collide with the custom `Skeleton-recall` fork used everywhere else.

> **Note:** nnDetection's official releases are only tested against PyTorch 1.x (its README states PyTorch 2.0+ is not supported), while `Skeleton-recall` requires `torch>=2.1.2`. The steps below install nnDetection against PyTorch 2.x anyway, overriding its pinned dependencies. 

### 1. Main environment (`ctp-postprocess`)

```bash
conda create -n ctp-postprocess python=3.10
conda activate ctp-postprocess

# --- PyTorch (CUDA 11.8 build; adjust to your driver/CUDA toolkit) ---
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu118

# --- Build tools needed to compile nnDetection's CUDA extensions ---
conda install -c conda-forge gxx_linux-64 ninja
conda install -c nvidia/label/cuda-11.8.0 cuda-toolkit

# Install SimpleITK and Lightning
pip install -U "SimpleITK>=2.2.1" "pytorch-lightning>=2.0"

# --- nnDetection --- (to be updated with nnDetection v2 when this new version is out)
git clone https://github.com/MIC-DKFZ/nnDetection.git
cd nnDetection
pip install -r requirements.txt
pip install hydra-core --upgrade --pre
pip install git+https://github.com/mibaumgartner/pytorch_model_summary.git
FORCE_CUDA=1 pip install -v -e .
cd ..

# --- Skeleton-recall (custom nnU-Net v2 fork bundled in this repo for CTA-to-time-vessel map generator) ---
pip install -e ./Skeleton-recall

# --- SimpleElastix build of SimpleITK ---
pip install SimpleITK-SimpleElastix

# --- Remaining direct dependencies ---
pip install pandas scipy scikit-image scikit-learn scikit-fmm \
    nibabel pydicom tifffile matplotlib seaborn monai \
    loguru tqdm h5py joblib xmltodict PyYAML \
    iterative-stratification blosc2 einops
```

nnDetection also requires a environment variables to be set:

```bash
export det_data=/path/to/det_data # Folder with CTA data and vessel occlusion labels
export det_models=/path/to/det_models # Folder where to store vessel occlusion detectors
export OMP_NUM_THREADS=1
export det_num_threads=6
```

### 2. TotalSegmentator environment (`ctp-totalseg`)

```bash
conda create -n ctp-totalseg python=3.10
conda activate ctp-totalseg
pip install TotalSegmentator
```

`postprocess_end2end.py` calls TotalSegmentator as an external subprocess and does not need `ctp-totalseg` to be active — just point it at the environment's binary:

```bash
export TOTALSEG_BIN=/path/to/conda/envs/ctp-totalseg/bin/TotalSegmentator
# or pass --totalseg_bin /path/to/conda/envs/ctp-totalseg/bin/TotalSegmentator
```

## Main commands

# If you already have a trained vessel occlusion detector in nnDetection and a trained CTA-to-time-vessel map generator
```
conda activate ctp-postprocess
python postprocess_end2end.py --d /path/to/cta/data --m /path/to/generator/model --p /path/to/det_models/TaskXYZ/model_name/foldF/val_or_test_predictions --o /path/to/postprocessed/predictions --cfg nndet/fpr_cfg.json --brain_cache /path/to/store/brain/segmentations --totalseg_bin /path/to/conda/envs/ctp-totalseg/bin/TotalSegmentator 

```

# If you want to start from a folder with CTP data and a folder with CTA data

# Required data structure
ctp_folder (including 3D time steps for every case ID as .nii.gz)
   |__case0
      |__case0_t_00.nii.gz
      |__case0_t_01.nii.gz
      ...
      |__case0_t_NN.nii.gz
      
   ...
   
   |__caseM
      |__caseM_t_00.nii.gz
      |__caseM_t_01.nii.gz
      ...
      |__caseM_t_NN.nii.gz
      
ctp_folder_with_time_information (.npy files with time steps in seconds for each CTP slice acquired)
   |__case0_AcquisitionDateTime.npy
   
   ...
   
   |__caseM_AcquisitionDateTime.npy
   
cta_folder
   |__case0.nii.gz
   
   ...
   
   |__caseM.nii.gz
   
   
# 1. CTP-to-CTA registration
```
conda activate ctp-postprocess
python register.py --cta /your/cta/folder --ctp /your/ctp/folder --out /output/registered/ctp/folder
```

# 2. CTP curation (start with the output registered CTP folder from Step 1). USE THE TOTALSEG ENV!!
```
conda activate ctp-totalseg

```
