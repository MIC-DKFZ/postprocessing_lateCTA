# Anatomy- And Time-Based Post-Processing with CT Perfusion-Derived Time–Vessel Maps to Reduce Incorrect Occlusion Detections in Late-Phase CT Angiography

Background/Objectives: Computer-aided vessel occlusion detection for acute ischemic stroke is typically developed on early-phase CT angiography (CTA). In late-phase CTA, enhanced venous contrast can mimic arterial occlusions and lead to incorrect occlusion detections. To counteract this, we propose a constraint-based post-processing framework that leverages anatomical and temporal vascular information from CT perfusion (CTP) scans. Methods: Vascular anatomy and relative contrast-arrival times were encoded as time–vessel maps. Three complementary constraints worked on these maps to remove incorrect detections based on distance-to-vessel truncation points, contrast-arrival time, and spatial priors. Constraints were fixed on a 32-case late-phase development subset and applied to an external held-out cohort of late-phase CTA (N = 354). To enable application without CTP, an auxiliary CTA-to-time–vessel map generator was designed to estimate maps directly from CTA. Results: In the external cohort, the framework reduced incorrect occlusions per scan from 3.9 (95% CI 3.6–4.2) to 2.1 (95% CI 1.9–2.3), preserving occlusion-level sensitivity at 80% (95% CI 69–91). Patient-level AUROC was 83 for the baseline versus 88 with post-processing. Improvements were also observed in detectors exposed to late-phase CTA, where post-processing reduced incorrect occlusions per scan from 2.8 to 1.9. As only 51 occlusion-positive cases were present in the external cohort, confidence intervals were wide. Consequently, these estimates should be interpreted with caution. Conclusions: Anatomically and temporally informed post-processing reduced incorrect occlusion detections in late-phase CTA without measurable loss of sensitivity, potentially improving robustness against CTA phase changes in external institutions.

Please cite the following paper if you use this coder: PLACEHOLDER

## Installation

All of the code in this repository runs from **two conda environments**:

- **`ctp-postprocess`** — the main environment: PyTorch, [nnDetection](https://github.com/MIC-DKFZ/nnDetection) (occlusion detection), the `Skeleton-recall` nnU-Net v2 fork bundled in this repo (CTA-to-time–vessel map generator), SimpleElastix (registration), and other dependencies.
- **`ctp-totalseg`** — an environment containing only [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) and its own pinned `nnunetv2`. It is kept separate because TotalSegmentator's PyPI `nnunetv2` would otherwise collide with the custom `Skeleton-recall` fork used everywhere else.

> **Note:** nnDetection's official releases are only tested against PyTorch 1.x (its README states PyTorch 2.0+ is not supported), while `Skeleton-recall` requires `torch>=2.1.2`. The steps below install nnDetection against PyTorch 2.x anyway, overriding its pinned dependencies. This combination has been built and smoke-tested end to end (CUDA extension compiles and runs, `box_iou_np`/`load_pickle` import correctly) on Python 3.10 + torch 2.5.1+cu118 + an RTX 4090 — but nnDetection's own Lightning-based training/inference pipeline (`nndet_train`, `nndet_predict`) was **not** exercised, only the lightweight `nndet.io` / `nndet.core.boxes` utilities this repo's scripts actually import.

### 1. Main environment (`ctp-postprocess`)

```bash
conda create -n ctp-postprocess python=3.10
conda activate ctp-postprocess

# --- PyTorch (CUDA 11.8 build; adjust to your driver/CUDA toolkit) ---
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu118

# --- Build tools needed to compile nnDetection's CUDA extensions ---
conda install -y -c conda-forge gxx_linux-64 ninja
conda install -y -c nvidia/label/cuda-11.8.0 cuda-toolkit

# --- nnDetection (public) ---
# Clone this to LOCAL disk, not to a network-mounted path (e.g. a CIFS/NFS share
# such as /media/E132-Projekte). nnDetection compiles a CUDA extension (nndet/_C...
# .so) and editable installs load it directly from the clone directory; network
# mounts here are commonly exported with the `noexec` flag (check with `mount`),
# which makes the OS refuse to execute/mmap that shared object at import time.
git clone https://github.com/MIC-DKFZ/nnDetection.git
cd nnDetection

# `pip install -r requirements.txt` does NOT work here: it pins SimpleITK<2.1.0,
# which has no Python 3.10 wheel and fails to build from source. Install the same
# dependencies by hand instead, using nnDetection's own pinned pytorch_lightning/
# torchmetrics (its Lightning-based trainer is untested against modern Lightning)
# and letting SimpleITK resolve later via Skeleton-recall + SimpleElastix below.
# `setuptools<81` is required too: old torchmetrics imports `pkg_resources`, which
# newer setuptools no longer ships.
pip install "setuptools<81" \
    "pytorch_lightning>=1.3.1,<=1.4.2" "torchmetrics>=0.7.0,<=0.7.3" \
    batchgenerators nnunet==1.7.1 scipy scikit-learn "scikit-image>=0.14" \
    "pandas>=0.8.1" nevergrad dicom2nifti medpy loguru "hydra-core>=1.1.0" \
    mlflow GitPython matplotlib seaborn "python-gdcm<3.0.26"
pip install hydra-core --upgrade --pre
pip install git+https://github.com/mibaumgartner/pytorch_model_summary.git

# --no-build-isolation: setup.py does `import torch` at build time, which fails
# in pip's isolated build sandbox. --no-deps: skip nnDetection's own pinned
# SimpleITK<2.1.0 requirement (already handled above/below).
FORCE_CUDA=1 CUDA_HOME=$CONDA_PREFIX TORCH_CUDA_ARCH_LIST="<your GPU's compute capability, e.g. 8.9 for RTX 4090>" \
    pip install -v -e . --no-build-isolation --no-deps
cd ..

# --- Skeleton-recall (custom nnU-Net v2 fork bundled in this repo for CTA-to-time-vessel map generator) ---
# Installs a modern SimpleITK>=2.2.1 fresh (nnDetection was installed with --no-deps
# above, so there's no conflict to resolve here).
pip install -e ./Skeleton-recall

# --- SimpleElastix build of SimpleITK, installed LAST so it overwrites the plain
#     SimpleITK install above (both packages share the same import namespace).
#     This also happens to satisfy nnDetection's own SimpleITK<2.1.0 pin. ---
pip install SimpleITK-SimpleElastix

# --- Remaining direct dependencies of the top-level scripts ---
pip install pandas scipy scikit-image scikit-learn scikit-fmm \
    nibabel pydicom tifffile matplotlib seaborn \
    loguru tqdm h5py joblib xmltodict PyYAML \
    iterative-stratification blosc2 einops
```

> **Notes on nnDetection compatibility:** `check_label.py`, `obtain_ids_tp_removal.py`, `nndet_scripts_/fpr_skeleton.py`, `auxiliary/fp_thresholds.py`, and `auxiliary/convert_val_preds.py` originally imported `from nndet.core.ops_np import box_iou_np`. That module path only existed in a private, internal fork — the public nnDetection installed above keeps this function at `nndet.core.boxes.ops_np` instead (same underlying math, confirmed by diffing the two), and the imports in this repo have been updated accordingly. Separately, this repo's own `nndet_scripts_/` folder is deliberately *not* named `nndet/` — Python has no `__init__.py` there, so it would otherwise shadow the installed `nndet` package as an implicit namespace package for any script run from the repo root (e.g. `python check_label.py`).

nnDetection also requires a few environment variables to be set:

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

# Pin torch to a CUDA build your driver actually supports BEFORE installing
# TotalSegmentator. Left to its own defaults, `pip install TotalSegmentator` can
# pull the latest torch (e.g. a cu130 build) even if your driver only supports an
# older CUDA runtime, which silently leaves torch.cuda.is_available() == False.
# Check your driver's max supported CUDA version with `nvidia-smi` and adjust the
# --index-url below (cu118/cu121/cu124/...) accordingly.
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

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
python postprocess_end2end.py --d /path/to/cta/data --m /path/to/generator/model --p /path/to/det_models/TaskXYZ/model_name/foldF/val_or_test_predictions --o /path/to/postprocessed/predictions --cfg nndet_scripts_/fpr_cfg.json --brain_cache /path/to/store/brain/segmentations --totalseg_bin /path/to/conda/envs/ctp-totalseg/bin/TotalSegmentator 

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
python register.py --cta /your/cta/folder --ctp /your/ctp/folder --out /registered/ctp/folder
```

# 2. CTP curation (start with the output registered CTP folder from Step 1). USE THE TOTALSEG ENV!!
```
conda activate ctp-totalseg
python preprocessing.py --folder /registered/ctp/folder --time /ctp/time/folder --out /curated/ctp/folder
```

# 3. Time-vessel map extraction (start with the curated CTP folder from Step 2). USE THE TOTALSEG ENV!!
```
conda activate ctp-totalseg
python extract_tta.py --ctp /curated/ctp/folder --cta /your/cta/folder --time /ctp/time/folder --skull /folder/where/to/store/brain/segmentations --tta /folder/with/time-vessel-maps

```

# 4. Development of CTA-to-time-vessel map generator
# 4.1 Data preparation

