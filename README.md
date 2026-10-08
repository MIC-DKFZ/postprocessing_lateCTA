# Anatomy- And Time-Based Post-Processing with CT Perfusion-Derived Time–Vessel Maps to Reduce Incorrect Occlusion Detections in Late-Phase CT Angiography
Copyright German Cancer Research Center (DKFZ) and contributors.

**Background/Objectives**: Computer-aided vessel occlusion detection for acute ischemic stroke is typically developed on early-phase CT angiography (CTA). In late-phase CTA, enhanced venous contrast can mimic arterial occlusions and lead to incorrect occlusion detections. To counteract this, we propose a constraint-based post-processing framework that leverages anatomical and temporal vascular information from CT perfusion (CTP) scans. 

**Methods**: Vascular anatomy and relative contrast-arrival times were encoded as time–vessel maps. Three complementary constraints worked on these maps to remove incorrect detections based on distance-to-vessel truncation points, contrast-arrival time, and spatial priors. Constraints were fixed on a 32-case late-phase development subset and applied to an external held-out cohort of late-phase CTA (N = 354). To enable application without CTP, an auxiliary CTA-to-time–vessel map generator was designed to estimate maps directly from CTA. 

**Results**: In the external cohort, the framework reduced incorrect occlusions per scan from 3.9 (95% CI 3.6–4.2) to 2.1 (95% CI 1.9–2.3), preserving occlusion-level sensitivity at 80% (95% CI 69–91). Patient-level AUROC was 83 for the baseline versus 88 with post-processing. Improvements were also observed in detectors exposed to late-phase CTA, where post-processing reduced incorrect occlusions per scan from 2.8 to 1.9. As only 51 occlusion-positive cases were present in the external cohort, confidence intervals were wide. Consequently, these estimates should be interpreted with caution. 

**Conclusions**: Anatomically and temporally informed post-processing reduced incorrect occlusion detections in late-phase CTA without measurable loss of sensitivity, potentially improving robustness against CTA phase changes in external institutions.

Please cite the following paper if you use this code: (https://www.mdpi.com/2075-4418/16/18/2972)


Martínez Mora, A.; Mojtahedi, M.; de Vries, L.; Baumgartner, M.; Kirchhoff, Y.; Zenk, M.; Eckstein, K.; Kächele, J.; Brugnara, G.; Bendszus, M.; et al. Anatomy- and Time-Based Post-Processing with CT Perfusion-Derived Time–Vessel Maps to Reduce Incorrect Occlusion Detections in Late-Phase CT Angiography. Diagnostics 2026, 16, 2972. https://doi.org/10.3390/diagnostics16182972


## Purpose
Post-processing repository for vessel occlusion detection in late-phase CTA, tested with nnDetection-like models. The code has an end-to-end script ("postprocessing_end2end.py") that segments the brain from CTA images, infers time-vessel maps from CTA images with a modified nnU-Net skeleton-recall based module, and accesses nnDetection predictions to remove implausible detected boxes based on:
- R1: boxes far from skeletonized vessel tips, based on a voxel radius threshold in "cfg/config_postprocess.json" (field "tip_radius") are removed.
- R2: boxes in late-enhanced vessels, with a relative time of arrival over a maximum threshold in cfg/config_postprocess.json (field "t_high_percentile") and under a minimum threshold (field "t_low_percentile") in the vicinity of the time-vessel map are removed.
- R3: boxes exceeding superior or posterior positions relative to the brain centroid, specified in cfg/config_postprocess.json (field "relative_brain_pos") are removed 
- R4: boxes with a too small or too large overall volume are removed (fields "min_volume" and "max_volume" in cfg/config_postprocess.json)
Resulting boxes are saved in an alternative folder of your choice.

The repository also offers step-by-step execution, following this roadmap. Numbers match the steps under [Step by step processing](#step-by-step-processing):

```text
     Raw CTP + CTA                                  CTA only
           │                                           │
           ▼                                           │
┌──────────────────────┐                               │
│ 1. CTP → CTA         │                               │
│    registration      │                               │
│    register.py       │                               │
└──────────┬───────────┘                               │
           ▼                                           │
┌──────────────────────┐                               │
│ 2. CTP curation      │                               │
│    preprocessing.py  │                               │
└──────────┬───────────┘                               ▼
           ▼                                ┌──────────────────────┐
┌──────────────────────┐      trains        │ 5. CTA → time–vessel │
│ 3. CTP time–vessel   ├───────────────────►│    map generator     │
│    maps              │                    │    (Skeleton-recall) │
│    extract_tta.py    │                    └──────────┬───────────┘
└──────────┬───────────┘                               │ predicted
           │           ┌─────────────────────┐         │ maps
           │           │ nnDetection         │         │
           │           │ occlusion boxes     │         │
           │           └──────────┬──────────┘         │
           │                      │                    │
           └──────────────────────┼────────────────────┘
                                  ▼
               ┌─────────────────────────────────────┐
               │ 4. Anatomical & temporal            │
               │    constraints R1–R4                │
               │    (+ TotalSegmentator brain mask)  │
               │    postprocess_end2end.py           │
               └──────────────────┬──────────────────┘
                                  ▼
               ┌─────────────────────────────────────┐
               │ Post-processed occlusion detections │
               └─────────────────────────────────────┘
```

Time–vessel maps reach the constraints (step 4) in one of two ways:
- **With CTP:** derived from the registered and curated CTP scan (steps 1–3) and passed with `--tta_maps`.
- **CTA only:** predicted from the CTA by the generator (step 5), which is trained on the CTP-derived maps.

`fpr_skeleton.py` applies the same constraints to precomputed time–vessel maps and brain masks, and is used during development.

## Installation

All of the code in this repository runs from **two conda environments**:

- **`ctp-postprocess`** — the main environment: PyTorch, [nnDetection](https://github.com/MIC-DKFZ/nnDetection) (occlusion detection), the `Skeleton-recall` nnU-Net v2 fork bundled in this repo (CTA-to-time–vessel map generator), SimpleElastix (registration), and other dependencies.
- **`ctp-totalseg`** — an environment containing only [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) and its own pinned `nnunetv2`. It is kept separate because TotalSegmentator's PyPI `nnunetv2` would otherwise collide with the custom `Skeleton-recall` fork used everywhere else for the CTA-to-time-vessel map generator.

> **Note:** nnDetection's official releases are only tested against PyTorch 1.x (its README states PyTorch 2.0+ is not supported), while `Skeleton-recall` requires `torch>=2.1.2`. The steps below install nnDetection against PyTorch 2.x anyway, overriding its pinned dependencies. This combination has been built and smoke-tested end to end (CUDA extension compiles and runs, `box_iou_np`/`load_pickle` import correctly) on Python 3.10 + torch 2.5.1+cu118 + an RTX 4090 — but nnDetection's own Lightning-based training/inference pipeline (`nndet_train`, `nndet_predict`) was **not** exercised, only the lightweight `nndet.io` / `nndet.core.boxes` utilities this repo's scripts actually import.

Run the commands below from the repository root. Both environments use `constraints.txt` to pin PyTorch: without it, any later `pip install` that re-resolves `torch` can pull the latest PyTorch with CUDA 13 libraries, which overwrite PyTorch 2.5.1's cuDNN and make every convolution fail with `cuDNN error: CUDNN_STATUS_NOT_INITIALIZED`. Setting `PIP_CONSTRAINT` in the environment applies the pins to every `pip install` run in it, including the editable installs below.

### 1. Main environment (`ctp-postprocess`)

```bash
conda create -n ctp-postprocess python=3.10
conda env config vars set -n ctp-postprocess PIP_CONSTRAINT="$(pwd)/constraints.txt"
conda activate ctp-postprocess

# --- PyTorch (CUDA 11.8 build; adjust to your driver/CUDA toolkit) ---
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu118

# --- Build tools needed to compile nnDetection's CUDA extensions ---
conda install -y -c conda-forge gxx_linux-64 ninja
conda install -y -c nvidia/label/cuda-11.8.0 cuda-toolkit

# --- Direct dependencies of nnDetection + this repo's top-level scripts ---
pip install -r requirements.txt
pip install hydra-core --upgrade --pre

# --- nnDetection. In the future to be updated to nnDetection v2, when this repository is out ---
git clone https://github.com/MIC-DKFZ/nnDetection.git
cd nnDetection

# Install nnDetection with argument --no-build-isolation:
FORCE_CUDA=1 CUDA_HOME=$CONDA_PREFIX TORCH_CUDA_ARCH_LIST="<your GPU's compute capability, e.g. 8.9 for RTX 4090>" \
    pip install -v -e . --no-build-isolation --no-deps
cd ..

# --- Skeleton-recall (custom nnU-Net v2 fork bundled in this repo for CTA-to-time-vessel map generator) ---
pip install -e ./Skeleton-recall

# --- SimpleElastix build of SimpleITK, for registration capabilities
pip install SimpleITK-SimpleElastix

# --- Check that PyTorch, CUDA and cuDNN work together (should print "cuDNN OK") ---
python -c "import torch; torch.nn.Conv3d(1, 1, 3).cuda()(torch.zeros(1, 1, 8, 8, 8, device='cuda')); print('cuDNN', torch.backends.cudnn.version(), 'OK')"
```

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
conda env config vars set -n ctp-totalseg PIP_CONSTRAINT="$(pwd)/constraints.txt"
conda activate ctp-totalseg

# Pin torch to a CUDA build your driver actually supports BEFORE installing
# TotalSegmentator
# Check your driver's max supported CUDA version with `nvidia-smi` and adjust the
# --index-url below (cu118/cu121/cu124/...) accordingly.
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements-totalseg.txt

# Should print "cuDNN OK"
python -c "import torch; torch.nn.Conv3d(1, 1, 3).cuda()(torch.zeros(1, 1, 8, 8, 8, device='cuda')); print('cuDNN', torch.backends.cudnn.version(), 'OK')"
```

#### Troubleshooting: `cuDNN error: CUDNN_STATUS_NOT_INITIALIZED`

This means CUDA 13 libraries were installed into an environment without the constraint and replaced PyTorch's cuDNN. In `ctp-postprocess`, remove them and restore the CUDA 11 files:

```bash
pip uninstall -y nvidia-cudnn-cu13 nvidia-nccl-cu13 nvidia-cusparselt-cu13 nvidia-nvshmem-cu13 \
  nvidia-cublas nvidia-cuda-runtime nvidia-cuda-cupti nvidia-cuda-nvrtc nvidia-cufft nvidia-cufile \
  nvidia-curand nvidia-cusolver nvidia-cusparse nvidia-nvjitlink nvidia-nvtx cuda-toolkit cuda-bindings cuda-pathfinder
pip install --force-reinstall --no-deps nvidia-cudnn-cu11==9.1.0.70 nvidia-nccl-cu11==2.21.5
```

`postprocess_end2end.py` calls TotalSegmentator as an external subprocess and does not need `ctp-totalseg` to be active — just point it at the environment's binary:

```bash
export TOTALSEG_BIN=/path/to/conda/envs/ctp-totalseg/bin/TotalSegmentator
# or pass --totalseg_bin /path/to/conda/envs/ctp-totalseg/bin/TotalSegmentator
```

## Main commands

All scripts skip cases whose output already exists, so an interrupted run can be resumed. Pass `--overwrite` to `register.py`, `preprocessing.py`, `extract_tta.py` or `postprocess_end2end.py` to process those cases again. In `postprocess_end2end.py` it recomputes brain masks and post-processed predictions; time–vessel maps in `--tta_maps` are always treated as inputs and never overwritten.

### Quick start: trained models available
If you already have a trained vessel occlusion detector in nnDetection and a trained CTA-to-time-vessel map generator, run the post-processing directly. To access our own trained nnDetection and generator models, contact us; we share them on reasonable request.
```
conda activate ctp-postprocess
python postprocess_end2end.py --d /path/to/cta/data --m /path/to/generator/model --p /path/to/det_models/TaskXYZ/model_name/foldF/val_or_test_predictions --o /path/to/postprocessed/predictions --cfg cfg/config_postprocess.json --brain_cache /path/to/store/brain/segmentations --totalseg_bin /path/to/conda/envs/ctp-totalseg/bin/TotalSegmentator 

```

### Full pipeline: data structure
To start from a folder with CTP data (one .nii.gz file for every CTP frame, up to NN frames) and a folder with CTA data, use this structure:
```text
ctp_folder/
├── case0/
│   ├── case0_t_00.nii.gz
│   ├── case0_t_01.nii.gz
│   ├── ...
│   └── case0_t_NN.nii.gz
├── ...
└── caseM/
    ├── caseM_t_00.nii.gz
    ├── caseM_t_01.nii.gz
    ├── ...
    └── caseM_t_NN.nii.gz

ctp_folder_with_time_information/
├── case0_AcquisitionDateTime.npy
├── ...
└── caseM_AcquisitionDateTime.npy

cta_folder/
├── case0.nii.gz
├── ...
└── caseM.nii.gz
```

## Step-by-step processing

### 1. CTP-to-CTA registration
```
conda activate ctp-postprocess
python register.py --cta /your/cta/folder --ctp /your/ctp/folder --out /registered/ctp/folder
```

### 2. CTP curation
Start with the registered CTP folder from step 1.

> **Note:** run this step in the `ctp-totalseg` environment, as it requires brain segmentation in TotalSegmentator.

It requires a nnU-Net vessel segmentation model, with a folder path in 'cfg/config_preprocess.json' ('mca_cpt'), to extract region of interest for AIF extraction. Right now this field is named as PLACEHOLDER

Contact us if you require the model. 
```
conda activate ctp-totalseg
python preprocessing.py --folder /registered/ctp/folder --time /ctp/time/folder --out /curated/ctp/folder
```

If CTP time files (.npy) require resampling after CTP image information curation, run:
```
conda activate ctp-postprocess
python utils/resample_time_info.py --time /ctp/time/folder --cta /your/cta/folder --out /resampled/ctp/time/folder
```


### 3. Time-vessel map extraction
Start with the curated CTP folder from step 2.

> **Note:** run this step in the `ctp-totalseg` environment.

Brain segmentations are outputted in the folder under 'skull' argument
If you have resampled your CTP time files, use the path /resampled/ctp/time/folder for the --time argument
```
conda activate ctp-totalseg
python extract_tta.py --ctp /curated/ctp/folder --cta /your/cta/folder --time /ctp/time/folder --skull /folder/where/to/store/brain/segmentations --tta /folder/with/time-vessel-maps

```

### 4. Post-processing constraints
If you have to use the CTA-to-time-vessel map generator model
```
conda activate ctp-postprocess
python postprocess_end2end.py --d /path/to/cta/data --m /path/to/generator/model --p /path/to/det_models/TaskXYZ/model_name/foldF/val_or_test_predictions --o /path/to/postprocessed/predictions --cfg cfg/config_postprocess.json --brain_cache /path/to/store/brain/segmentations --totalseg_bin /path/to/conda/envs/ctp-totalseg/bin/TotalSegmentator 

```

If you have already-derived time-vessel maps (named `<case>_0001.nii.gz`), pass their folder with `--tta_maps`. Maps found there are loaded instead of predicted, and missing ones are predicted and saved there. `--m` can be omitted if every case already has a map.
```
conda activate ctp-postprocess
python postprocess_end2end.py --d /path/to/cta/data --m /path/to/generator/model --tta_maps /folder/with/time-vessel-maps --p /path/to/det_models/TaskXYZ/model_name/foldF/val_or_test_predictions --o /path/to/postprocessed/predictions --cfg cfg/config_postprocess.json --brain_cache /path/to/store/brain/segmentations --totalseg_bin /path/to/conda/envs/ctp-totalseg/bin/TotalSegmentator 

```


### 5. Development of the CTA-to-time-vessel map generator

#### 5.1 Distance map computation
```
conda activate ctp-postprocess
python compute_distance.py --t /folder/with/time-vessel-maps --b /folder/where/to/store/brain/segmentations --o /folder/with/distance-maps
```

#### 5.2 Data preparation
```
conda activate ctp-postprocess
python utils/prepare_raw_data_generator.py --c /your/cta/folder --t /folder/with/time-vessel-maps --d /folder/with/distance-maps --b /folder/where/to/store/brain/segmentations --task task_name (nnUNet style, "DatasetXYZ") --o /folder/with/raw/generator/data
```

#### nnU-Net environment variables
nnU-Net also requires a few environment variables to be set:

```bash
export nnUNet_raw=/folder/with/raw/generator/data/raw_cropped # Folder with raw data processed by utils/prepare_raw_data_generator.py
export nnUNet_preprocessed=/folder/with/nnUNet/preprocessed/data # Folder where to store preprocessed generator data
export nnUNet_results=/folder/with/generator/model
```

#### 5.3 Preprocessing
```
conda activate ctp-postprocess
nnUNetv2_extract_fingerprint -d TASK_ID
nnUNetv2_plan_experiment -d TASK_ID -pl nnUNetPlannerResEncL -preprocessor_name MultiChannelSegPreprocessor
nnUNetv2_preprocess -d TASK_ID -plans_name nnUNetResEncUNetLPlans -c 3d_fullres
```

#### 5.4 Training
This assumes a separate test set exists.
```
conda activate ctp-postprocess
nnUNetv2_train TASK_ID 3d_fullres all -p nnUNetResEncUNetLPlans -tr nnUNetRegressionTrainer --use_compressed
```

#### 5.5 Inference
If test data is raw and has not been processed by prepare_raw_data_generator.py, prepare inference data by masking out non-brain voxels with -1024 HU
```
conda activate ctp-postprocess
python utils/obtain_brainProd_images.py --i /raw/CTA/folder --b /folder/where/to/store/brain/segmentations --o /folder/with/processed/inference/data
```
nnU-Net inference
```
conda activate ctp-postprocess
python Skeleton-recall/nnunetv2/utilities/predict_folder.py --d /folder/with/processed/inference/data --o /folder/with/output/predictions --m /folder/with/generator/model/DatasetXYZ/nnUNetRegressionTrainer__nnUNetResEncUNetLPlans__3d_fullres --b /folder/where/to/store/brain/segmentations --cfg cfg/params_generator.json

```

#### 5.6 Inference evaluation
```
conda activate ctp-postprocess
python Skeleton-recall/nnunetv2/evaluation/evaluate_predictions.py --folder_ref /folder/with/ground-truth/time-vessel-maps --folder_pred /folder/with/output/predictions --output_file /output/metric/file.json

```
