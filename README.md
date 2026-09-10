# Anatomy- And Time-Based Post-Processing with CT Perfusion-Derived Time–Vessel Maps to Reduce Incorrect Occlusion Detections in Late-Phase CT Angiography
Copyright German Cancer Research Center (DKFZ) and contributors.

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
conda activate ctp-totalseg

# Pin torch to a CUDA build your driver actually supports BEFORE installing
# TotalSegmentator
# Check your driver's max supported CUDA version with `nvidia-smi` and adjust the
# --index-url below (cu118/cu121/cu124/...) accordingly.
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements-totalseg.txt
```

`postprocess_end2end.py` calls TotalSegmentator as an external subprocess and does not need `ctp-totalseg` to be active — just point it at the environment's binary:

```bash
export TOTALSEG_BIN=/path/to/conda/envs/ctp-totalseg/bin/TotalSegmentator
# or pass --totalseg_bin /path/to/conda/envs/ctp-totalseg/bin/TotalSegmentator
```

## Main commands

# If you already have a trained vessel occlusion detector in nnDetection and a trained CTA-to-time-vessel map generator
To access our own nnDetection and generator trained models, contact us for sharing at reasonable enquiry
```
conda activate ctp-postprocess
python postprocess_end2end.py --d /path/to/cta/data --m /path/to/generator/model --p /path/to/det_models/TaskXYZ/model_name/foldF/val_or_test_predictions --o /path/to/postprocessed/predictions --cfg fpr_cfg.json --brain_cache /path/to/store/brain/segmentations --totalseg_bin /path/to/conda/envs/ctp-totalseg/bin/TotalSegmentator 

```

# If you want to start from a folder with CTP data and a folder with CTA data. Required data structure:
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
   
# 1. CTP-to-CTA registration
```
conda activate ctp-postprocess
python register.py --cta /your/cta/folder --ctp /your/ctp/folder --out /registered/ctp/folder
```

# 2. CTP curation (start with the output registered CTP folder from Step 1). 
USE THE TOTALSEG ENV!!

It requires a nnU-Net vessel segmentation model, with a folder path in 'config.json' ('mca_cpt'). Right now this field is named as PLACEHOLDER

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


# 3. Time-vessel map extraction (start with the curated CTP folder from Step 2). 
USE THE TOTALSEG ENV!!
Brain segmentations are outputted in the folder under 'skull' argument
If you have resampled your CTP time files, use the path /resampled/ctp/time/folder for the --time argument
```
conda activate ctp-totalseg
python extract_tta.py --ctp /curated/ctp/folder --cta /your/cta/folder --time /ctp/time/folder --skull /folder/where/to/store/brain/segmentations --tta /folder/with/time-vessel-maps

```

# 4. Postprocessing constraints application
If you have to use the CTA-to-time-vessel map generator model
```
conda activate ctp-postprocess
python postprocess_end2end.py --d /path/to/cta/data --m /path/to/generator/model --p /path/to/det_models/TaskXYZ/model_name/foldF/val_or_test_predictions --o /path/to/postprocessed/predictions --cfg fpr_cfg.json --brain_cache /path/to/store/brain/segmentations --totalseg_bin /path/to/conda/envs/ctp-totalseg/bin/TotalSegmentator 

```

If you have already-derived time-vessel maps
```
conda activate ctp-postprocess
python postprocess_end2end.py --d /path/to/cta/data --m /path/to/generator/model --t /folder/with/time-vessel-maps --p /path/to/det_models/TaskXYZ/model_name/foldF/val_or_test_predictions --o /path/to/postprocessed/predictions --cfg fpr_cfg.json --brain_cache /path/to/store/brain/segmentations --totalseg_bin /path/to/conda/envs/ctp-totalseg/bin/TotalSegmentator 

```


# 5. Development of CTA-to-time-vessel map generator
# 5.1 Distance map computation
```
conda activate ctp-postprocess
python compute_distance.py --t /folder/with/time-vessel-maps --b /folder/where/to/store/brain/segmentations --o /folder/with/distance-maps
```

# 5.2 Data preparation
```
conda activate ctp-postprocess
python utils/prepare_raw_data_generator.py --c /your/cta/folder --t /folder/with/time-vessel-maps --d /folder/with/distance-maps --b /folder/where/to/store/brain/segmentations --task task_name (nnUNet style, "DatasetXYZ") --o /folder/with/raw/generator/data
```

# Adapted nnU-Net setup
nnU-Net also requires a few environment variables to be set:

```bash
export nnUNet_raw=/folder/with/raw/generator/data/raw_cropped # Folder with processed raw data by utils/prepare_raw_generator.py
export nnUNet_preprocessed=/folder/with/nnUNet/preprocessed/data # Folder where to store vessel occlusion detectors
export nnUNet_results=/folder/with/generator/model
```

# 5.3 Preprocessing
```
conda activate ctp-postprocess
nnUNetv2_extract_fingerprint -d TASK_ID
nnUNetv2_plan_experiment -d TASK_ID -pl nnUNetPlannerResEncL -preprocessor_name MultiChannelSegPreprocessor
nnUNetv2_preprocess -d TASK_ID -plans_name nnUNetResEncUNetLPlans -c 3d_fullres
```

# 5.4 Training (assuming some separate test set exists)
```
conda activate ctp-postprocess
nnUNetv2_train TASK_ID 3d_fullres all -p nnUNetResEncUNetLPlans -tr nnUNetRegressionTrainer --use_compressed
```

# 5.5 Inference
If test data is raw and has not been processed by prepare_raw_data_generator.py, prepare inference data by masking out non-brain voxels with -1024 HU
```
conda activate ctp-postprocess
python utils/obtain_brainProd_images.py --i /raw/CTA/folder --b /folder/where/to/store/brain/segmentations --o /folder/with/processed/inference/data
```
nnU-Net inference
```
conda activate ctp-postprocess
python Skeleton-Recall/nnunetv2/utilities/predict_folder.py --d /folder/with/processed/inference/data --o /folder/with/output/predictions --m /folder/with/generator/model/DatasetXYZ/nnUNetRegressionTrainer__nnUNetResEncUNetLPlans__3d_fullres --b /folder/where/to/store/brain/segmentations --cfg params_generator.json

```

# 5.6 Inference evaluation
```
conda activate ctp-postprocess
python Skeleton-recall/nnunetv2/evaluation/evaluate_predictions.py --folder_ref /folder/with/ground-truth/time-vessel-maps --folder_pred /folder/with/output/predictions --output_file /output/metric/file.json

```
