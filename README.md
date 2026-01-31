# nnUNet Interactive Segmentation Scripts

Scripts for preparing neuron segmentation data and training nnUNet models with interactive prompts (8-channel input).


## Installation

```bash
# Create conda environment
conda env create -f environment.yml
conda activate nnunet

# Or use existing venv
source ./nnunet_venv/bin/activate
```

## Data Paths

Configure these paths before running:

```python
# Input: Single neuron zarr files
INPUT_DIR = "/path/to/single_neurons"

# Output: Synthesized combined data
SYNTHETIC_DIR = "/path/to/synthetic_data"

# nnUNet directories
nnUNet_raw = "/path/to/nnUNet_rawdata"
nnUNet_preprocessed = "/path/to/nnUNet_preprocessed"
nnUNet_results = "/path/to/nnUNet_results"
```

## Workflow

### Step 1: Synthesize Training Data

Combine single neurons into multi-instance volumes:

```bash
# Option A: Basic synthesis (2-6 neurons per volume)
python 1_synthesis_combined_data_hybrid.py

# Option B: Curriculum learning stages
python 1_stage2.py  # Separated neurons
python 1_stage3.py  # Closely packed neurons
```

### Step 2: Convert to nnUNet Format

Generate 8-channel interactive segmentation data:

```bash
# Basic version
python 2_data_nninteractive8C3.py

# Improved version with better negative prompts
python 2_data_nninteractive8C_negative_prompt.py
```

**8-Channel Format:**
| Channel | Name | Description |
|---------|------|-------------|
| 0 | image | Raw image |
| 1 | coarse_segmentation | Previous mask (60% probability) |
| 2 | bbox_pos | Bounding box positive (unused) |
| 3 | bbox_neg | Bounding box negative (unused) |
| 4 | point_pos | Positive point prompts |
| 5 | point_neg | Negative point prompts |
| 6 | scribble_pos | Scribble positive (unused) |
| 7 | scribble_neg | Scribble negative (unused) |

### Step 3: nnUNet Preprocessing

```bash
# Set environment variables
export nnUNet_raw="/path/to/nnUNet_rawdata"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"

# Plan and preprocess
nnUNetv2_plan_and_preprocess -d DATASET_ID -c 3d_fullres --verify_dataset_integrity
```

### Step 4: Training

```bash
# Single GPU
nnUNetv2_train DATASET_ID 3d_fullres 0

# Multi-GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 nnUNetv2_train DATASET_ID 3d_fullres 0 -num_gpus 4
```

### Step 5: Inference

```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_ID -c 3d_fullres -f 0
```

## Script Parameters

### 1_synthesis_combined_data_hybrid.py
- `TARGET_SIZE`: Output volume size (Z, Y, X)
- `N_NEURONS_MIN/MAX`: Neurons per volume (2-6)
- `TRAIN_SAMPLES/EVAL_SAMPLES`: Number of samples to generate

### 2_data_nninteractive8C_negative_prompt.py
- `number_of_samples`: Total samples to generate
- `min_voxels_per_inst`: Minimum neuron size
- `prev_mask_use_prob`: Probability of including previous mask (0.6)
- `prev_mask_coverage_range`: Coverage ratio range (0.1-0.7)
- `add_bg_noise`: Add background noise (True)
