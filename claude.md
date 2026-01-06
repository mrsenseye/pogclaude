# Point of Gaze (POG) Estimation Project

## Overview

Deep learning-based point-of-gaze estimation using eye and face images. Predicts where a user is looking on a screen using multi-stream CNN architectures with attention mechanisms.

**Deployed Model**: ITracker with Multi-Head Self-Attention (v1), pretrained on GazeCapture and finetuned on upside-down data from annotation study.

## Datasets

### Annotation Study (Primary)
- **Image Root**: `/media/s/pogDatasetPrepped`
  - `faces/<user_id>/` - Face crops (448x448)
  - `left_eye_crops/<user_id>/` - Left eye crops (128x128)
  - `right_eye_crops/<user_id>/` - Right eye crops (128x128)
- **JSON Manifest**: `/media/a/saw/pog/merged_dataset_first_session.json`
  - Contains train/val/test splits with paths and metadata
  - Each entry has: `face_path`, `left_eye_path`, `right_eye_path`, `user_id`, `valid`

### GazeCapture (Pretraining)
- **Location**: `/data/datasets/prepped/gazecapture_prepped`
- **Loader**: `gaze_dataloader.py`

## Model Checkpoints

| Model | Path | Description |
|-------|------|-------------|
| Best POG (Deployed) | `/media/a/saw/pog/Best_POG_model_scripted.pt` | Production model |
| GazeCapture trained | `/media/a/saw/pog/GC_trained_POG_model_scripted_cuda0.pt` | Right-side-up images |

---

## ITrackerMHSA Architecture Family

### Shared Setup
- **Inputs**: Face (3x448x448) + Left eye (3x128x128) + Right eye (3x128x128)
- **Backbones**: Three ResNet-50 models
  - `face_backbone`: Standard ResNet-50
  - `leye_backbone`: ResNet-50 with dilated convolutions
  - `reye_backbone`: ResNet-50 with dilated convolutions
- **Features**: 2048-D per stream, forms three tokens [left_eye, right_eye, face]
- **Attention**: 4 heads x 256-dim keys
- **Output**: (x, y) gaze coordinates via MLP
- **Parameters**: ~25M

### v1 - MultiHeadAttBlock (Deployed)
**File**: `ItrackerMHSA.py`

Basic MHSA with linear Q/K/V projections and scaled dot-product attention. Includes residual connections, LayerNorm, and FFN.

**Output Layers**:
- `fc_eye`: Concatenated eyes (4096 → 128)
- `fc_face`: Face features (2048 → 128 → 128)
- `fc_out`: Final prediction (256 → 2)

### v2 - HybridGazeAttention
**File**: `ItrackerMHSAv2.py`

Adds learned positional encodings. First runs self-attention over eye tokens to create combined eye vector, then face token queries eye vector (cross-attention). Fast and lightweight with explicit eye→face guidance.

### v3 - TransformerEncoderBlock
**File**: `ItrackerMHSAv2.py`

Full transformer encoder over [left_eye, right_eye, face]. MHSA + residual + LayerNorm, then FFN + residual + LayerNorm. Richest token mixing, extensible with extra tokens (landmarks, depth), but heavier.

### ITrackerMultiHeadAttention (New Variant)
Custom MHSA with separate Wq/Wk/Wv, residual + LN, FFN + LN. Processes eyes and face with small MLPs. Compact one-layer encoder: cleaner than v1, lighter than v3.

---

## Training

### Main Script
```bash
python train.py
```
**File**: `train.py`

### Configuration
```python
learning_rate: 0.0001
batch_size: 64 (train), 128 (val/test)
epochs: 30
weight_decay: 0.05
optimizer: AdamW
scheduler: CosineAnnealingLR
loss: SmoothL1Loss
metric: RMSE
```

### Backbone Freezing Strategy
- **Epochs 1-5**: Frozen ResNet backbones (stabilizes early training)
- **Epochs 6+**: All layers unfrozen for fine-tuning

### Data Augmentation
- Horizontal flipping (50%) with coordinate adjustment
- Synchronized augmentation across face/eye crops
- Small Gaussian noise on training labels
- Translation, blur, color jitter, brightness/contrast (25% probability)

### Weighted Sampling
Computes inverse user frequency weights to balance representation and prevent bias toward users with more samples.

---

## Data Loaders

### Upside-Down Loader (Production)
**File**: `upside_down_gaze_dataloader.py`

- Face: 448x448 RGB, ImageNet normalization
- Eyes: 128x128 RGB each, ImageNet normalization
- Converts device pixels to physical distances (cm)
- `SynchronizedAugmentation` for consistent transforms

**Device Calibration**:
| Device | PPI | Screen Size |
|--------|-----|-------------|
| iPhone 15 Pro | 460 | 6.5cm |
| iPhone 13 Pro | 460 | 6.45cm |
| iPhone 12 Pro Max | 458 | 7.11cm |

### GazeCapture Loader
**File**: `gaze_dataloader.py`

For pretraining on GazeCapture dataset.

---

## Other Model Architectures

### GazeTR (Transformer)
**File**: `GazeTR.py`

ResNet18 backbone + 6-layer Transformer encoder (8 heads, 512 hidden dim). Uses CLS token, learnable position encoding for 7x7 patches. ~2M parameters.

**Inputs**: Face (224x224) + bounding box (142-dim) + IMU data

### HRNet (High-Resolution)
**File**: `Hrnet.py` | **Training**: `train_hrnet.py`

Multi-scale parallel branches maintaining high-resolution representations. 8 variants (w18_small to w64). Good for fine-grained spatial details.

### GazeNet (Backbone Factory)
**File**: `gazenet.py` | **Training**: `train_resnest.py`

Supports 25+ backbones: ResNet, DenseNet, EfficientNet, Swin, DeiT, ResNeSt, MobileNet. FC layers: 128 → 128 → 2.

### AffNet (Adaptive Normalization)
**File**: `Affnet.py`

Adaptive Group Normalization conditioned on face+rectangle features. Separate eye/face processing with SE-blocks.

### BotNet (Bottleneck Attention)
**File**: `botnet.py`

ResNet-50 with self-attention replacing layer4. Multi-head attention with positional embeddings.

---

## Pupillometry / Eye Segmentation

Eye segmentation using Mask2Former for pupil/iris detection and pupil-to-iris ratio (PIR) analysis.

### Location
```
/home/esha/Segmentation/
├── mask2former/
│   ├── model.py      # Mask2FormerFinetuner
│   ├── dataset.py    # SegmentationDataModule
│   ├── config.py     # Hyperparameters
│   └── losses.py     # Boundary, Tversky, Focal, CE+Dice
├── train.py
└── test.py
```

### Classes
0: Background, 1: Iris, 2: Pupil, 3: Sclera

### Data
- RGB eye crops: `/data/datasets/cv/mp_good_images`
- Training data: `/data/datasets/cv/Rgb_pupillometry_data`

### Usage
```bash
# Training
python /home/esha/Segmentation/train.py

# Testing
python /home/esha/Segmentation/test.py

# Inference with visualization
python Jai_inference.py --input_dir images/ --output_dir results/ --ckpt model.ckpt

# PIR analysis
python plr_testing.py --input_dir left_eyes/ --ckpt model.ckpt --output_csv results.csv
```

### Checkpoint
`mask2former_RGB_pupillometry_with_MrSenseyeFlickrFace_Ref85.ckpt`

---

## Project Structure

```
/media/a/saw/pog/
├── ItrackerMHSA.py           # Deployed model (v1)
├── ItrackerMHSAv2.py         # v2 and v3 variants
├── train.py                  # Main training script
├── upside_down_gaze_dataloader.py  # Production data loader
├── gaze_dataloader.py        # GazeCapture loader
├── GazeTR.py                 # Transformer model
├── Hrnet.py                  # High-resolution model
├── gazenet.py                # Backbone factory
├── Affnet.py                 # Adaptive normalization
├── botnet.py                 # Bottleneck attention
├── train_hrnet.py            # HRNet training
├── train_resnest.py          # ResNest training
├── models/backbone/          # Neural network backbones
├── malfoy/                   # ML training framework
├── dobby/                    # Data processing framework
├── checkpoints_*/            # Model checkpoints
├── results/                  # Experiment outputs
└── wandb/                    # Experiment tracking
```

---

## Dependencies

- PyTorch 2.0+ with CUDA
- PyTorch Lightning
- timm (PyTorch Image Models)
- MediaPipe (facial landmarks)
- WandB (experiment tracking)
- Transformers (HuggingFace) - for Mask2Former
- pandas, numpy, opencv

## Quick Reference

| Task | Command |
|------|---------|
| Train POG model | `python train.py` |
| Train HRNet | `python train_hrnet.py` |
| Train ResNest | `python train_resnest.py` |
| Train segmentation | `python /home/esha/Segmentation/train.py` |
| PIR analysis | `python plr_testing.py --input_dir eyes/ --ckpt model.ckpt` |
