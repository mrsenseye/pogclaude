# Point of Gaze (POG) Estimation Project

## Overview

This project focuses on deep learning-based point-of-gaze estimation using eye and face images. The codebase at `/media/a/saw/pog` contains multiple neural network architectures, training pipelines, and evaluation tools for predicting where a user is looking on a screen.

## Project Structure

```
/media/a/saw/pog/
├── models/backbone/          # Neural network backbones (ResNet, DenseNet, Swin, MobileNet, etc.)
├── malfoy/                   # ML training framework (trainer, model zoo, data loaders)
│   ├── trainer.py            # Main training loop with WandB integration
│   ├── model_manager/model_zoo/  # 17+ model implementations
│   └── data_loader/          # Dataset classes
├── dobby/                    # Data processing framework (preprocessing, scheduling)
├── Gaze/                     # Gaze estimation wrappers
├── POG_Wrapper/              # POG-specific utilities
├── checkpoints_*/            # 37 checkpoint directories (~115GB)
├── results/                  # Experiment outputs (~100GB)
├── pog_preprocessed/         # Preprocessed training data (train/val/test splits)
└── wandb/                    # Experiment tracking (40+ runs)
```

## Key Model Architectures

### 1. ItrackerMHSAv2 (Primary Model)
- **File**: `ItrackerMHSAv2.py`
- **Architecture**: 3-stream ResNet50 with Multi-Head Self-Attention
- **Inputs**: Face (448x448), Left eye (128x128), Right eye (128x128)
- **Features**: HybridGazeAttention (eyes self-attention + face-to-eyes cross-attention)

### 2. GazeTR (Transformer-based)
- **File**: `GazeTR.py`
- **Architecture**: ResNet18 + 6-layer Transformer Encoder
- **Features**: Learned positional embeddings, CLS token, facial landmarks input

### 3. Itracker_Transformer
- **File**: `Itracker_Transformer.py`
- **Architecture**: ResNet18 + Swin Transformer V2
- **Features**: Window-based attention, hierarchical processing

### 4. Mobile iTracker
- **File**: `mobile_iTracker.py`
- **Architecture**: MobileNetV3/EfficientNet + Latent Attention Fusion
- **Purpose**: Mobile deployment optimization

### 5. GazeNet
- **File**: `gazenet.py`
- **Architecture**: Configurable backbone (ResNet, ResNest, DenseNet, EfficientNet, Swin, DeiT)
- **Simple**: Face image → Backbone → FC layers → 2D gaze

## Data Format

### Inputs
- **Face images**: RGB, typically 224x224 to 512x512
- **Eye crops**: Left and right eye regions, 128x128 to 300x300
- **Facial landmarks**: 468-point MediaPipe FaceMesh coordinates
- **Auxiliary**: IMU data, face bounding boxes

### Outputs
- **Gaze coordinates**: 2D (x, y) values representing screen position
- **Units**: Pixels or centimeters depending on configuration

### Dataset Files
- `train_val_test_filepaths*.json` - Data split manifests (285MB+)
- `landmarks_data_facemesh_points.json` - Facial landmarks (1.5GB)
- `pog_dataset_complete.parquet` - Complete dataset in columnar format
- CSV files for calibration data, model predictions, and results

## Training

### Main Entry Points
```bash
# Primary training script
python train.py

# Specific architectures
python train_hrnet.py
python train_resnest.py
python train_mobile.py
python train_distillation.py
```

### Configuration
- **Config file**: `train_config_gaze.yml`
- **Key parameters**:
  - batch_size: 512
  - epochs: 50
  - learning_rate: 0.0005
  - optimizer: AdamW
  - warmup: 3 epochs

### Experiment Tracking
- WandB integration for metrics and visualization
- MLflow in `malfoy/model_manager/mlruns/`

## Data Loading

### Primary Data Loaders
- `upside_down_gaze_dataloader.py` - Main production loader with augmentation
- `gaze_dataloader.py` - Alternative with MediaPipe landmarks
- `GazeCapture_dataloader.py` - iTracker-style loading
- `malfoy/data_loader/gazeData.py` - Flexible multi-modal loader

### Augmentations
- Color jitter, blur, brightness adjustments
- Geometric transforms (translation)
- Upside-down variants for device orientation robustness
- CutMix for data mixing

## Pre-trained Models

| Model | File | Size | Description |
|-------|------|------|-------------|
| Best POG | `Best_POG_model_scripted.pt` | 338MB | Production model |
| GazeCapture trained | `GC_trained_POG_model_scripted_cuda0.pt` | 338MB | GazeCapture dataset |
| Device agnostic | `device_agnostic_traced_model.pt` | 263MB | Cross-device |
| Mobile iTracker | `Itracker_Mobile_Traced.pt` | 58MB | Mobile deployment |
| XGaze pretrained | `xgaze_resnet50_pretrained.pt` | 99MB | XGaze dataset |

## Dependencies

Key libraries:
- PyTorch 2.0+ with CUDA
- pytorch-lightning
- timm (PyTorch Image Models)
- MediaPipe (facial landmarks)
- wandb (experiment tracking)
- pandas, numpy, opencv

Environment files:
- `environment_hogwarts.yml`
- `hogwarts_env.yml`
- `requirements.txt`

## Evaluation

```bash
python model_eval.py
python model_eval_final.py
python gaze_model_test.py
```

Metrics are logged to CSV files and WandB dashboards.

## Mobile Deployment

- Core ML package: `Itracker_Mobile.mlpackage/`
- TorchScript exports for iOS/Android
- MediaPipe face landmarker: `face_landmarker.task`

## Directory Conventions

- `checkpoints_<model>_<variant>/` - Model checkpoints by architecture
- `results/` - Evaluation outputs and metrics
- `inference_output_*/` - Inference results (cm, pixels, viz)
- `*_preprocessed/` - Preprocessed data splits

## Common Tasks

### Train a new model
1. Prepare data in `pog_preprocessed/` format
2. Configure `train_config_gaze.yml`
3. Run `python train.py`

### Evaluate a checkpoint
1. Load checkpoint from `checkpoints_*/`
2. Run `python model_eval.py --checkpoint <path>`

### Export for mobile
1. Use `jit_exporter_pog.ipynb`
2. Convert to Core ML or TorchScript

### Add a new backbone
1. Add implementation to `models/backbone/`
2. Register in `malfoy/model_manager/model_zoo/__init__.py`
