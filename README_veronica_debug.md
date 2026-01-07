# POG Model Output Comparison - Debug Data

## Overview

This bucket contains a comparison between:
1. **Original parquet predictions** - POG predictions from the production pipeline
2. **New YOLO+POG predictions** - POG predictions from a simplified local pipeline using YOLO face/eye detection

The goal is to understand discrepancies between the two pipelines and debug why they produce different outputs.

## Data Sources

### Original Data (not in this bucket)
- `s3://senseye-ptsd/public/ptsd_ios/0014c300.../` - Session 1 videos and data
- `s3://senseye-ptsd/public/ptsd_ios/0132028e.../` - Session 2 videos and data

### Sessions
- **Session 1 (0014c300)**: 4 calibration videos (s1_v1 through s1_v4)
- **Session 2 (0132028e)**: 4 calibration videos (s2_v1 through s2_v4)

## Files in This Bucket

### New POG Outputs (`new_pog_outputs/`)
Raw model outputs from our YOLO+POG pipeline:
- `s1_v1_pog.parquet` through `s1_v4_pog.parquet` - Session 1 videos
- `s2_v1_pog.parquet` through `s2_v4_pog.parquet` - Session 2 videos

Each parquet contains columns:
- `frame` - Frame number (0-indexed)
- `timestamp` - Frame timestamp in seconds
- `pog_x_cm` - Raw model output X coordinate (cm)
- `pog_y_cm` - Raw model output Y coordinate (cm)

### Comparison Data
- `all_videos_merged.parquet` - All 8 videos merged with original and new predictions aligned by frame
  - `orig_x`, `orig_y` - Original pipeline predictions
  - `new_x`, `new_y` - New pipeline predictions
  - `diff_x`, `diff_y`, `diff_euclidean` - Differences
  - `video` - Video identifier (s1_v1, s1_v2, etc.)
  - `session` - Session identifier (s1, s2)

- `all_videos_summary.csv` - Summary statistics per video

### Visualizations
- `all_videos_timeseries.png` - X/Y coordinates over time for all videos
- `all_videos_2d_scatter.png` - 2D POG positions (original vs new)
- `all_videos_correlation.png` - Correlation plots (original vs new)
- `all_videos_histograms.png` - Distribution of X/Y values
- `all_videos_diff_dist.png` - Distribution of differences

### Analysis Notebook
- `pog_8video_comparison.ipynb` - Jupyter notebook with full analysis

## Key Findings

### Systematic Differences
1. **Y offset**: New predictions are consistently higher than original (+0.9 to +4.1 cm)
2. **X offset**: Varies by session - Session 1 shifted left, Session 2 shifted right
3. **Correlations**: Range from -0.27 to 0.80 depending on video/axis

### Per-Video Summary (Euclidean difference in cm)
| Video | X diff | Y diff | Eucl diff | X corr | Y corr |
|-------|--------|--------|-----------|--------|--------|
| s1_v1 | -0.81  | +1.04  | 2.34      | 0.31   | 0.48   |
| s1_v2 | -0.75  | +1.88  | 3.01      | 0.31   | 0.50   |
| s1_v3 | +0.03  | +0.87  | 3.48      | 0.28   | -0.23  |
| s1_v4 | -1.21  | +1.48  | 3.56      | 0.21   | -0.27  |
| s2_v1 | +0.61  | +4.12  | 4.68      | 0.60   | 0.46   |
| s2_v2 | +0.83  | +3.56  | 4.17      | 0.63   | 0.79   |
| s2_v3 | +0.74  | +4.02  | 4.46      | 0.64   | 0.72   |
| s2_v4 | +0.55  | +1.93  | 3.03      | 0.50   | 0.80   |

## Per-User Calibration

We tested whether a simple per-user calibration can improve the new pipeline predictions using the known calibration dot positions as ground truth.

### Calibration Method

The calibration dot positions (`displayed_cal_dot_at_x`, `displayed_cal_dot_at_y`) are converted from phone coordinates to cm using:

```python
def convert_dot_to_cm(device, x_pt, y_pt):
    ppi = 460
    # Device-specific screen parameters
    if "Pro Max" in device:
        screen_cm, x_offset, y_offset = 7.12, 2.9, 0.51
    elif "Pro" in device:
        screen_cm, x_offset, y_offset = 6.5, 2.5, 0.47
    # ... etc

    x_cm = ((x_pt * 3) / ppi) * 2.54
    y_cm = ((y_pt * 3) / ppi) * 2.54
    x_true = (screen_cm - x_cm) - x_offset
    y_true = y_cm - y_offset
    x_true2 = (-1 * x_true) + x_range  # flip X
    return x_true2, y_true
```

A linear affine transform is then fitted to map raw POG predictions to calibrated positions:
```
X_calibrated = a0 + a1*X_raw + a2*Y_raw
Y_calibrated = b0 + b1*X_raw + b2*Y_raw
```

### Calibration Methods Tested

We systematically tested multiple calibration approaches:
- **Linear/Quadratic/Cubic polynomial** regression
- **KNN** (K-Nearest Neighbors with distance weighting)
- **Per-dot calibration** (separate model for each dot position)
- **Ensemble** (average of KNN + Cubic + Per-dot)
- **Stable frames filtering** (using only middle 50% of each fixation period)
- **1, 2, or 3 training videos** to find optimal amount

### Recommended Calibration Method

**Linear affine transform** trained on stable frames from a single calibration video.

| Parameter | Value |
|-----------|-------|
| Algorithm | **Linear Ridge Regression** |
| Transform | `X_cal = a0 + a1*X_raw + a2*Y_raw` |
| Preprocessing | **Stable frames only** (middle 50% of each fixation) |
| Training Data | **Single calibration video** per user |
| Expected Improvement | **60-70%** |
| Expected Error | **~2.5-3.0 cm** (from ~7-8 cm raw) |

### Per-Video Results (Linear, Single Video Training)

**Session 1** - iPhone 13 Pro Max (trained on v1):
| Video | Raw Error | Calibrated | Improvement |
|-------|-----------|------------|-------------|
| v1 (train) | 7.62 cm | 2.17 cm | 72% |
| v2 (test) | 7.66 cm | 2.84 cm | 63% |
| v3 (test) | 7.80 cm | 2.50 cm | 68% |
| v4 (test) | 8.27 cm | 3.39 cm | 59% |
| **TEST avg** | **7.91 cm** | **2.91 cm** | **63%** |

**Session 2** - iPhone 16 Pro (trained on v1):
| Video | Raw Error | Calibrated | Improvement |
|-------|-----------|------------|-------------|
| v1 (train) | 7.33 cm | 2.72 cm | 63% |
| v2 (test) | 7.28 cm | 3.05 cm | 58% |
| v3 (test) | 7.22 cm | 2.77 cm | 62% |
| v4 (test) | 6.20 cm | 3.32 cm | 46% |
| **TEST avg** | **6.90 cm** | **3.05 cm** | **56%** |

### Implementation

```python
from sklearn.linear_model import Ridge

def filter_stable_frames(df, keep_ratio=0.5):
    """Keep middle 50% of each dot fixation"""
    filtered = []
    df = df.sort_values('frame').copy()
    df['dot_changed'] = (df['dot_x'].diff().abs() > 0.01) | (df['dot_y'].diff().abs() > 0.01)
    df['dot_group'] = df['dot_changed'].cumsum()

    for _, group in df.groupby('dot_group'):
        n = len(group)
        if n > 10:
            start = int(n * (1 - keep_ratio) / 2)
            end = int(n * (1 + keep_ratio) / 2)
            filtered.append(group.iloc[start:end])
    return pd.concat(filtered)

def calibrate_user(calibration_df):
    """Build linear calibration model from a calibration video"""
    stable = filter_stable_frames(calibration_df)

    X = stable[['pog_x_raw', 'pog_y_raw']].values  # model outputs
    y = stable[['dot_x_cm', 'dot_y_cm']].values     # ground truth

    model = Ridge(alpha=1.0)
    model.fit(X, y)
    return model

def apply_calibration(model, pog_x, pog_y):
    """Apply calibration to raw POG predictions"""
    return model.predict([[pog_x, pog_y]])[0]
```

### Calibration Files
- `calibration_coefficients.json` - Linear calibration parameters (legacy)
- `calibrated_predictions_best.parquet` - All predictions with calibration applied
- `calibration_summary_best.csv` - Summary statistics
- `pog_calibration_analysis.ipynb` - Interactive analysis notebook with systematic comparison

### Key Takeaways
1. **Linear Ridge regression** provides smooth, well-behaved calibration
2. **Stable frames filtering** (middle 50% of fixations) removes saccade noise
3. **Single calibration video** per user is sufficient
4. Final calibrated error: **~2.5-3.0 cm** (vs 7-8 cm raw) = **60-70% improvement**

## Pipeline Differences

### Original Pipeline
1. Face detection: YOLO
2. Eye segmentation: **v2_ensemble** (produces precise pupil/iris centers)
3. POG model: Best_POG_model_scripted.pt
4. Output: `pog_x_coord`/`pog_y_coord` (screen pixels) -> converted to `pog_x_cm`/`pog_y_cm`

### New Pipeline (this debug data)
1. Face detection: yolov8n-face-lindevs.pt
2. Eye detection: **yolo_eye_detector_best.pt** (bounding boxes, not segmentation)
3. POG model: Best_POG_model_scripted.pt (same model)
4. Output: Direct cm output from model

### Likely Cause of Mismatch
The POG model expects eye crops centered on actual pupil positions (from segmentation). Our YOLO bounding box crops are approximate, causing shifted gaze estimates.

## Model Details

The POG model (`Best_POG_model_scripted.pt`) is an ITrackerMHSA model that takes:
- Face crop: 448x448 pixels
- Left eye crop: 128x128 pixels
- Right eye crop: 128x128 pixels

Output: [x_cm, y_cm] - gaze point relative to camera position

## Models (in `models/` folder)

The following models are included for reproducing the new pipeline outputs:

| Model | Size | Description |
|-------|------|-------------|
| `Best_POG_model_scripted.pt` | 338 MB | ITrackerMHSA POG model (TorchScript) |
| `yolov8n-face-lindevs.pt` | 6 MB | YOLOv8n face detector |
| `yolo_eye_detector_best.pt` | 6 MB | YOLO eye detector |

## Inference Script

`run_local_pog_inference.py` - Complete inference pipeline script

### Requirements
```bash
pip install torch torchvision ultralytics opencv-python pandas pyarrow tqdm
```

### Usage
```bash
# Basic usage
python run_local_pog_inference.py --video path/to/video.mp4 --output output.parquet

# Full options
python run_local_pog_inference.py \
    --video path/to/video.mp4 \
    --output output.parquet \
    --pog-model Best_POG_model_scripted.pt \
    --face-model yolov8n-face-lindevs.pt \
    --eye-model yolo_eye_detector_best.pt \
    --device cuda  # or cpu
```

### Key Configuration
The script uses these crop sizes (matching model training):
- Face crop: 448x448 pixels
- Eye crop: 128x128 pixels
- Videos are rotated 180° for detection (iOS front camera is upside-down)

### Output Format
Parquet with columns:
- `frame` - Frame number (0-indexed)
- `timestamp` - Frame timestamp in seconds
- `pog_x_cm` - Raw model output X coordinate (cm)
- `pog_y_cm` - Raw model output Y coordinate (cm)

## Notes

- Videos are upside-down (iOS front camera) - we rotate 180 degrees for face/eye detection
- All videos are ~1850-1860 frames at 60fps (~30 seconds calibration sequences)
- The model outputs are in cm, representing gaze position relative to the device camera
