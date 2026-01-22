# RSNA 2025 Intracranial Aneurysm Detection - AI Coding Agent Instructions

## Project Overview
This is a **Kaggle competition submission** for intracranial aneurysm detection using 3D medical imaging (DICOM). The codebase implements an ensemble inference pipeline using EfficientNetV2 models trained on 32-channel volumetric CT/MR data. The primary deliverable is a Kaggle competition submission script that processes DICOM series and outputs multi-label classification predictions.

## Architecture & Data Flow

### Key Components

**1. DICOM Preprocessing Pipeline** (`DICOMPreprocessorKaggle` class)
- **Input**: Directory containing DICOM files (`.dcm`)
- **Processing steps**:
  1. Load and parse DICOM files using `pydicom`
  2. Extract 3D pixel arrays from single 3D DICOM OR stack multiple 2D DICOM slices
  3. Sort slices by z-position (ImagePositionPatient) to ensure correct ordering
  4. Apply modality-specific windowing: CT uses hardcoded 0-500 HU range; MR uses 1-99 percentile normalization
  5. Normalize to 0-255 uint8 with fallback to min-max if normalization fails
  6. Resize 3D volume to (32, 384, 384) using `scipy.ndimage.zoom` with linear interpolation
- **Output**: (32, 384, 384) numpy array

**Critical detail**: Slice ordering by z-position is essential—incorrect ordering breaks the spatial relationships the model expects.

**2. Model Ensemble** (5-fold cross-validation)
- Loads 5 pre-trained EfficientNetV2-S models from `/kaggle/input/rsna2025-effnetv2-32ch/`
- Each model expects 32-channel input (preprocessed volume)
- Predictions aggregated via simple averaging (equal weights) or custom weights if specified
- Output: 14 sigmoid probabilities for multi-label anatomical targets

**3. Inference Server Integration**
- Wraps the ensemble pipeline via `kaggle_evaluation.rsna_inference_server.RSNAInferenceServer`
- Runs either as local gateway (development) or remote competition server (production)
- Detects environment via `KAGGLE_IS_COMPETITION_RERUN` environment variable

## Project-Specific Conventions

### Configuration Management
- All settings centralized in `InferenceConfig` class (model names, sizes, paths, ensemble config)
- **Never hardcode paths outside CFG**—use `CFG.model_dir`, `CFG.target_shape`, etc.
- Label definitions in `LABEL_COLS` global must match training labels exactly (14 anatomical targets + "Aneurysm Present")

### Memory & Resource Management
- Kaggle environment has strict memory constraints
- **Always call `gc.collect()`** after processing series
- `process_dicom_series_safe()` enforces garbage collection via try-finally
- `predict()` function also clears CUDA cache and `/kaggle/shared` directory on exit
- **Never leave large arrays in memory between inference calls**

### Error Handling Pattern
- Graceful degradation: if DICOM processing fails, return **conservative predictions (0.1 for all classes)**
- Log errors with series ID for debugging
- Two-level error handling: `_predict_inner()` handles processing; `predict()` wraps with fallback + cleanup

### Data Shape Transformations
- **Preprocessing output**: (D, H, W) = (32, 384, 384)
- **Albumentations expects**: (H, W, C) format
- **Critical transform step**: `image.transpose(1, 2, 0)` converts (32, 384, 384) → (384, 384, 32) before augmentation
- **Model input**: (batch, channels=32, 384, 384)

## Critical Workflows

### Adding a New Model
1. Place checkpoint at: `/kaggle/input/rsna2025-effnetv2-32ch/tf_efficientnetv2_s.in21k_ft_in1k_fold{N}_best.pth`
2. Update `CFG.trn_fold` to include fold N
3. Model auto-loads in `load_models()` if it follows naming convention `{model_name}_fold{N}_best.pth`

### Modifying Preprocessing
- Always test with both single 3D DICOM and multi-2D DICOM series (different hospitals use different formats)
- Verify z-position sorting with `print([s['z_position'] for s in sorted_slices])`
- If adding new windowing logic, update both `_process_single_3d_dicom()` and `_process_multiple_2d_dicoms()`

### Adjusting Ensemble Weights
```python
CFG.ensemble_weights = {0: 1.2, 1: 1.0, 2: 1.0, 3: 0.8, 4: 1.0}  # Example custom weights
```
Leave as `None` for equal weighting (most common).

## Key Files & Their Responsibilities
- [rsna2025-efficientnetv2-32ch.ipynb](rsna2025-efficientnetv2-32ch.ipynb): Single file containing full pipeline—DICOM loading, preprocessing, ensemble inference, and Kaggle server integration
- Last cells (7-13): Exploratory analysis for NIfTI segmentation masks (not part of inference pipeline)

## External Dependencies
- **Medical imaging**: `pydicom`, `nibabel` (segmentation exploration only)
- **Deep learning**: `torch`, `timm` (EfficientNetV2), `albumentations` (augmentation)
- **Data handling**: `polars` (output format), `pandas`, `numpy`
- **Kaggle-specific**: `kaggle_evaluation.rsna_inference_server` (competition framework)

## Common Pitfalls & Solutions

| Issue | Solution |
|-------|----------|
| Model doesn't load | Check `/kaggle/input/rsna2025-effnetv2-32ch/` path and fold naming convention |
| Slice ordering incorrect | Verify `ImagePositionPatient` extraction and z-position sorting logic |
| Out of memory on inference | Ensure `gc.collect()` is called; check if large arrays leak between series |
| Predictions all ~0.5 | Check model is in `.eval()` mode and sigmoid is applied to raw logits |
| Shape mismatch errors | Verify transpose is applied before augmentation: (D,H,W) → (H,W,D) |

## Testing & Validation
- Local development: Run with `inference_server.run_local_gateway()` (no environment variable set)
- Submission: Script auto-detects Kaggle competition environment and runs `inference_server.serve()`
- Verify output: Expected return is Polars DataFrame with LABEL_COLS schema (14 float predictions per series)
