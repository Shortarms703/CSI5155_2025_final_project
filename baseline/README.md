# Baseline Methods

This directory contains non-ML baseline approaches for polygon vertex detection using Harris Corner Detection.

## Prerequisites

```bash
pip install opencv-python numpy scipy
```

## Running the Baseline

### 1. Hyperparameter Tuning

Tune Harris Corner Detection parameters for each polygon type:

```bash
cd baseline/non-ml
python tuning_harris.py
```

This will:
- Test different parameter combinations in parallel
- Find optimal parameters for each polygon type (3-6 vertices)
- Save results to `harris_tuning_results_per_class.json`

**Note**: Tuning uses 1000 samples per class and 30 parallel workers. Adjust `TUNING_SAMPLES_PER_CLASS` and `NUM_WORKERS` in the script if needed.

### 2. Run Vertex Detection

Apply the tuned Harris Corner Detector to detect polygon vertices:

```bash
python trace_bitmap_with_harris.py
```

This uses the best parameters found during tuning to detect corners in the test images.

## Output

- **Tuning Results**: `harris_tuning_results_per_class.json` contains:
  - Best parameters for each polygon type
  - Vertex count accuracy
  - Average Hausdorff distance
  - Top 10 parameter configurations

- **Metrics**:
  - Vertex Accuracy: Percentage of correct vertex count predictions
  - Hausdorff Distance: Geometric accuracy of detected vertices (in pixels)

## How It Works

The Harris Corner Detection baseline:
1. Converts images to grayscale and applies thresholding
2. Optionally applies morphological operations
3. Detects corners using Harris Corner Detection
4. Clusters nearby corners to merge duplicates
5. Orders vertices by angle from centroid