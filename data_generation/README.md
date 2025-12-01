# Dataset Generation

The data generation pipeline creates train/test datasets from clustered polygon images. Follow these steps:

## 1. Generate Initial Polygons

Create polygon images using `generate_polygons.py`:

```bash
python generate_polygons.py \
  --width 64 --height 64 \
  --sides 3-6 \
  --count 1200 \
  --output .
```

This generates 1200 images for each polygon type (3-6 vertices) in both vector (SVG) and raster (PNG) formats.

## 2. Cluster Shapes

Run `cluster_shapes.py` to reduce dataset size by grouping similar shapes:

```bash
cd clusterization
python cluster_shapes.py
```

This creates clustering JSON files (e.g., `polygon_3_clustering.json`) with 10% of original images selected as cluster representatives.

## 3. Create Train/Test Split

Split clustered data using `create_train_test_split.py`:

```bash
python create_train_test_split.py --test-split 0.1
```

This creates a JSON file mapping which cluster representatives go into train vs test sets.

## 4. Generate Augmented Dataset

Create the final augmented dataset using `augment_dataset.py`:

```bash
cd ..
python augment_dataset.py --augmentations 5 --width 64 --height 64
```

This applies random transformations (rotation, scaling, translation) to each cluster representative, generating multiple augmented versions for training/testing.

The output will be in `64x64/train/` and `64x64/test/` with both raster and vector formats.