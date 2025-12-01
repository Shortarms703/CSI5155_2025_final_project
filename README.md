# Vectorization of Convex Polygons Using a Multi-Stage CNN Architecture

This project explores the reconstruction of vector geometry from raster images using a deep learning approach. We developed a multi-stage pipeline consisting of a Convolutional Neural Network (CNN) classifier to determine the number of polygon vertices, followed by class-specific regression CNNs to predict the vertex coordinates of the polygon. To enable end-to-end training without ground-truth coordinate data, we utilized differentiable rasterization to compute image-based loss. We trained and evaluated this system on our own synthetic dataset of 4,800 images of convex polygons. We compared our approach against a baseline algorithm using Harris Corner Detection. Our results indicate that while the neural networks successfully learned to approximate the shapes, the algorithmic baseline outperformed the deep learning models in both classification accuracy and geometric precision.

## Project Structure

### `data_generation/`
Scripts for creating the polygon dataset:
- `generate_polygons.py`: Creates initial polygon images in SVG and PNG formats
- `clusterization/`: Reduces dataset size by clustering similar shapes and selecting representatives
- `augment_dataset.py`: Generates augmented train/test datasets with random transformations

See [`data_generation/README.md`](data_generation/README.md) for detailed usage instructions.

### `baseline/`
Non-ML baseline implementation using Harris Corner Detection:
- `non-ml/tuning_harris.py`: Hyperparameter tuning for Harris corner detector
- `non-ml/apply_harris_test.py`: Applies tuned parameters to test set
- `non-ml/trace_bitmap_with_harris.py`: Vertex detection using Harris corners

See [`baseline/README.md`](baseline/README.md) for usage instructions.

### `tuning/`
Neural network model training and hyperparameter tuning:
- `main_tuning.py`: Main training script for classifier and polygon models
- `classifier_config.py`: Hyperparameter grid for classifier model
- `polygon_config.py`: Hyperparameter grid for polygon regression models
- `config.py`: Shared configuration (paths, data splits)

### `evaluation/`
Scripts for evaluating model performance:
- `ground_truth_to_json.py`: Extracts polygon coordinates from SVG files
- `evaluate_classifier.py`: Compares classifier vs baseline classification accuracy
- `evaluate_hausdorff.py`: Computes Hausdorff distance between predicted and ground truth vertices
- `evaluate_mse.py`: Calculates pixel-wise MSE between rendered images

### `util/`
Utility scripts for data conversion:
- `convert_points_to_svg.py`: Converts vertex coordinates to SVG format
- `convert_svg_to_png.py`: Renders SVG files to PNG images

### `models/`
Saved trained model checkpoints (`.pth` files)

### `output/`
Results and evaluation outputs:
- `tuning_results_v2/`: Hyperparameter tuning results
- `evaluation_results/`: Performance metrics, confusion matrices, plots
- Ground truth and prediction JSON files

## Commands to submit jobs with new hyperparameter tuning scripts:

### Classifier

To run classifier model hyperparameter tuning:
```bash
sbatch ~/morningstar_scripts/run_tuning_classifier.sbatch
```

To view the logs of the classifier tuning job:
```bash
tail -f ~/output/tuning_results_v2/logs/tuning_<job_id>.out
```

### Polygon

To run polygon model hyperparameter tuning for 4-vertex polygons:
```bash
sbatch --export=ALL,MODEL_VERTEX_COUNT=4 ~/morningstar_scripts/run_tuning_polygon.sbatch
```
Change the `MODEL_VERTEX_COUNT` value to tune for different polygon vertex counts (3 to 6).

To view the logs of the classifier tuning job:
```bash
tail -f ~/output/tuning_results_v2/logs/poly_tuning_<job_id>.out
```

## Hyper parameter tuning (outside Morning Star):
Just run the sbatch script as a bash script:

./run_tuning_classifier.sbatch
./run_tuning_polygons_parallel.sbatch
./run_tuning_polygons_sequential.sbatch

Make them executable first (chmod +x)

## Morning Star

### Submiting jobs:

#### Dry run first (recommended)
sbatch_tuning.sh classifier true

#### Submit classifier tuning
sbatch_tuning.sh classifier

#### Submit parallel polygon tuning (8 jobs, one per vertex count)
sbatch_tuning.sh polygon

#### Submit sequential polygon tuning (1 job, all vertices)
sbatch_tuning.sh polygon-sequential

### Checks:

#### Check job status
squeue -u $USER

#### Check specific job
squeue -j <job_id>

#### View live output
tail -f logs/tuning_<job_id>.out

#### Cancel job
scancel <job_id>

#### Cancel all your jobs
scancel -u $USER

Adjust the --time, --mem, and --cpus-per-task based on your needs and experience with the first runs.


## AI Usage Acknowledgment
	
Per the course policy, we acknowledge the use of ChatGPT, Claude, and Gemini in some portions of this project. These tools were used for help with coding on tasks that are not the focus of the course: create visualizations, processing images for training the model, managing the pipeline, reading and writing files, dataset inspection and debugging errors. The deep learning implementation and initial drafts of the report were produced by the authors.

### Visualization
- `evaluation/evaluate_classifier.py` - Plotting confusion matrices and metrics comparisons

### Image Processing
- `data_generation/generate_polygons.py` - Creating polygon images
- `util/convert_svg_to_png.py` - Rendering SVG to PNG
- `util/convert_points_to_svg.py` - Converting coordinates to SVG
- `data_generation/augment_dataset.py` - Applying transformations to images

### Pipeline Management
- `data_generation/clusterization/cluster_shapes.py` - Clustering workflow, except PCA and KMeans implementation.
- `data_generation/clusterization/create_train_test_split.py` - Train/test splitting
- Refinement of SLURM/sbatch scripts for job submission on Morning Star.

## Reading/Writing Files
- `evaluation/ground_truth_to_json.py` - Extracting polygon data from SVG
- `evaluation/evaluate_hausdorff.py` - Loading and processing JSON files
- `evaluation/evaluate_mse.py` - File I/O for evaluation
- `data_generation/clusterization/create_train_test_split.py` - JSON file operations

## Dataset Inspection
- Scripts in `evaluation/` that load and analyze datasets
- Data loading utilities in `tuning/main_tuning.py` (DataLoader, Dataset classes)

## Debugging
- Error handling and debugging code across various files

---