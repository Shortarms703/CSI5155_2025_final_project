# Vectorization of Convex Polygons Using a Multi-Stage CNN Architecture

This project explores the reconstruction of vector geometry from raster images using a deep learning approach. We developed a multi-stage pipeline consisting of a Convolutional Neural Network (CNN) classifier to determine the number of polygon vertices, followed by class-specific regression CNNs to predict the vertex coordinates of the polygon. To enable end-to-end training without ground-truth coordinate data, we utilized differentiable rasterization to compute image-based loss. We trained and evaluated this system on our own synthetic dataset of 4,800 images of convex polygons. We compared our approach against a baseline algorithm using Harris Corner Detection. Our results indicate that while the neural networks successfully learned to approximate the shapes, the algorithmic baseline outperformed the deep learning models in both classification accuracy and geometric precision.


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
