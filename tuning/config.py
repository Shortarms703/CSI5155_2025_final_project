import os

IMAGE_DIR = "data/64x64/train/raster"
TUNING_RESULTS_FOLDER = "output/tuning_results_v2"
MODEL_SAVE_FOLDER = "models"
CLASSIFIER_MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_FOLDER, "best_classifier_model.pth")
POLYGON_MODEL_SAVE_PATH_LAMBDA = lambda x: os.path.join(MODEL_SAVE_FOLDER, f"best_polygon_{x}_model.pth")

CONTINUE_LAST_RUN = False
VERBOSE = True

train_fraction, val_fraction, test_fraction = 0.8, 0.1, 0.1
width = height = 64

# will ignore certain cells
DEBUG = True
