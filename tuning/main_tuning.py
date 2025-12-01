# %% md
### Imports
# %%
import glob
import os
import re
import time

import matplotlib.pyplot as plt

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # make CUDA errors synchronous for debugging
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import layers, models, Input
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.patches as patches

from collections import Counter

import os

# Set random seeds for reproducibility
import random
import json

print(keras.__version__)

import numpy as np
from PIL import Image

# noinspection PyUnresolvedReferences
import pydiffvg

print("done imports")
# %% md
### Args
# %%
import argparse

parser = argparse.ArgumentParser(description='Model To train - Classifier or Polygon')

parser.add_argument('--model', type=str, choices=['polygon', 'classifier'])
parser.add_argument('--num-vertices', type=int,
                    help='Number of vertices for polygon model training (required for polygon training)')
parser.add_argument('--sbatch', type=bool, default=False, help='If running as sbatch job')

args, unknown = parser.parse_known_args()
POLYGON_VERTICES = args.num_vertices

if args.model == 'classifier':
    try:
        from tuning_v2.classifier_config import VERTEX_RANGE, BATCH_SIZE, NUM_EPOCHS, NUM_TRIALS, \
            CLASSIFIER_PARAMETER_GRID
    except ModuleNotFoundError as e:
        print("ModuleNotFoundError, assuming this is because the file is being run in a jupyter notebook.")
        os.chdir(os.path.dirname(os.getcwd()))
        print("New working directory:", os.getcwd())

        from tuning_v2.classifier_config import VERTEX_RANGE, BATCH_SIZE, NUM_EPOCHS, NUM_TRIALS, \
            CLASSIFIER_PARAMETER_GRID

        print("Import successful")

if args.model == 'polygon':
    from tuning_v2.polygon_config import VERTEX_RANGE, BATCH_SIZE, NUM_EPOCHS, NUM_TRIALS, POLYGON_PARAMETER_GRID

from tuning_v2.config import IMAGE_DIR, TUNING_RESULTS_FOLDER, CLASSIFIER_MODEL_SAVE_PATH, \
    POLYGON_MODEL_SAVE_PATH_LAMBDA, CONTINUE_LAST_RUN, VERBOSE, train_fraction, val_fraction, test_fraction, width, \
    height, DEBUG

num_classes = VERTEX_RANGE[1] - VERTEX_RANGE[0] + 1

if POLYGON_VERTICES is not None:
    if POLYGON_VERTICES < VERTEX_RANGE[0] or POLYGON_VERTICES > VERTEX_RANGE[1]:
        raise ValueError(f"num_vertices ({POLYGON_VERTICES}) must be in range {VERTEX_RANGE}")
    print(f"Polygon model training configured for {POLYGON_VERTICES} vertices")
elif args.model == 'polygon':
    print("Note: --num-vertices not specified. Polygon training will require this argument.")

print("Current arguments:")
for arg in vars(args):
    print(f"  {arg}: {getattr(args, arg)}")
print("Batch size:", BATCH_SIZE)
print()

SLURM_JOB_ID = os.environ.get('SLURM_JOB_ID', None)
if SLURM_JOB_ID:
    print(f"SLURM Job ID: {SLURM_JOB_ID}")
else:
    print("SLURM Job ID: Not available")

if args.sbatch:
    DEBUG = False
    print()
    print("Running in sbatch mode, setting DEBUG = False")
    print()
else:
    # FIXME THIS IS THE THING TO CHANGE FOR RUNNING LOCALLYish
    args.model = 'classifier'

input_shape = (width, height, 1)


# %%
# Clear CUDA cache and reset device state (run this if you get CUDA errors)
# NOTE: If this cell also fails with CUDA errors, restart the kernel!
def clear_cuda_cache():
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Reset CUDA device if there was a previous error
            try:
                torch.cuda.reset_peak_memory_stats()
            except:
                pass
            print("CUDA cache cleared successfully")
        else:
            print("CUDA not available")
    except Exception as e:
        print(f"ERROR: Could not clear CUDA cache. The CUDA device state is corrupted.")
        print(f"ERROR: You MUST restart the kernel to continue!")
        print(f"ERROR: Go to Kernel -> Restart Kernel (or Runtime -> Restart runtime in Colab)")
        print(f"ERROR Details: {e}")
        raise


clear_cuda_cache()


# %%
def set_seeds(seed=42):
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)


# %%
def rasterize(cmds, width, height):
    if width != height:
        raise Exception("Width and height must be the same")

    cmds = cmds * width

    polygon = pydiffvg.Polygon(points=cmds, is_closed=True)
    shapes = [polygon]

    shape_group = pydiffvg.ShapeGroup(
        shape_ids=torch.tensor([0]),
        fill_color=torch.tensor([0, 0, 0, 1.0])
    )
    shape_groups = [shape_group]

    scene_args = pydiffvg.RenderFunction.serialize_scene(
        width, height, shapes, shape_groups
    )

    background = torch.ones(width, height, 4, device=pydiffvg.get_device())

    render = pydiffvg.RenderFunction.apply
    img = render(width, height, 2, 2, 0, background, *scene_args)

    img_gray = img[:, :, :3].mean(dim=2, keepdim=True)

    return img_gray


# %% md
### Classifier Model Definition
# %%
def create_cls_model(input_shape, num_classes, conv_filters=[16, 32, 64], dense_units=64, dropout_rate=0.0,
                     activation='relu'):
    """
    Create classifier model with configurable hyperparameters.

    Args:
        input_shape: Input image shape (H, W, C)
        num_classes: Number of output classes
        conv_filters: List of filter counts for each Conv2D layer (default: [16, 32, 64])
        dense_units: Number of units in dense layer (default: 64)
        dropout_rate: Dropout rate, 0.0 = no dropout (default: 0.0)
        activation: Activation function name (default: 'relu')
    """
    raise NotImplementedError
    inputs = Input(shape=input_shape)
    x = inputs

    # CNN layers
    for filters in conv_filters:
        x = layers.Conv2D(filters, 3, activation=activation)(x)
        x = layers.MaxPooling2D(2)(x)
        if dropout_rate > 0.0:
            x = layers.Dropout(dropout_rate)(x)

    # Flatten and dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(dense_units, activation=activation)(x)
    if dropout_rate > 0.0:
        x = layers.Dropout(dropout_rate)(x)

    # Output layer
    class_outputs = layers.Dense(num_classes, activation='softmax', name='class')(x)

    model = models.Model(inputs, class_outputs)
    return model


# %%
import torch.nn as nn
import torch.nn.functional as F


class ClassifierCNN(nn.Module):
    def __init__(self, input_shape, num_classes, conv_filters=[16, 32, 64],
                 dense_units=64, dropout2d_rate_CNN=0.0, dropout_rate_FFN=0.0, activation='relu', batch_norm=False):
        super().__init__()

        H, W, C = input_shape
        in_channels = C

        act = getattr(F, activation)

        layers_list = []
        current_channels = in_channels
        for f in conv_filters:
            layers_list.append(nn.Conv2d(current_channels, f, kernel_size=3, padding=1))
            if batch_norm:
                layers_list.append(nn.BatchNorm2d(f))
            layers_list.append(nn.ReLU() if activation == "relu" else nn.SiLU())
            layers_list.append(nn.MaxPool2d(2))
            if dropout2d_rate_CNN > 0:
                layers_list.append(nn.Dropout2d(dropout2d_rate_CNN))
            current_channels = f

        self.conv = nn.Sequential(*layers_list)

        # compute output size
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, H, W)
            conv_out = self.conv(dummy).shape
            conv_flat = conv_out[1] * conv_out[2] * conv_out[3]

        self.fc1 = nn.Linear(conv_flat, dense_units)
        self.dropout = nn.Dropout(dropout_rate_FFN) if dropout_rate_FFN > 0 else nn.Identity()
        self.out = nn.Linear(dense_units, num_classes)

        self.activation = act

    def forward(self, x):
        # convert NHWC → NCHW
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        return self.out(x)


# %% md
### Polygon Model Definition
# %%
class PolygonCNN(nn.Module):
    def __init__(self, input_shape, num_vertices, conv_filters=[16, 32, 64], dense_units=64, dropout_rate=0.0,
                 activation='relu'):
        super().__init__()

        H, W, C = input_shape
        in_channels = C

        act = getattr(F, activation)

        layers_list = []
        current_channels = in_channels
        for f in conv_filters:
            layers_list.append(nn.Conv2d(current_channels, f, kernel_size=3, padding=1))
            # if batch_norm:
            #     layers_list.append(nn.BatchNorm2d(f))
            layers_list.append(nn.ReLU() if activation == "relu" else nn.SiLU())
            layers_list.append(nn.MaxPool2d(2))
            # if dropout2d_rate_CNN > 0:
            #     layers_list.append(nn.Dropout2d(dropout2d_rate_CNN))
            current_channels = f

        self.conv = nn.Sequential(*layers_list)

        # compute flattened conv output size
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, H, W)
            conv_out = self.conv(dummy).shape
            conv_flat = conv_out[1] * conv_out[2] * conv_out[3]

        self.fc1 = nn.Linear(conv_flat, dense_units)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

        # output: num_vertices * 2 coordinates
        self.out = nn.Linear(dense_units, num_vertices * 2)

        self.num_vertices = num_vertices
        self.activation = act

    def forward(self, x):
        # NHWC → NCHW
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)

        x = torch.flatten(x, 1)
        x = self.activation(self.fc1(x))
        x = self.dropout(x)

        # raw linear → reshape → sigmoid
        coords = self.out(x)
        coords = coords.view(-1, self.num_vertices, 2)
        coords = torch.sigmoid(coords)

        return coords


# %%
def create_poly_model(input_shape, num_vertices, conv_filters=[16, 32, 64], dense_units=64, dropout_rate=0.0,
                      activation='relu'):
    """
    Create polygon model with configurable hyperparameters.

    Args:
        input_shape: Input image shape (H, W, C)
        num_vertices: Number of vertices to predict
        conv_filters: List of filter counts for each Conv2D layer (default: [16, 32, 64])
        dense_units: Number of units in dense layer (default: 64)
        dropout_rate: Dropout rate, 0.0 = no dropout (default: 0.0)
        activation: Activation function name (default: 'relu')
    """
    set_seeds()
    model = PolygonCNN(input_shape, num_vertices, conv_filters, dense_units, dropout_rate, activation)

    return model

    # inputs = Input(shape=input_shape)
    # x = inputs
    #
    # # Convolutional layers
    # for filters in conv_filters:
    #     x = layers.Conv2D(filters, 3, activation=activation)(x)
    #     x = layers.MaxPooling2D(2)(x)
    #     if dropout_rate > 0.0:
    #         x = layers.Dropout(dropout_rate)(x)
    #
    # # Flatten and dense layers
    # x = layers.Flatten()(x)
    # x = layers.Dense(dense_units, activation=activation)(x)
    # if dropout_rate > 0.0:
    #     x = layers.Dropout(dropout_rate)(x)
    #
    # # Coordinates head (regression)
    # coord_vals = layers.Dense(num_vertices * 2)(x)
    # coord_outputs = layers.Reshape((num_vertices, 2))(coord_vals)
    # coord_outputs = layers.Activation("sigmoid", name="coords")(coord_outputs)
    #
    # model = models.Model(inputs, coord_outputs)
    # return model


# %%
class ImageDataset(Dataset):
    def __init__(self, image_dir, width, height, vertex_range=None):
        all_paths = glob.glob(os.path.join(image_dir, "polygon_*", "*.png"))

        self.image_paths = []
        self.width = width
        self.height = height
        self.vertex_range = vertex_range

        self.class_ids = []

        for path in all_paths:
            fname = os.path.basename(path)
            match = re.search(r"polygon_(\d+)_", fname)
            if match:
                vertex_count = int(match.group(1))

                # normalize label to 0-based index if vertex_range is provided
                if vertex_range is not None:
                    if not (vertex_range[0] <= vertex_count <= vertex_range[1]):
                        continue  # skip this image completely

                    normalized = vertex_count - vertex_range[0]
                    self.image_paths.append(path)
                    self.class_ids.append(normalized)
            else:
                raise ValueError(f"Filename {fname} doesn't match pattern polygon_<class>_*.png")

        print(f"Found {len(self.image_paths)} images in {image_dir} within vertex range {vertex_range}")

        if len(self.class_ids) > 0:
            min_label = min(self.class_ids)
            max_label = max(self.class_ids)
            print(f"Label range: [{min_label}, {max_label}], unique labels: {sorted(set(self.class_ids))}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # Open image, convert to grayscale, and resize
        img = Image.open(img_path).convert("L").resize((self.width, self.height))

        # Convert to numpy array and normalize to [0, 1]
        img_np = np.array(img).astype(np.float32) / 255.0

        # Add channel dimension: (H, W) -> (H, W, 1) - channels last for Keras
        img_np = np.expand_dims(img_np, axis=-1)

        img_tensor = torch.tensor(img_np, dtype=torch.float32)
        img_label = torch.tensor(self.class_ids[idx], dtype=torch.long)

        return img_tensor, img_label


# %%
def cls_validate_labels(label_batch, num_classes):
    if label_batch.min() < 0 or label_batch.max() >= num_classes:
        raise ValueError(
            f"Labels out of range: min={label_batch.min().item()}, max={label_batch.max().item()}, num_classes={num_classes}")


# %%
def cls_train_step(model, img_batch, label_batch, optimizer):
    model.train()

    pred_batch = model(img_batch)

    if not isinstance(label_batch, torch.LongTensor):
        label_batch = label_batch.long()

    num_classes = pred_batch.shape[1]
    cls_validate_labels(label_batch, num_classes)

    loss = torch.nn.functional.cross_entropy(pred_batch, label_batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


# %%
def get_cls_val_loss(model, val_loader):
    val_loss = 0.0
    model.eval()
    with torch.no_grad():
        for val_batch, label_batch in val_loader:
            val_batch = val_batch.to(device)
            if not isinstance(label_batch, torch.LongTensor):
                label_batch = label_batch.long()
            label_batch = label_batch.to(device)

            pred_batch = model(val_batch)

            num_classes = pred_batch.shape[1]
            cls_validate_labels(label_batch, num_classes)

            loss = torch.nn.functional.cross_entropy(pred_batch, label_batch)
            val_loss += loss

    return val_loss


# %%
def poly_train_step(model, target_img_batch, width, height, optimizer):
    model.train()

    coord_pred_batch = model(target_img_batch)

    batch_loss = 0.0
    batch_size = target_img_batch.shape[0]

    for i in range(batch_size):
        target_img = target_img_batch[i]  # Shape (H, W, 1)
        coord_pred = coord_pred_batch[i]  # Shape (num_commands, 2)

        pred_img = rasterize(coord_pred, width=width, height=height)
        loss = torch.mean((pred_img - target_img) ** 2)
        batch_loss += loss

    avg_loss = batch_loss / batch_size

    # Backprop
    optimizer.zero_grad()
    avg_loss.backward()
    optimizer.step()

    return avg_loss.item()


# %%
def get_poly_val_loss(model, val_loader, width, height):
    val_loss = 0.0
    model.eval()
    with torch.no_grad():
        for val_batch, _ in val_loader:  # DataLoader returns (img, label) tuples
            val_batch = val_batch.to(device)
            coord_pred_batch = model(val_batch)
            batch_size = val_batch.shape[0]
            batch_loss = 0.0
            for i in range(batch_size):
                target_img = val_batch[i]
                coord_pred = coord_pred_batch[i]
                pred_img = rasterize(coord_pred, width=width, height=height)
                loss = torch.mean((pred_img - target_img) ** 2)
                batch_loss += loss

            val_loss += batch_loss / batch_size
    return val_loss


# %%
pydiffvg.set_use_gpu(torch.cuda.is_available())
device = pydiffvg.get_device()
print(f"Using device: {device}")

dataset = ImageDataset(IMAGE_DIR, width, height, vertex_range=VERTEX_RANGE)
if len(dataset) == 0:
    raise Exception("No images found in " + IMAGE_DIR)

train_dataset, val_test_dataset = keras.utils.split_dataset(
    dataset, train_fraction, val_fraction + test_fraction, shuffle=True, seed=42
)

val_dataset, test_dataset = keras.utils.split_dataset(
    val_test_dataset, val_fraction / (val_fraction + test_fraction), test_fraction / (val_fraction + test_fraction),
    shuffle=True, seed=42
)


def seed_worker(worker_id):
    worker_seed = 42 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    worker_init_fn=seed_worker
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    worker_init_fn=seed_worker
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    worker_init_fn=seed_worker
)


# %%
# # Examine class labels in the dataloader (train_loader as example)
# class_labels = []
# for batch in train_loader:
#     # Assuming batch is (images, labels)
#     if isinstance(batch, (list, tuple)) and len(batch) > 1:
#         labels = batch[1]  # batch[1] should be labels
#         class_labels.extend(labels.cpu().numpy().tolist() if hasattr(labels, "cpu") else labels)
#     elif hasattr(batch, 'labels'):
#         class_labels.extend(batch.labels.cpu().numpy().tolist() if hasattr(batch.labels, "cpu") else batch.labels)
#     else:
#         print("Could not extract labels from batch structure:", type(batch))
#         break
#
# print("First 30 class labels in train_loader:", class_labels[:30])
# print("Unique class labels in train_loader:", sorted(set(class_labels)))
# from collections import Counter
#
# class_counts = Counter(class_labels)
#
# print("Number of instances per class in train_loader:")
# for class_id in sorted(class_counts.keys()):
#     print(f"Class {class_id}: {class_counts[class_id]}")
#
# # Show one of the images from class 6 in the training set
#
# # First, get indices of class 6 in train_dataset (assuming train_dataset has attribute class_ids)
# target_class_id = 6
# example_idx = None
#
# base_ds = train_dataset.dataset
# indices = train_dataset.indices
# for rel_idx, base_idx in enumerate(indices):
#     if base_ds.class_ids[base_idx] == target_class_id:
#         example_idx = rel_idx
#         break
#
# # Display the image
# if example_idx is not None:
#     img, label = train_dataset[example_idx]
#     # img may be a tensor or numpy array
#     import matplotlib.pyplot as plt
#     if hasattr(img, "numpy"):
#         img_disp = img.numpy()
#     else:
#         img_disp = img
#     # Squeeze if image is single-channel
#     if img_disp.shape[0] == 1 or img_disp.ndim == 3:
#         img_disp = img_disp.squeeze()
#     plt.figure(figsize=(4,4))
#     plt.imshow(img_disp, cmap='gray')
#     plt.title(f"Class {label} example from training set")
#     plt.axis('off')
#     plt.show()
# else:
#     print("Could not find an example for class 6 in train_dataset.")
# %% md
### Classifier Training Loop
# %% md
#### Model Training
# %%
def train_classifier_model(train_loader, cls_model, num_epochs, learning_rate, weight_decay, patience, verbose=False):
    optimizer = torch.optim.Adam(cls_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    print("Starting Classifier Model training...")
    training_start = time.time()

    training_loss_history = []
    validation_loss_history = []
    training_accuracy_history = []
    validation_accuracy_history = []

    for epoch in range(num_epochs):
        if verbose:
            print(f"--- Epoch {epoch + 1}/{num_epochs} ---")

        epoch_loss = 0.0
        epoch_start = time.time()

        if len(train_loader) == 0:
            raise Exception("No images found in train loader")

        for target_batch, label_batch in train_loader:
            target_batch = target_batch.to(device)
            if not isinstance(label_batch, torch.LongTensor):
                label_batch = label_batch.long()
            label_batch = label_batch.to(device)

            loss = cls_train_step(cls_model, target_batch, label_batch, optimizer)
            epoch_loss += loss

        avg_epoch_loss = epoch_loss / len(train_loader)
        training_loss_history.append(avg_epoch_loss)

        val_loss = get_cls_val_loss(cls_model, val_loader)
        avg_val_loss = val_loss / len(val_loader)
        validation_loss_history.append(avg_val_loss.item())

        training_accuracy = 0.0
        validation_accuracy = 0.0
        for img_batch, label_batch in train_loader:
            img_batch = img_batch.to(device)
            if not isinstance(label_batch, torch.LongTensor):
                label_batch = label_batch.long()
            label_batch = label_batch.to(device)

            pred_batch = cls_model(img_batch)
            _, predicted = torch.max(pred_batch, 1)
            training_accuracy += (predicted == label_batch).sum().item()
        for img_batch, label_batch in val_loader:
            img_batch = img_batch.to(device)
            if not isinstance(label_batch, torch.LongTensor):
                label_batch = label_batch.long()
            label_batch = label_batch.to(device)

            pred_batch = cls_model(img_batch)
            _, predicted = torch.max(pred_batch, 1)
            validation_accuracy += (predicted == label_batch).sum().item()

        training_accuracy /= len(train_loader.dataset)
        validation_accuracy /= len(val_loader.dataset)
        training_accuracy_history.append(training_accuracy)
        validation_accuracy_history.append(validation_accuracy)

        best_val_loss_epoch = validation_loss_history.index(min(validation_loss_history))
        epoch_since_min_val_loss = epoch - best_val_loss_epoch

        if verbose:
            print(f"Training Loss: {avg_epoch_loss:.6f} -",
                  f"Validation Loss: {avg_val_loss:.6f} -\n",
                  f"Training Accuracy: {(training_accuracy * 100):.4f} -",
                  f"Validation Accuracy: {(validation_accuracy * 100):.4f} -",
                  f"Time taken {round(time.time() - epoch_start, 2)} -",
                  f"Epochs since min val loss: {epoch_since_min_val_loss}")

        if epoch_since_min_val_loss > patience:
            print()
            print(f"No validation loss improvement since {best_val_loss_epoch}, stopping training")
            break

    clear_cuda_cache()

    print("Training complete, took " + str(round(time.time() - training_start, 2)))

    return training_loss_history, validation_loss_history, training_accuracy_history, validation_accuracy_history


# %% md
#### Hyperparameter Tuning
# %%
def hp_tuning_trial(input_shape, num_classes, num_epochs, trial, verbose):
    model = ClassifierCNN(input_shape,
                          num_classes,
                          conv_filters=trial["conv_filters"],
                          dense_units=trial["dense_units"],
                          dropout_rate_FFN=trial["dropout_rate"],
                          activation=trial["activation"]
                          )
    # model = create_cls_model(input_shape, num_classes,
    #                          conv_filters=trial["conv_filters"],
    #                          dense_units=trial["dense_units"],
    #                          dropout_rate=trial["dropout_rate"],
    #                          activation=trial["activation"]
    #                          )
    model.to(device)
    try:
        training_loss_history, validation_loss_history, training_accuracy_history, validation_accuracy_history = train_classifier_model(
            train_loader,
            model,
            num_epochs=num_epochs,
            learning_rate=trial["learning_rate"],
            weight_decay=trial["weight_decay"],
            patience=trial["patience"],
            verbose=verbose,
        )
    finally:
        del model
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        clear_cuda_cache()

    return validation_loss_history


# %%
def print_gpu_memory():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 3  # GB
        reserved = torch.cuda.memory_reserved() / 1024 ** 3  # GB
        max_allocated = torch.cuda.max_memory_allocated() / 1024 ** 3  # GB
        print(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB, Max: {max_allocated:.2f} GB")
        torch.cuda.reset_peak_memory_stats()


# %%
def run_classifier_hp_tuning(tuning_results_folder, parameter_grid, num_epochs, verbose):
    input_shape = (width, height, 1)
    classifier_tuning_results_folder = os.path.join(tuning_results_folder, 'classifier')

    initial_trial_index = 0

    if CONTINUE_LAST_RUN:
        # look in the folder, and find the most recent file
        results_files = glob.glob(
            os.path.join(classifier_tuning_results_folder, "time_classifier_hp_tuning_results_*.json"))
        if len(results_files) != 0:
            results_file = os.path.basename(max(results_files, key=os.path.getctime))
            with open(os.path.join(classifier_tuning_results_folder, results_file), 'r') as f:
                data = json.load(f)
                completed_trials = data['meta']["completed_trials"]
                total_trials = data['meta']["total_trials"]

                if completed_trials >= total_trials:
                    print(f"All {total_trials} trials already completed in {results_file}. Starting new tuning run.")
                elif total_trials != NUM_TRIALS:
                    print(
                        f"Warning: Mismatch in total trials. Previous run had {total_trials} trials, current run has {NUM_TRIALS} trials. Starting new tuning run.")
                else:
                    initial_trial_index = completed_trials
                    print(f"Continuing from last run, starting with trial_index = {initial_trial_index}")

    for trial_index in range(initial_trial_index, NUM_TRIALS):
        print(f"\n=== Hyperparameter Tuning Trial {trial_index + 1}/{NUM_TRIALS} ===")

        if trial_index == 0:
            job_id_suffix = f"job{SLURM_JOB_ID}_" if SLURM_JOB_ID else ""
            results_file = "time_classifier_hp_tuning_results_" + job_id_suffix + time.strftime(
                "%Y%m%d-%H%M%S") + ".json"
            data = {
                "meta": {
                    "total_trials": NUM_TRIALS,
                    "completed_trials": 0
                },
                "results": []
            }
            with open(os.path.join(classifier_tuning_results_folder, results_file), 'w') as f:
                json.dump(data, f, indent=4)
        else:
            # look in the folder, and find the most recent file
            results_files = glob.glob(
                os.path.join(classifier_tuning_results_folder, "time_classifier_hp_tuning_results_*.json"))
            if len(results_files) == 0:
                raise Exception(
                    "No results file found in " + classifier_tuning_results_folder + ". Expected one from trial 0.")
            results_file = os.path.basename(max(results_files, key=os.path.getctime))

        # randomly sample hyperparameters from parameter grid
        trial = {
            'learning_rate': random.choice(parameter_grid['learning_rate']),
            'conv_filters': random.choice(parameter_grid['conv_filters']),
            'dense_units': random.choice(parameter_grid['dense_units']),
            'dropout_rate': random.choice(parameter_grid['dropout_rate']),
            'activation': random.choice(parameter_grid['activation']),
            'weight_decay': random.choice(parameter_grid['weight_decay']),
            'patience': random.choice(parameter_grid['patience']),
        }
        print(f"Trial {trial_index + 1} hyperparameters: {trial}")

        val_loss_history = hp_tuning_trial(input_shape, num_classes, num_epochs, trial, verbose)

        best_val_loss = min(val_loss_history)
        best_val_loss_epoch = val_loss_history.index(best_val_loss) + 1

        with open(os.path.join(classifier_tuning_results_folder, results_file), 'r+') as f:
            data = json.load(f)
            data["meta"]["completed_trials"] = trial_index + 1
            data["results"].append({
                'trial_index': trial_index + 1,
                'hyperparameters': trial,
                'best_val_loss': best_val_loss,
                'best_epoch': best_val_loss_epoch,
            })
            f.seek(0)
            json.dump(data, f, indent=4)

        print(f"Trial {trial_index + 1} best validation loss: {best_val_loss:.6f} at epoch {best_val_loss_epoch}")

        print_gpu_memory()


# %%
if args.model == 'classifier':
    run_classifier_hp_tuning(TUNING_RESULTS_FOLDER, CLASSIFIER_PARAMETER_GRID, NUM_EPOCHS, VERBOSE)
# %% md
#### Evaluation on Test Set
# %% md
#### Train Classifier with Best Hyperparameters
# %%
if args.model == 'classifier':
    input_shape = (width, height, 1)
    # find json file with tuning results, most recent that is completed
    results_files = glob.glob(
        os.path.join(TUNING_RESULTS_FOLDER, 'classifier', "time_classifier_hp_tuning_results_*.json"))
    if len(results_files) == 0:
        raise Exception("No tuning results file found in " + os.path.join(TUNING_RESULTS_FOLDER, 'classifier'))

    results_file = os.path.basename(max(results_files, key=os.path.getctime))
    print(f"Results file found: {results_file}")

    best_hyperparameters = best_epoch = None
    with open(os.path.join(TUNING_RESULTS_FOLDER, 'classifier', results_file), 'r') as f:
        data = json.load(f)
        completed_trials = data['meta']["completed_trials"]
        total_trials = data['meta']["total_trials"]
        if completed_trials < total_trials:
            print(
                f"Warning: Tuning run in {results_file} not complete. Completed {completed_trials}/{total_trials} trials. Please complete tuning or delete incomplete tuning file")

        results = data['results']
        best_trial = sorted(results, key=lambda x: x['best_val_loss'])[0]
        best_hyperparameters = best_trial['hyperparameters']
        best_epoch = best_trial['best_epoch']

    if best_hyperparameters is None or best_epoch is None:
        raise Exception("Could not find best hyperparameters or best epoch in " + results_file)

    print(f"Best hyperparameters from tuning in {results_file}:")
    for k, v in best_hyperparameters.items():
        print(f"  {k}: {v}")

    print(f"Best epoch from tuning: {best_epoch}")

    set_seeds()
    cls_model = ClassifierCNN(input_shape,
                              num_classes,
                              conv_filters=best_hyperparameters['conv_filters'],
                              dense_units=best_hyperparameters['dense_units'],
                              dropout_rate_FFN=best_hyperparameters['dropout_rate'],
                              activation=best_hyperparameters['activation']
                              )

    cls_model.to(device)
    # Train classifier model with best hyperparameters for best_epoch epochs
    training_loss_history, validation_loss_history, training_accuracy_history, validation_accuracy_history = train_classifier_model(
        train_loader,
        cls_model,
        num_epochs=best_trial["best_epoch"],
        learning_rate=best_hyperparameters['learning_rate'],
        weight_decay=best_hyperparameters['weight_decay'],
        patience=best_hyperparameters['patience'],
        verbose=True
    )
# %% md
#### Save model
# %%
if args.model == 'classifier':
    # timestamp model path
    os.makedirs(os.path.dirname(CLASSIFIER_MODEL_SAVE_PATH), exist_ok=True)

    torch.save({
        "model_state": cls_model.state_dict(),
        "hyperparameters": best_hyperparameters,
        "best_epoch": best_epoch,
    }, CLASSIFIER_MODEL_SAVE_PATH)

    print(f"Saved model to {CLASSIFIER_MODEL_SAVE_PATH}")

# %% md
#### Load model
# %%
if args.model == 'classifier':
    loaded = torch.load(CLASSIFIER_MODEL_SAVE_PATH, map_location=device)

    loaded_hparams = loaded["hyperparameters"]

    try:
        set_seeds()
        model_loaded = ClassifierCNN(input_shape,
                                     num_classes,
                                     conv_filters=loaded_hparams["conv_filters"],
                                     dense_units=loaded_hparams["dense_units"],
                                     dropout_rate_FFN=loaded_hparams["dropout_rate"],
                                     activation=loaded_hparams["activation"],
                                     )

        model_loaded.load_state_dict(loaded["model_state"])
        model_loaded.to(device)
        model_loaded.eval()

        print("Model loaded and ready.")
    except RuntimeError as e:
        print("Error loading model")
# %% md
#### Evaluate Classifier on Test Set
# %%
if DEBUG:
    # set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    print_gpu_memory()
    clear_cuda_cache()
    print(torch.cuda.memory_allocated(), torch.cuda.memory_reserved())
    # print mem size of all variables
# %%
# TRYING TO FIX CLASSIFIER MODEL

if DEBUG and args.model == 'classifier':
    best_hyperparameters = {
        'learning_rate': 0.0001,
        'weight_decay': 0.00005,
        'batch_norm': True,
        'conv_filters': [32, 48, 48, 64],
        'dense_units': 64,
        'dropout2d_rate_CNN': 0.1,
        'dropout_rate_FFN': 0.5,
        'activation': 'relu',
        'patience': 25
    }
    best_epoch = 200
# %%
if DEBUG and args.model == 'classifier':
    # cls_model = create_cls_model(
    #     input_shape,
    #     num_classes,
    #     conv_filters=best_hyperparameters['conv_filters'],
    #     dense_units=best_hyperparameters['dense_units'],
    #     dropout_rate=best_hyperparameters['dropout_rate'],
    #     activation=best_hyperparameters['activation']
    # )
    set_seeds()
    cls_model = ClassifierCNN(input_shape,
                              num_classes,
                              conv_filters=best_hyperparameters["conv_filters"],
                              dense_units=best_hyperparameters["dense_units"],
                              dropout2d_rate_CNN=best_hyperparameters["dropout2d_rate_CNN"],
                              dropout_rate_FFN=best_hyperparameters["dropout_rate_FFN"],
                              activation=best_hyperparameters["activation"],
                              batch_norm=best_hyperparameters["batch_norm"]
                              )

    cls_model.to(device)
    training_loss_history, validation_loss_history, training_accuracy_history, validation_accuracy_history = train_classifier_model(
        train_loader,
        cls_model,
        num_epochs=best_epoch,
        learning_rate=best_hyperparameters['learning_rate'],
        weight_decay=best_hyperparameters["weight_decay"],
        patience=best_hyperparameters["patience"],
        verbose=True
    )
# %%
if DEBUG and args.model == 'classifier':
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ------- LOSS -------
    ax1.plot(training_loss_history, label='Training Loss')
    ax1.plot(validation_loss_history, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training vs Validation Loss')
    ax1.legend()
    ax1.set_xlim(0, 200)
    ax1.grid(True)

    # ------- ACCURACY -------
    ax2.plot(training_accuracy_history, label='Training Accuracy')
    ax2.plot(validation_accuracy_history, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training vs Validation Accuracy')
    ax2.legend()
    ax2.set_xlim(0, 200)
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
    print("best_hyperparameters:")
    for k, v in best_hyperparameters.items():
        print(f"  {k}: {v}")
    print()
    print("Final validation accuracy: ", validation_accuracy_history[-1])
    print("Best validation accuracy: ", max(validation_accuracy_history), "occurred at epoch",
          validation_accuracy_history.index(max(validation_accuracy_history)))
# %%
if args.model == 'classifier':
    # Evaluate classifier model on test set
    cls_model.eval()
    test_loss = 0.0
    all_predictions = []
    all_labels = []
    correct = 0
    total = 0

    print("Evaluating classifier model on test set...")
    with torch.no_grad():
        for img_batch, label_batch in test_loader:
            # Move data to device
            img_batch = img_batch.to(device)
            if not isinstance(label_batch, torch.LongTensor):
                label_batch = label_batch.long()
            label_batch = label_batch.to(device)

            # Get predictions
            pred_batch = cls_model(img_batch)

            # Calculate loss
            loss = torch.nn.functional.cross_entropy(pred_batch, label_batch)
            test_loss += loss.item()

            # Get predicted classes
            _, predicted = torch.max(pred_batch, 1)

            # Store predictions and labels
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(label_batch.cpu().numpy())

            # Calculate accuracy
            total += label_batch.size(0)
            correct += (predicted == label_batch).sum().item()

    # Calculate average test loss
    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = 100 * correct / total

    print(f"\n{'=' * 50}")
    print(f"Classifier Test Set Evaluation Results")
    print(f"{'=' * 50}")
    print(f"Average Test Loss: {avg_test_loss:.6f}")
    print(f"Test Accuracy: {test_accuracy:.2f}% ({correct}/{total})")
    print(f"{'=' * 50}\n")

    # Convert to numpy arrays for sklearn
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Calculate confusion matrix
    class_names = [f"{i + VERTEX_RANGE[0]} vertices" for i in range(num_classes)]
    cm = confusion_matrix(all_labels, all_predictions)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Classifier Test Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions,
                                target_names=class_names, digits=4))

    # Calculate per-class accuracy
    print("\nPer-Class Accuracy:")
    for i in range(num_classes):
        class_mask = all_labels == i
        if class_mask.sum() > 0:
            class_correct = (all_predictions[class_mask] == i).sum()
            class_total = class_mask.sum()
            class_acc = 100 * class_correct / class_total
            print(f"  {class_names[i]}: {class_acc:.2f}% ({class_correct}/{class_total})")
        else:
            print(f"  {class_names[i]}: No samples in test set")

    # Check data distribution
    print("\n" + "=" * 50)
    print("Data Distribution Analysis")
    print("=" * 50)

    label_counts = Counter(all_labels)
    print("\nTest set distribution:")
    for i in range(num_classes):
        count = label_counts.get(i, 0)
        percentage = 100 * count / len(all_labels) if len(all_labels) > 0 else 0
        print(f"  {class_names[i]}: {count} samples ({percentage:.2f}%)")

    # Check training data distribution if available
    print("\nTraining set distribution:")
    train_label_counts = Counter()
    for img_batch, label_batch in train_loader:
        train_label_counts.update(label_batch.cpu().numpy())
    total_train = sum(train_label_counts.values())
    for i in range(num_classes):
        count = train_label_counts.get(i, 0)
        percentage = 100 * count / total_train if total_train > 0 else 0
        print(f"  {class_names[i]}: {count} samples ({percentage:.2f}%)")

    # show all the test images, their proper label, and what they were classified as
    # ---------------------------------------------------------
    # SHOW ALL TEST IMAGES WITH TRUE + PREDICTED LABELS
    # ---------------------------------------------------------
    if DEBUG:
        print("\nDisplaying all test images with true and predicted labels...\n")

        # Make sure the model is in eval mode
        cls_model.eval()

        all_imgs = []
        all_true = []
        all_pred = []
        all_conf = []

        cls_model.eval()
        with torch.no_grad():
            for img_batch, label_batch in test_loader:
                img_batch = img_batch.to(device)
                label_batch = label_batch.long().to(device)

                logits = cls_model(img_batch)
                probs = torch.softmax(logits, dim=1)
                conf_vals, pred_classes = torch.max(probs, dim=1)

                all_imgs.append(img_batch.cpu())
                all_true.append(label_batch.cpu())
                all_pred.append(pred_classes.cpu())
                all_conf.append(conf_vals.cpu())

        # Concatenate everything
        all_imgs = torch.cat(all_imgs, dim=0)
        all_true = torch.cat(all_true, dim=0).numpy()
        all_pred = torch.cat(all_pred, dim=0).numpy()
        all_conf = torch.cat(all_conf, dim=0).numpy()

        sorted_idx = np.argsort(all_conf)

        # Number of images
        n = all_imgs.shape[0]


        def format_img_for_display(img):
            """
            Accepts torch tensor (various shapes) and returns a numpy array
            in H x W or H x W x C format suitable for plt.imshow.
            """

            # Convert to CPU + numpy first
            if isinstance(img, torch.Tensor):
                img = img.cpu()

            # Case 1: (C, H, W)
            if img.ndim == 3 and img.shape[0] in [1, 3]:
                img = img.permute(1, 2, 0)  # → (H, W, C)

            # Case 2: (H, W, C)
            elif img.ndim == 3 and img.shape[2] in [1, 3]:
                pass  # already correct

            # Case 3: (H, W)
            elif img.ndim == 2:
                pass  # grayscale OK

            # Case 4: (H, W, 1) → grayscale
            elif img.ndim == 3 and img.shape[2] == 1:
                img = img.squeeze(2)

            # Case 5: (1, H, W) → grayscale
            elif img.ndim == 3 and img.shape[0] == 1:
                img = img.squeeze(0)

            else:
                # Last-resort fallback: flatten extra dims until we get HWC or HW
                while img.ndim > 3:
                    img = img.squeeze(0)
                if img.ndim == 3 and img.shape[0] not in [1, 3]:
                    # Move the smallest dim to channel position
                    cdim = img.shape.index(min(img.shape))
                    img = img.permute(*(i for i in range(img.ndim) if i != cdim), cdim)

            return img.numpy()


        for rank, idx in enumerate(sorted_idx):
            img = format_img_for_display(all_imgs[idx])
            true_label = all_true[idx]
            pred_label = all_pred[idx]
            conf = all_conf[idx]

            plt.figure(figsize=(3, 3))
            plt.imshow(img, cmap="gray" if img.ndim == 2 else None)
            plt.axis("off")
            plt.title(
                f"Confidence: {conf:.2f}\n"
                f"True: {class_names[true_label]}\n"
                f"Pred: {class_names[pred_label]}",
                fontsize=10
            )
            plt.tight_layout()
            plt.show()


# %%
# def predict_vertex_count(image_path, model, device, vertex_range=[3, 10], width=64, height=64):
#     """
#     Run inference on a single image to predict vertex count.
#
#     Args:
#         image_path: Path to the PNG image file
#         model: Trained classifier model
#         device: Device to run inference on (cuda/cpu)
#         vertex_range: Range of vertex counts [min, max]
#         width: Image width (default 64)
#         height: Image height (default 64)
#
#     Returns:
#         predicted_vertex_count: The predicted number of vertices
#         probabilities: Dictionary of probabilities for each vertex count
#         confidence: Confidence score for the prediction
#     """
#     # Load and preprocess image
#     img = Image.open(image_path).convert("L")  # Convert to grayscale
#     img = img.resize((width, height))  # Resize to model input size
#
#     # Convert to numpy array and normalize to [0, 1]
#     img_np = np.array(img).astype(np.float32) / 255.0
#
#     # Add channel dimension: (H, W) -> (H, W, 1) - channels last for Keras
#     img_np = np.expand_dims(img_np, axis=-1)
#
#     # Add batch dimension: (H, W, 1) -> (1, H, W, 1)
#     img_np = np.expand_dims(img_np, axis=0)
#
#     # Convert to tensor and move to device
#     img_tensor = torch.tensor(img_np, dtype=torch.float32).to(device)
#
#     # Run inference
#     model.eval()
#     with torch.no_grad():
#         pred = model(img_tensor)
#         probs = torch.softmax(pred, dim=1)
#         _, predicted_class = torch.max(pred, 1)
#
#     # Convert to numpy
#     probs_np = probs[0].cpu().numpy()
#     predicted_class_idx = predicted_class[0].cpu().item()
#
#     # Map class index to vertex count
#     predicted_vertex_count = predicted_class_idx + vertex_range[0]
#     confidence = probs_np[predicted_class_idx]
#
#     # Create probability dictionary
#     num_classes = vertex_range[1] - vertex_range[0] + 1
#     probabilities = {}
#     for i in range(num_classes):
#         vertex_count = i + vertex_range[0]
#         probabilities[vertex_count] = float(probs_np[i])
#
#     return predicted_vertex_count, probabilities, confidence
#
#
# # Example usage:
# # Specify your image path here
# image_path = "new_pentagon.png"  # Change this to your image path
#
# # Run prediction
# predicted_vertices, probs, conf = predict_vertex_count(
#     image_path,
#     cls_model,
#     device,
#     vertex_range=VERTEX_RANGE,
#     width=width,
#     height=height
# )
#
# # Print results
# print(f"\n{'='*50}")
# print(f"Prediction Results")
# print(f"{'='*50}")
# print(f"Image: {image_path}")
# print(f"Predicted vertex count: {predicted_vertices}")
# print(f"Confidence: {conf:.4f} ({conf*100:.2f}%)")
# print(f"\nAll probabilities:")
# for vertex_count in sorted(probs.keys()):
#     print(f"  {vertex_count} vertices: {probs[vertex_count]:.4f} ({probs[vertex_count]*100:.2f}%)")
#
# # Visualize the image and prediction
# fig, axes = plt.subplots(1, 2, figsize=(10, 5))
#
# # Show original image
# img = Image.open(image_path).convert("L")
# axes[0].imshow(img, cmap='gray')
# axes[0].set_title(f'Original Image\n{os.path.basename(image_path)}')
# axes[0].axis('off')
#
# # Show preprocessed image (what the model sees)
# img_preprocessed = img.resize((width, height))
# axes[1].imshow(img_preprocessed, cmap='gray')
# axes[1].set_title(f'Preprocessed (64x64)\nPredicted: {predicted_vertices} vertices\nConfidence: {conf*100:.2f}%')
# axes[1].axis('off')
#
# plt.tight_layout()
# plt.show()
# %%
### Filtered Dataset for Polygon Models

class FilteredImageDataset(Dataset):
    """Wrapper dataset that filters another dataset by class_id"""

    def __init__(self, base_dataset, target_class_id):
        self.base_dataset = base_dataset
        self.target_class_id = target_class_id

        # Find all indices that match the target class_id
        # Handle both regular datasets and Subset objects (from train/val/test splits)
        self.filtered_indices = []
        for idx in range(len(base_dataset)):
            # Get the item to check its label
            _, label = base_dataset[idx]
            # Label is a tensor, so we need to get its value
            if isinstance(label, torch.Tensor):
                label_value = label.item()
            else:
                label_value = label

            if label_value == target_class_id:
                self.filtered_indices.append(idx)

        print(
            f"Filtered dataset: {len(self.filtered_indices)} samples for class_id {target_class_id} (vertex count {target_class_id + VERTEX_RANGE[0]})")

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        # Map filtered index to original dataset index
        original_idx = self.filtered_indices[idx]
        return self.base_dataset[original_idx]


# %% md
### Polygon Model Training Loop
# %% md
#### Model Training
# %%
def train_polygon_model(train_loader, val_loader, poly_model, num_epochs, learning_rate, width, height, verbose=False):
    optimizer = torch.optim.Adam(poly_model.parameters(), lr=learning_rate)

    print("Starting Polygon Model training...")
    training_start = time.time()

    training_loss_history = []
    validation_loss_history = []

    for epoch in range(num_epochs):
        if verbose:
            print(f"--- Epoch {epoch + 1}/{num_epochs} ---")

        epoch_loss = 0.0
        epoch_start = time.time()

        if len(train_loader) == 0:
            raise Exception("No images found in train loader")

        for target_batch, _ in train_loader:  # DataLoader returns (img, label) tuples
            target_batch = target_batch.to(device)

            loss = poly_train_step(poly_model, target_batch, width, height, optimizer)
            epoch_loss += loss

        avg_epoch_loss = epoch_loss / len(train_loader)
        training_loss_history.append(avg_epoch_loss)

        val_loss = get_poly_val_loss(poly_model, val_loader, width, height)

        avg_val_loss = val_loss / len(val_loader)
        validation_loss_history.append(avg_val_loss.item())

        if verbose:
            print(f"Epoch {epoch + 1} Average Validation Loss: {avg_val_loss:.6f}")

            print(
                f"Epoch {epoch + 1} Average Training Loss: {avg_epoch_loss:.6f} Time taken {round(time.time() - epoch_start, 2)}")

    clear_cuda_cache()

    print("Training complete, took " + str(round(time.time() - training_start, 2)))

    return training_loss_history, validation_loss_history


# %% md
#### Hyperparameter Tuning

# %%
def hp_tuning_trial_polygon(input_shape, num_vertices, num_epochs, trial, width, height, verbose):
    # Convert vertex number to class_id
    class_id = num_vertices - VERTEX_RANGE[0]

    # Filter datasets for this vertex number
    filtered_train_dataset = FilteredImageDataset(train_dataset, class_id)
    filtered_val_dataset = FilteredImageDataset(val_dataset, class_id)

    # Create filtered data loaders
    filtered_train_loader = DataLoader(
        filtered_train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        worker_init_fn=seed_worker
    )

    filtered_val_loader = DataLoader(
        filtered_val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        worker_init_fn=seed_worker
    )

    # Create model with trial hyperparameters
    model = create_poly_model(input_shape, num_vertices,
                              conv_filters=trial["conv_filters"],
                              dropout_rate=trial["dropout_rate"],
                              activation=trial["activation"]
                              )
    model.to(device)

    try:
        training_loss_history, validation_loss_history = train_polygon_model(
            filtered_train_loader,
            filtered_val_loader,
            model,
            num_epochs=num_epochs,
            learning_rate=trial["learning_rate"],
            width=width,
            height=height,
            verbose=verbose
        )
    finally:
        del model
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        clear_cuda_cache()

    return validation_loss_history


# %%
def run_polygon_hp_tuning(tuning_results_folder, parameter_grid, num_epochs, num_vertices, verbose):
    input_shape = (width, height, 1)
    # Check if POLYGON_VERTICES is specified
    if POLYGON_VERTICES is None:
        raise ValueError(
            "--num-vertices argument is required for polygon model training. Please specify it when running the notebook.")

    if num_vertices != POLYGON_VERTICES:
        raise ValueError(
            f"Mismatch: num_vertices parameter ({num_vertices}) does not match POLYGON_VERTICES ({POLYGON_VERTICES})")

    # Create results folder for this vertex number
    polygon_tuning_results_folder = os.path.join(tuning_results_folder, f'polygon_{num_vertices}')
    os.makedirs(polygon_tuning_results_folder, exist_ok=True)

    initial_trial_index = 0

    if CONTINUE_LAST_RUN:
        # look in the folder, and find the most recent file
        results_files = glob.glob(
            os.path.join(polygon_tuning_results_folder, f"time_polygon_{num_vertices}_hp_tuning_results_*.json"))
        if len(results_files) != 0:
            results_file = os.path.basename(max(results_files, key=os.path.getctime))
            with open(os.path.join(polygon_tuning_results_folder, results_file), 'r') as f:
                data = json.load(f)
                # Check if vertex number matches
                if 'meta' in data and 'num_vertices' in data['meta']:
                    stored_vertices = data['meta']['num_vertices']
                    if stored_vertices != num_vertices:
                        raise ValueError(
                            f"Vertex number mismatch: JSON file has {stored_vertices} vertices but current run specifies {num_vertices} vertices.")

                completed_trials = data['meta']["completed_trials"]
                total_trials = data['meta']["total_trials"]

                if completed_trials >= total_trials:
                    print(f"All {total_trials} trials already completed in {results_file}. Starting new tuning run.")
                elif total_trials != NUM_TRIALS:
                    print(
                        f"Warning: Mismatch in total trials. Previous run had {total_trials} trials, current run has {NUM_TRIALS} trials. Starting new tuning run.")
                else:
                    initial_trial_index = completed_trials
                    print(f"Continuing from last run, starting with trial_index = {initial_trial_index}")

    for trial_index in range(initial_trial_index, NUM_TRIALS):
        print(f"\n=== Hyperparameter Tuning Trial {trial_index + 1}/{NUM_TRIALS} for {num_vertices} vertices ===")

        if trial_index == 0:
            job_id_suffix = f"job{SLURM_JOB_ID}_" if SLURM_JOB_ID else ""
            results_file = f"time_polygon_{num_vertices}_hp_tuning_results_" + job_id_suffix + time.strftime(
                "%Y%m%d-%H%M%S") + ".json"
            data = {
                "meta": {
                    "total_trials": NUM_TRIALS,
                    "completed_trials": 0,
                    "num_vertices": num_vertices
                },
                "results": []
            }
            with open(os.path.join(polygon_tuning_results_folder, results_file), 'w') as f:
                json.dump(data, f, indent=4)
        else:
            # look in the folder, and find the most recent file
            results_files = glob.glob(
                os.path.join(polygon_tuning_results_folder, f"time_polygon_{num_vertices}_hp_tuning_results_*.json"))
            if len(results_files) == 0:
                raise Exception(f"No results file found in {polygon_tuning_results_folder}. Expected one from trial 0.")
            results_file = os.path.basename(max(results_files, key=os.path.getctime))

        # randomly sample hyperparameters from parameter grid
        trial = {
            'learning_rate': random.choice(parameter_grid['learning_rate']),
            'conv_filters': random.choice(parameter_grid['conv_filters']),
            'dropout_rate': random.choice(parameter_grid['dropout_rate']),
            'activation': random.choice(parameter_grid['activation']),
        }
        print(f"Trial {trial_index + 1} hyperparameters: {trial}")

        val_loss_history = hp_tuning_trial_polygon(input_shape, num_vertices, num_epochs, trial, width, height, verbose)

        best_val_loss = min(val_loss_history)
        best_val_loss_epoch = val_loss_history.index(best_val_loss) + 1

        with open(os.path.join(polygon_tuning_results_folder, results_file), 'r+') as f:
            data = json.load(f)
            data["meta"]["completed_trials"] = trial_index + 1
            data["results"].append({
                'trial_index': trial_index + 1,
                'hyperparameters': trial,
                'best_val_loss': best_val_loss,
                'best_epoch': best_val_loss_epoch,
            })
            f.seek(0)
            json.dump(data, f, indent=4)

        print(f"Trial {trial_index + 1} best validation loss: {best_val_loss:.6f} at epoch {best_val_loss_epoch}")

        print_gpu_memory()


# %%
# Run polygon hyperparameter tuning
# Note: POLYGON_VERTICES must be set via --num-vertices command line argument
if args.model == 'polygon':
    if POLYGON_VERTICES is not None:
        run_polygon_hp_tuning(TUNING_RESULTS_FOLDER, POLYGON_PARAMETER_GRID, NUM_EPOCHS, POLYGON_VERTICES, VERBOSE)
    else:
        print("Skipping polygon hyperparameter tuning: --num-vertices not specified")

# %% md
#### Train Polygon Model with Best Hyperparameters
# %%
if args.model == 'polygon':
    if POLYGON_VERTICES is None:
        raise ValueError(
            "--num-vertices argument is required for polygon model training. Please specify it when running the notebook.")

    num_vertices = POLYGON_VERTICES
    class_id = num_vertices - VERTEX_RANGE[0]

    # Find json file with tuning results, most recent that is completed
    polygon_tuning_results_folder = os.path.join(TUNING_RESULTS_FOLDER, f'polygon_{num_vertices}')
    results_files = glob.glob(
        os.path.join(polygon_tuning_results_folder, f"time_polygon_{num_vertices}_hp_tuning_results_*.json"))
    if len(results_files) == 0:
        raise Exception(f"No tuning results file found in {polygon_tuning_results_folder}")

    results_file = os.path.basename(max(results_files, key=os.path.getctime))

    best_hyperparameters = best_epoch = None
    with open(os.path.join(polygon_tuning_results_folder, results_file), 'r') as f:
        data = json.load(f)
        # Verify vertex number matches
        if 'meta' in data and 'num_vertices' in data['meta']:
            stored_vertices = data['meta']['num_vertices']
            if stored_vertices != num_vertices:
                raise ValueError(
                    f"Vertex number mismatch: JSON file has {stored_vertices} vertices but current run specifies {num_vertices} vertices.")

        completed_trials = data['meta']["completed_trials"]
        total_trials = data['meta']["total_trials"]
        if completed_trials < total_trials:
            print(
                f"Warning: Tuning run in {results_file} not complete. Completed {completed_trials}/{total_trials} trials. Please complete tuning or delete incomplete tuning file")

        results = data['results']
        best_trial = sorted(results, key=lambda x: x['best_val_loss'])[0]
        best_hyperparameters = best_trial['hyperparameters']
        best_epoch = best_trial['best_epoch']

    if best_hyperparameters is None or best_epoch is None:
        raise Exception(f"Could not find best hyperparameters or best epoch in {results_file}")

    print(f"Best hyperparameters from tuning in {results_file}:")
    for k, v in best_hyperparameters.items():
        print(f"  {k}: {v}")

    print(f"Best epoch from tuning: {best_epoch}")

    # Filter datasets for this vertex number
    filtered_train_dataset = FilteredImageDataset(train_dataset, class_id)
    filtered_val_dataset = FilteredImageDataset(val_dataset, class_id)

    # Create filtered data loaders
    filtered_train_loader = DataLoader(
        filtered_train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        worker_init_fn=seed_worker
    )

    filtered_val_loader = DataLoader(
        filtered_val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        worker_init_fn=seed_worker
    )

    # Create polygon model with best hyperparameters
    poly_model = create_poly_model(
        input_shape,
        num_vertices,
        conv_filters=best_hyperparameters['conv_filters'],
        dropout_rate=best_hyperparameters['dropout_rate'],
        activation=best_hyperparameters['activation']
    )

    poly_model.to(device)

    # Train polygon model with best hyperparameters
    training_loss_history, validation_loss_history = train_polygon_model(
        filtered_train_loader,
        filtered_val_loader,
        poly_model,
        num_epochs=best_epoch,  # Train for more epochs with best hyperparameters
        learning_rate=best_hyperparameters['learning_rate'],
        width=width,
        height=height,
        verbose=True
    )

# %% md
#### Save model

# %%
if args.model == 'polygon':
    torch.save({
        "model_state": poly_model.state_dict(),
        "hyperparameters": best_hyperparameters,
        "best_epoch": best_epoch,
        "num_vertices": num_vertices,
    }, POLYGON_MODEL_SAVE_PATH_LAMBDA(num_vertices))

    print(f"Saved model to {POLYGON_MODEL_SAVE_PATH_LAMBDA(num_vertices)}")

# %% md
#### Load model

# %%
if args.model == 'polygon':
    # Check if POLYGON_VERTICES is specified
    if POLYGON_VERTICES is None:
        raise ValueError(
            "--num-vertices argument is required for polygon model training. Please specify it when running the notebook.")

    num_vertices = POLYGON_VERTICES

    loaded = torch.load(POLYGON_MODEL_SAVE_PATH_LAMBDA(num_vertices), map_location=device)

    # Verify vertex number matches
    if 'num_vertices' in loaded:
        if loaded['num_vertices'] != num_vertices:
            raise ValueError(
                f"Vertex number mismatch: Model file has {loaded['num_vertices']} vertices but current run specifies {num_vertices} vertices.")

    loaded_hparams = loaded["hyperparameters"]
    loaded_num_vertices = loaded.get("num_vertices", num_vertices)

    try:
        model_loaded = create_poly_model(
            input_shape,
            loaded_num_vertices,
            conv_filters=loaded_hparams["conv_filters"],
            dropout_rate=loaded_hparams["dropout_rate"],
            activation=loaded_hparams["activation"],
        )

        model_loaded.load_state_dict(loaded["model_state"])
        model_loaded.to(device)
        model_loaded.eval()

        print("Model loaded and ready.")
    except RuntimeError as e:
        print("Error loading model")

# %% md
#### Evaluate Polygon on Test Set

# %%
if args.model == 'polygon':
    # Check if POLYGON_VERTICES is specified
    if POLYGON_VERTICES is None:
        raise ValueError(
            "--num-vertices argument is required for polygon model training. Please specify it when running the notebook.")

    num_vertices = POLYGON_VERTICES
    class_id = num_vertices - VERTEX_RANGE[0]

    # Filter test dataset for this vertex number
    filtered_test_dataset = FilteredImageDataset(test_dataset, class_id)

    # Create filtered test loader
    filtered_test_loader = DataLoader(
        filtered_test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        worker_init_fn=seed_worker
    )

    # Evaluate polygon model on test set
    poly_model.eval()
    test_loss = 0.0
    num_samples = 0

    print(f"Evaluating polygon model ({num_vertices} vertices) on test set...")
    with torch.no_grad():
        for target_batch, _ in filtered_test_loader:
            target_batch = target_batch.to(device)
            coord_pred_batch = poly_model(target_batch)
            batch_size = target_batch.shape[0]
            batch_loss = 0.0
            for i in range(batch_size):
                target_img = target_batch[i]
                coord_pred = coord_pred_batch[i]
                pred_img = rasterize(coord_pred, width=width, height=height)
                loss = torch.mean((pred_img - target_img) ** 2)
                batch_loss += loss
            test_loss += batch_loss / batch_size
            num_samples += batch_size

    # Calculate average test loss
    avg_test_loss = test_loss / len(filtered_test_loader)

    print(f"\n{'=' * 50}")
    print(f"Polygon Model ({num_vertices} vertices) Test Set Evaluation Results")
    print(f"{'=' * 50}")
    print(f"Average Test Loss: {avg_test_loss:.6f}")
    print(f"Number of test samples: {num_samples}")
    print(f"{'=' * 50}\n")

    # Visualize some predictions
    print("Visualizing sample predictions...")
    poly_model.eval()
    with torch.no_grad():
        for target_batch, _ in filtered_test_loader:
            target_batch = target_batch.to(device)
            coord_pred_batch = poly_model(target_batch)

            # Show first 3 samples from this batch
            num_show = min(3, target_batch.shape[0])
            for i in range(num_show):
                coord_pred = coord_pred_batch[i]
                pred_img = rasterize(coord_pred, width=width, height=height)
                loss = torch.mean((pred_img - target_batch[i]) ** 2)

                target_np = target_batch[i].cpu().detach().squeeze(-1).numpy()
                pred_np = pred_img.cpu().detach().squeeze(-1).numpy()

                fig, axes = plt.subplots(1, 3, figsize=(12, 4))

                axes[0].imshow(target_np, cmap='gray')
                axes[0].set_title('Target')
                axes[0].axis('off')

                axes[1].imshow(pred_np, cmap='gray')
                axes[1].set_title(f'Predicted (Loss: {loss.item():.4f})')
                axes[1].axis('off')

                # Add a third image rendering the points as a polygon
                axes[2].set_title('SVG Coordinates')
                points_np = coord_pred.detach().cpu().numpy()
                # Scale from [0,1] to [0,64] image coordinates
                points_scaled = points_np * 64

                polygon = patches.Polygon(points_scaled, closed=True,
                                          edgecolor='black', facecolor='black',
                                          linewidth=0)
                axes[2].set_xlim(0, 64)
                axes[2].set_ylim(0, 64)
                axes[2].invert_yaxis()
                axes[2].set_aspect('equal')
                axes[2].set_facecolor('white')
                axes[2].add_patch(polygon)
                axes[2].axis('off')

                plt.tight_layout()
                plt.show()

            break  # Only show first batch
# %% md
### Evaluation with tuned models
# %% md
#### Classifier
# %%
if DEBUG and args.model == 'classifier':
    results_file = "output/tuning_results_v2/classifier/time_classifier_hp_tuning_results_job2678_20251128-113516.json"

    with open(results_file, "r") as f:
        data = json.load(f)

    best = min(data["results"], key=lambda x: x["best_val_loss"])

    print()

    print("Best trial_id:", best["trial_index"])
    print("Best val_accuracy:", best["best_val_loss"])
    print("Hyperparameters:", best["hyperparameters"])
    print("Best epoch:", best["best_epoch"])

    best_hyperparameters = best["hyperparameters"]
    best_epoch = best["best_epoch"]
# %%
if DEBUG and args.model == 'classifier':
    set_seeds()
    cls_model = ClassifierCNN(input_shape,
                              num_classes,
                              conv_filters=best_hyperparameters["conv_filters"],
                              dense_units=best_hyperparameters["dense_units"],
                              dropout2d_rate_CNN=0,
                              dropout_rate_FFN=best_hyperparameters["dropout_rate"],
                              activation=best_hyperparameters["activation"],
                              batch_norm=0
                              )

    cls_model.to(device)
    training_loss_history, validation_loss_history, training_accuracy_history, validation_accuracy_history = train_classifier_model(
        train_loader,
        cls_model,
        num_epochs=best_epoch,
        learning_rate=best_hyperparameters['learning_rate'],
        weight_decay=best_hyperparameters["weight_decay"],
        patience=best_hyperparameters["patience"],
        verbose=True
    )
# %%
