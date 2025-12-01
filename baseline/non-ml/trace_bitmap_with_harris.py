import os
import json
import cv2
import random
import numpy as np
from itertools import product
from scipy.spatial.distance import directed_hausdorff
from multiprocessing import Pool, cpu_count
from functools import partial

random.seed(42)
np.random.seed(42)

def load_ground_truth(filepath='evaluation/ground_truth_train_set.json'):
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    gt_dict = {}
    for item in data:
        base_name = item['filename'].replace('.svg', '')
        gt_dict[base_name] = item
    
    return gt_dict

def extract_polygon_vertices_harris(png_path, params):
    img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    
    blur_size = params['blur_size']
    if blur_size > 0:
        img = cv2.GaussianBlur(img, (blur_size, blur_size), 0)
    
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    
    if params['use_morph']:
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    binary_float = np.float32(binary)
    
    harris_response = cv2.cornerHarris(
        binary_float,
        blockSize=params['block_size'],
        ksize=params['ksize'],
        k=params['k']
    )
    
    if params['use_dilate']:
        harris_response = cv2.dilate(harris_response, None)
    
    threshold = params['threshold_factor'] * harris_response.max()
    corner_mask = harris_response > threshold
    
    corner_coords = np.argwhere(corner_mask)
    
    if len(corner_coords) == 0:
        return None
    
    corners = [[float(coord[1]), float(coord[0])] for coord in corner_coords]
    
    clustered_corners = cluster_corners(corners, params['min_distance'])
    
    if len(clustered_corners) == 0:
        return None
    
    centroid = np.mean(clustered_corners, axis=0)
    angles = [np.arctan2(p[1] - centroid[1], p[0] - centroid[0]) for p in clustered_corners]
    sorted_indices = np.argsort(angles)
    points = [clustered_corners[i] for i in sorted_indices]
    
    return points

def cluster_corners(corners, min_distance=5.0):
    if len(corners) == 0:
        return []
    
    corners = np.array(corners)
    clusters = []
    used = np.zeros(len(corners), dtype=bool)
    
    for i in range(len(corners)):
        if used[i]:
            continue
        
        distances = np.linalg.norm(corners - corners[i], axis=1)
        nearby = distances < min_distance
        used[nearby] = True
        
        cluster_points = corners[nearby]
        centroid = np.mean(cluster_points, axis=0)
        clusters.append(centroid.tolist())
    
    return clusters

def calculate_hausdorff(gt_points, det_points):
    if not gt_points or not det_points:
        return None
    
    gt_np = np.array(gt_points)
    det_np = np.array(det_points)
    
    forward = directed_hausdorff(gt_np, det_np)[0]
    backward = directed_hausdorff(det_np, gt_np)[0]
    
    return max(forward, backward)

def evaluate_params(params, train_files, ground_truth, max_samples=None):
    vertex_correct = 0
    hausdorff_distances = []
    total = 0
    
    if max_samples and len(train_files) > max_samples:
        train_files = random.sample(train_files, max_samples)

    for png_path, gt_key, expected_vertices in train_files:
        try:
            points = extract_polygon_vertices_harris(png_path, params)
            detected = len(points) if points else 0
            
            if detected == expected_vertices:
                vertex_correct += 1
            
            if points and gt_key in ground_truth:
                gt_points = ground_truth[gt_key]['points']
                hausdorff = calculate_hausdorff(gt_points, points)
                if hausdorff is not None:
                    hausdorff_distances.append(hausdorff)
            
            total += 1
            
        except Exception:
            total += 1
            continue
    
    vertex_accuracy = vertex_correct / total if total > 0 else 0
    avg_hausdorff = np.mean(hausdorff_distances) if hausdorff_distances else float('inf')
    
    return {
        'vertex_accuracy': vertex_accuracy,
        'avg_hausdorff': avg_hausdorff,
        'vertex_correct': vertex_correct,
        'total': total
    }

def evaluate_single_params(args):
    params, train_files, ground_truth, max_samples = args
    metrics = evaluate_params(params, train_files, ground_truth, max_samples)
    
    score = metrics['vertex_accuracy'] - 0.01 * min(metrics['avg_hausdorff'], 100)
    
    return {
        'params': params,
        'metrics': metrics,
        'score': score
    }

param_grid = {
    'block_size': [2, 3, 5],
    'ksize': [3, 5, 7],
    'k': [0.02, 0.04, 0.06],
    'threshold_factor': [0.005, 0.01, 0.02, 0.05],
    'min_distance': [3.0, 5.0, 7.0, 10.0],
    'blur_size': [3, 5, 7],
    'use_morph': [True, False],
    'use_dilate': [True, False]
}

print("Loading ground truth...")
ground_truth = load_ground_truth()

print("Collecting train files...")
base_dir = 'data/64x64/train/raster'

files_by_class = {}
for polygon_type in ['polygon_3', 'polygon_4', 'polygon_5', 'polygon_6']:
    png_dir = os.path.join(base_dir, polygon_type)
    expected_vertices = int(polygon_type.split('_')[1])
    
    class_files = []
    for png_file in sorted(os.listdir(png_dir)):
        if png_file.endswith('.png'):
            png_path = os.path.join(png_dir, png_file)
            gt_key = png_file.replace('.png', '')
            class_files.append((png_path, gt_key, expected_vertices))
    
    files_by_class[polygon_type] = class_files
    print(f"  {polygon_type}: {len(class_files)} files")

TUNING_SAMPLES_PER_CLASS = 1000
NUM_WORKERS = 30

all_results = {}

keys = list(param_grid.keys())
values = list(param_grid.values())
total_combinations = np.prod([len(v) for v in param_grid.values()])

print(f"\nTotal parameter combinations: {total_combinations}")
print(f"Using {TUNING_SAMPLES_PER_CLASS} samples per class for tuning")
print(f"Using {NUM_WORKERS} parallel workers (out of {cpu_count()} available)")

for polygon_type in ['polygon_3', 'polygon_4', 'polygon_5', 'polygon_6']:
    print("\n" + "="*60)
    print(f"TUNING FOR {polygon_type.upper()}")
    print("="*60)
    
    train_files = files_by_class[polygon_type]
    
    param_combinations = []
    for combination in product(*values):
        params = dict(zip(keys, combination))
        param_combinations.append((params, train_files, ground_truth, TUNING_SAMPLES_PER_CLASS))
    
    print(f"Evaluating {len(param_combinations)} parameter combinations in parallel...")
    
    with Pool(processes=NUM_WORKERS) as pool:
        results = pool.map(evaluate_single_params, param_combinations)
    
    best_result = max(results, key=lambda x: x['score'])
    best_params = best_result['params']
    best_metrics = best_result['metrics']
    best_score = best_result['score']
    
    print(f"\nBest Parameters for {polygon_type}:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    print(f"\nBest Metrics (on {TUNING_SAMPLES_PER_CLASS} samples):")
    print(f"  Vertex Accuracy: {best_metrics['vertex_accuracy']:.2%}")
    print(f"  Correct: {best_metrics['vertex_correct']}/{best_metrics['total']}")
    print(f"  Avg Hausdorff Distance: {best_metrics['avg_hausdorff']:.3f} pixels")
    
    print(f"\nEvaluating on full {polygon_type} dataset...")
    full_metrics = evaluate_params(best_params, train_files, ground_truth, max_samples=None)
    
    print(f"Full Dataset Results:")
    print(f"  Vertex Accuracy: {full_metrics['vertex_accuracy']:.2%}")
    print(f"  Correct: {full_metrics['vertex_correct']}/{full_metrics['total']}")
    print(f"  Avg Hausdorff Distance: {full_metrics['avg_hausdorff']:.3f} pixels")
    
    all_results[polygon_type] = {
        'best_params': best_params,
        'tuning_metrics': best_metrics,
        'full_dataset_metrics': full_metrics,
        'top_10_configurations': sorted(results, key=lambda x: x['score'], reverse=True)[:10]
    }

print("\n" + "="*60)
print("OVERALL SUMMARY")
print("="*60)

total_correct = 0
total_samples = 0
all_hausdorff = []

for polygon_type in ['polygon_3', 'polygon_4', 'polygon_5', 'polygon_6']:
    metrics = all_results[polygon_type]['full_dataset_metrics']
    total_correct += metrics['vertex_correct']
    total_samples += metrics['total']
    if metrics['avg_hausdorff'] != float('inf'):
        all_hausdorff.append(metrics['avg_hausdorff'])
    
    print(f"\n{polygon_type}:")
    print(f"  Accuracy: {metrics['vertex_accuracy']:.2%}")
    print(f"  Hausdorff: {metrics['avg_hausdorff']:.3f}")

overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
overall_hausdorff = np.mean(all_hausdorff) if all_hausdorff else float('inf')

print(f"\nOVERALL:")
print(f"  Accuracy: {overall_accuracy:.2%} ({total_correct}/{total_samples})")
print(f"  Avg Hausdorff: {overall_hausdorff:.3f}")

output_file = 'baseline/non-ml/harris_tuning_results_per_class.json'
with open(output_file, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\nResults saved to {output_file}")