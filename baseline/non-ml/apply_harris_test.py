import os
import json
import cv2
import argparse
import numpy as np
from pathlib import Path

def load_tuning_results(filepath='baseline/non-ml/harris_tuning_results_per_class.json'):
    """Load the tuning results with best parameters per class"""
    with open(filepath, 'r') as f:
        results = json.load(f)
    return results

def extract_polygon_vertices_harris(png_path, params):
    """Extract polygon vertices using Harris corner detection with given params"""
    # Read image as grayscale
    img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        return None
    
    # Apply Gaussian blur
    blur_size = params['blur_size']
    if blur_size > 0:
        img = cv2.GaussianBlur(img, (blur_size, blur_size), 0)
    
    # Threshold to binary
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Morphological operations
    if params['use_morph']:
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Convert to float32
    binary_float = np.float32(binary)
    
    # Harris corner detection
    harris_response = cv2.cornerHarris(
        binary_float,
        blockSize=params['block_size'],
        ksize=params['ksize'],
        k=params['k']
    )
    
    # Dilate
    if params['use_dilate']:
        harris_response = cv2.dilate(harris_response, None)
    
    # Threshold
    threshold = params['threshold_factor'] * harris_response.max()
    corner_mask = harris_response > threshold
    
    # Get corner coordinates
    corner_coords = np.argwhere(corner_mask)
    
    if len(corner_coords) == 0:
        return None
    
    # Convert from (row, col) to (x, y)
    corners = [[float(coord[1]), float(coord[0])] for coord in corner_coords]
    
    # Cluster nearby corners
    clustered_corners = cluster_corners(corners, params['min_distance'])
    
    if len(clustered_corners) == 0:
        return None
    
    # Sort by angle from centroid
    centroid = np.mean(clustered_corners, axis=0)
    angles = [np.arctan2(p[1] - centroid[1], p[0] - centroid[0]) for p in clustered_corners]
    sorted_indices = np.argsort(angles)
    points = [clustered_corners[i] for i in sorted_indices]
    
    return points

def cluster_corners(corners, min_distance=5.0):
    """Cluster nearby corner points"""
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

def normalize_points(points, width=64, height=64):
    """Normalize points by dividing by image dimensions"""
    return [[x / width, y / height] for x, y in points]

def extract_polygon_type_from_filename(filename):
    """Extract polygon_type from filename (e.g., 'polygon_3_0039_aug_0000.png' -> 'polygon_3')"""
    # Filename format: polygon_X_YYYY_aug_ZZZZ.png
    parts = filename.split('_')
    if len(parts) >= 2 and parts[0] == 'polygon':
        return f"{parts[0]}_{parts[1]}"
    return None

def process_test_set(test_dir, tuning_results, width=64, height=64):
    """Process all test images and collect results"""
    
    results = []
    
    stats = {
        'total': 0,
        'successful': 0,
        'failed': 0,
        'by_class': {}
    }
    
    # Process each polygon class
    for polygon_type in ['polygon_3', 'polygon_4', 'polygon_5', 'polygon_6']:
        print(f"\nProcessing {polygon_type}...")
        
        # Get best params for this class
        best_params = tuning_results[polygon_type]['best_params']
        
        # Input directory for this class
        class_dir = os.path.join(test_dir, polygon_type)
        
        if not os.path.exists(class_dir):
            print(f"  Warning: Directory not found: {class_dir}")
            continue
        
        # Get all PNG files
        png_files = sorted([f for f in os.listdir(class_dir) if f.endswith('.png')])
        
        stats['by_class'][polygon_type] = {
            'total': len(png_files),
            'successful': 0,
            'failed': 0
        }
        
        for png_file in png_files:
            png_path = os.path.join(class_dir, png_file)
            
            # Extract polygon_type from filename
            filename_polygon_type = extract_polygon_type_from_filename(png_file)
            if not filename_polygon_type:
                print(f"  Warning: Could not extract polygon_type from {png_file}")
                filename_polygon_type = polygon_type  # fallback to directory name
            
            # Extract vertices
            try:
                points = extract_polygon_vertices_harris(png_path, best_params)
                
                if points and len(points) > 0:
                    # Create entry
                    num_vertices = len(points)
                    
                    entry = {
                        "filename": png_file.replace('.png', '.svg'),
                        "polygon_type": filename_polygon_type,
                        "num_vertices": num_vertices,
                        "points": points,
                        "normalized_points": normalize_points(points, width, height)
                    }
                    
                    results.append(entry)
                    
                    stats['successful'] += 1
                    stats['by_class'][polygon_type]['successful'] += 1
                else:
                    print(f"  Warning: No points detected for {png_file}")
                    stats['failed'] += 1
                    stats['by_class'][polygon_type]['failed'] += 1
                    
            except Exception as e:
                print(f"  Error processing {png_file}: {e}")
                stats['failed'] += 1
                stats['by_class'][polygon_type]['failed'] += 1
            
            stats['total'] += 1
        
        print(f"  Processed {len(png_files)} files")
        print(f"  Successful: {stats['by_class'][polygon_type]['successful']}")
        print(f"  Failed: {stats['by_class'][polygon_type]['failed']}")
    
    return results, stats

def main():
    parser = argparse.ArgumentParser(description='Apply tuned Harris corner detection to test set')
    parser.add_argument('--output_file', type=str,
                        default='output/baseline_test_set.json',
                        help='Output JSON file path')
    parser.add_argument('--test_dir', type=str, 
                       default='data/64x64/test/raster',
                       help='Test data directory (default: data/64x64/test/raster)')
    parser.add_argument('--tuning_results', type=str,
                       default='baseline/non-ml/harris_tuning_results_per_class.json',
                       help='Path to tuning results JSON file')
    parser.add_argument('--width', type=int, default=64,
                       help='Image width for normalization')
    parser.add_argument('--height', type=int, default=64,
                       help='Image height for normalization')
    
    args = parser.parse_args()
    
    print("="*60)
    print("HARRIS CORNER DETECTION - TEST SET INFERENCE")
    print("="*60)
    
    # Load tuning results
    print(f"\nLoading tuning results from: {args.tuning_results}")
    tuning_results = load_tuning_results(args.tuning_results)
    
    print("\nBest parameters per class:")
    for polygon_type in ['polygon_3', 'polygon_4', 'polygon_5', 'polygon_6']:
        if polygon_type in tuning_results:
            params = tuning_results[polygon_type]['best_params']
            print(f"\n{polygon_type}:")
            for key, value in params.items():
                print(f"  {key}: {value}")
    
    # Process test set
    print(f"\nTest directory: {args.test_dir}")
    print(f"Output file: {args.output_file}")
    
    results, stats = process_test_set(args.test_dir, tuning_results, args.width, args.height)
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Write results to JSON
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nTotal files: {stats['total']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Success rate: {stats['successful']/stats['total']*100:.2f}%")
    
    print(f"\nResults written to: {args.output_file}")
    print(f"Total entries: {len(results)}")
    
    # Show vertex distribution
    vertex_counts = {}
    for entry in results:
        n = entry['num_vertices']
        vertex_counts[n] = vertex_counts.get(n, 0) + 1
    
    print(f"\nDetected vertex distribution:")
    for n in sorted(vertex_counts.keys()):
        print(f"  {n} vertices: {vertex_counts[n]} polygons")

if __name__ == '__main__':
    main()
