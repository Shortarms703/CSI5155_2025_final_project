import os
import xml.etree.ElementTree as ET
import json
import argparse

def extract_polygon_points(svg_path):
    tree = ET.parse(svg_path)
    root = tree.getroot()
    ns = {'svg': 'http://www.w3.org/2000/svg'}
    
    polygon = root.find('.//svg:polygon', ns)
    if polygon is not None:
        points_str = polygon.get('points')
        # Parse into list of [x, y] pairs
        coords = [float(x) for x in points_str.replace(',', ' ').split()]
        points = [[coords[i], coords[i+1]] for i in range(0, len(coords), 2)]
        return points
    
    return None

def normalize_points(points, image_size):
    """Normalize points relative to image size."""
    if points is None:
        return None
    return [[x / image_size, y / image_size] for x, y in points]

def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract polygon ground truth from SVG files'
    )
    
    parser.add_argument(
        '--base-dir',
        type=str,
        default='data/64x64',
        help='Base directory containing train/test data (default: data/64x64)'
    )
    
    parser.add_argument(
        '--output-file',
        type=str,
        help='Path to output JSON file (default: evaluation/ground_truth_{dataset}_set.json)'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['train', 'test'],
        default='train',
        help='Dataset to process: train or test (default: train)'
    )
    
    parser.add_argument(
        '--normalize',
        action='store_true',
        help='Include normalized points (0-1 range) alongside absolute points'
    )
    
    parser.add_argument(
        '--image-size',
        type=int,
        default=64,
        help='Image size for normalization (default: 64)'
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Construct the full vector directory path
    vector_dir = os.path.join(args.base_dir, args.dataset, 'vector')
    
    # Set default output file if not specified
    if args.output_file is None:
        output_file = f'evaluation/ground_truth_{args.dataset}_set.json'
    else:
        output_file = args.output_file
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    data = []
    
    for polygon_type in ['polygon_3', 'polygon_4', 'polygon_5', 'polygon_6']:
        svg_dir = os.path.join(vector_dir, polygon_type)
        
        if not os.path.exists(svg_dir):
            print(f"Warning: Directory not found: {svg_dir}")
            continue
        
        for svg_file in sorted(os.listdir(svg_dir)):
            if svg_file.endswith('.svg'):
                svg_path = os.path.join(svg_dir, svg_file)
                points = extract_polygon_points(svg_path)
                
                entry = {
                    'filename': svg_file,
                    'polygon_type': polygon_type,
                    'num_vertices': int(polygon_type.split('_')[1]),
                    'points': points
                }
                
                # Add normalized points if requested
                if args.normalize:
                    normalized_points = normalize_points(points, args.image_size)
                    entry['normalized_points'] = normalized_points
                
                data.append(entry)
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Ground truth extracted to {output_file}")
    print(f"Total polygons processed: {len(data)}")
    print(f"Normalized points included: {args.normalize}")
    if args.normalize:
        print(f"Image size used for normalization: {args.image_size}")

if __name__ == '__main__':
    main()
