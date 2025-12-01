#!/usr/bin/env python3
"""
Create augmented train/test datasets from clustering results.

Usage:
    python augment_dataset.py --augmentations 5
    python augment_dataset.py --augmentations 10 --width 128 --height 128
    python augment_dataset.py --augmentations 5 --min-size 0.3 --seed 42
"""

import json
import random
import argparse
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import cairosvg


def parse_polygon_points(svg_path):
    """Extract polygon points from SVG file."""
    tree = ET.parse(svg_path)
    root = tree.getroot()
    
    polygon = root.find('.//{http://www.w3.org/2000/svg}polygon')
    if polygon is None:
        polygon = root.find('.//polygon')
    
    points_str = polygon.get('points')
    points = []
    coords = points_str.strip().split()
    for coord in coords:
        x, y = map(float, coord.split(','))
        points.append((x, y))
    
    return points


def get_bounding_box(points):
    """Calculate bounding box of polygon points."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def get_bbox_min_size(bbox):
    """Get minimum dimension of bounding box."""
    min_x, min_y, max_x, max_y = bbox
    width = max_x - min_x
    height = max_y - min_y
    return min(width, height)


def transform_polygon(points, width, height, min_size_ratio, seed):
    """Apply random transformations keeping polygon inside canvas."""
    random.seed(seed)
    transformed = points.copy()
    
    bbox = get_bounding_box(transformed)
    min_x, min_y, max_x, max_y = bbox
    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2
    normalized = [(x - cx, y - cy) for x, y in transformed]
    
    if random.random() > 0.5:
        normalized = [(-x, y) for x, y in normalized]
    if random.random() > 0.5:
        normalized = [(x, -y) for x, y in normalized]
    
    angle = random.uniform(0, 360)
    rad = math.radians(angle)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    
    rotated = []
    for x, y in normalized:
        rx = x * cos_a - y * sin_a
        ry = x * sin_a + y * cos_a
        rotated.append((rx, ry))
    
    normalized = rotated
    
    bbox = get_bounding_box(normalized)
    min_x, min_y, max_x, max_y = bbox
    current_width = max_x - min_x
    current_height = max_y - min_y
    current_size = min(current_width, current_height)
    
    margin = 2
    
    max_scale_x = (width - 2 * margin) / current_width if current_width > 0 else 1.0
    max_scale_y = (height - 2 * margin) / current_height if current_height > 0 else 1.0
    max_scale = min(max_scale_x, max_scale_y)
    
    canvas_min = min(width, height)
    min_allowed_size = canvas_min * min_size_ratio
    min_scale = min_allowed_size / current_size if current_size > 0 else 1.0
    
    # If min_scale > max_scale, the min_size requirement is impossible to meet while fitting.
    # In that case, ignore min_size and use a smaller scale that fits comfortably.
    if min_scale > max_scale:
        scale = max_scale * 0.95
    else:
        safe_max_scale = max_scale * 0.95
        scale = random.uniform(min_scale, safe_max_scale)
    
    scaled = [(x * scale, y * scale) for x, y in normalized]
    
    bbox = get_bounding_box(scaled)
    min_x, min_y, max_x, max_y = bbox
    
    tx_min = margin - min_x
    tx_max = width - margin - max_x
    ty_min = margin - min_y
    ty_max = height - margin - max_y
    
    # Clamp translation if ranges are invalid
    if tx_min > tx_max:
        tx = (tx_min + tx_max) / 2
    else:
        tx = random.uniform(tx_min, tx_max)
    
    if ty_min > ty_max:
        ty = (ty_min + ty_max) / 2
    else:
        ty = random.uniform(ty_min, ty_max)
    
    translated = [(x + tx, y + ty) for x, y in scaled]
    
    return translated


def create_svg(points, width, height, output_path):
    """Create SVG file with only polygon element."""
    svg = ET.Element('svg', {
        'xmlns': 'http://www.w3.org/2000/svg',
        'width': str(width),
        'height': str(height),
        'viewBox': f'0 0 {width} {height}'
    })

    ET.SubElement(svg, 'rect', {
    'width': str(width),
    'height': str(height),
    'fill': 'white'
    }) 
    
    points_str = ' '.join([f'{x},{y}' for x, y in points])
    ET.SubElement(svg, 'polygon', {
        'points': points_str,
        'fill': 'black',
        'stroke': 'none'
    })
    
    tree = ET.ElementTree(svg)
    tree.write(output_path, encoding='unicode', xml_declaration=True)


def render_svg_to_png(svg_path, png_path, width, height):
    """Render SVG to PNG."""
    cairosvg.svg2png(
        url=str(svg_path),
        write_to=str(png_path),
        output_width=width,
        output_height=height
    )


def create_folder_structure(base_path, polygon_types):
    """Create train/test folder structure."""
    for split in ['train', 'test']:
        for format_type in ['raster', 'vector']:
            for polygon_type in polygon_types:
                folder = base_path / split / format_type / polygon_type
                folder.mkdir(parents=True, exist_ok=True)


def process_dataset(data_generation_dir, split_data, source_folder, output_folder, 
                   num_augmentations, width, height, min_size_ratio, seed):
    """Process all images in train and test sets with augmentation."""
    source_path = data_generation_dir / source_folder
    output_path = data_generation_dir / output_folder
    original_vector = source_path / 'vector'
    
    polygon_types = list(split_data['polygon_types'].keys())
    create_folder_structure(output_path, polygon_types)
    
    for split in ['train', 'test']:
        print(f"\nProcessing {split} set...")
        
        for polygon_type in polygon_types:
            images = split_data['polygon_types'][polygon_type][split]
            print(f"  {polygon_type}: {len(images)} images × {num_augmentations} augmentations")
            
            vector_output = output_path / split / 'vector' / polygon_type
            raster_output = output_path / split / 'raster' / polygon_type
            
            for img_name in images:
                vector_input = original_vector / polygon_type / f"{img_name}.svg"
                
                if not vector_input.exists():
                    print(f"    Warning: {vector_input} not found, skipping")
                    continue
                
                points = parse_polygon_points(vector_input)
                
                for i in range(num_augmentations):
                    transformed_points = transform_polygon(
                        points, width, height, min_size_ratio, 
                        seed + hash(img_name) + i
                    )
                    
                    svg_path = vector_output / f"{img_name}_aug_{i:04d}.svg"
                    create_svg(transformed_points, width, height, svg_path)
                    
                    png_path = raster_output / f"{img_name}_aug_{i:04d}.png"
                    render_svg_to_png(svg_path, png_path, width, height)


def main():
    parser = argparse.ArgumentParser(description='Create augmented train/test datasets')
    
    parser.add_argument(
        '--augmentations',
        type=int,
        required=True,
        help='Number of augmented versions to create per image'
    )
    
    parser.add_argument(
        '--width',
        type=int,
        default=64,
        help='Output image width in pixels (default: 64)'
    )
    
    parser.add_argument(
        '--height',
        type=int,
        default=64,
        help='Output image height in pixels (default: 64)'
    )
    
    parser.add_argument(
        '--min-size',
        type=float,
        default=0.3,
        help='Minimum bounding box size as ratio of canvas (0.1-1.0, default: 0.3). Ignored if conflicts with fit requirement.'
    )
    
    parser.add_argument(
        '--source',
        type=str,
        default='64x64',
        help='Source folder with original images (default: 64x64)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output folder name (default: {width}x{height})'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='train_test_split.json',
        help='Input split JSON file (default: train_test_split.json)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    if not 0.1 <= args.min_size <= 1.0:
        parser.error('--min-size must be between 0.1 and 1.0')
    
    output_folder = args.output if args.output else f"{args.width}x{args.height}"
    
    data_generation_dir = Path(__file__).parent
    clusterization_dir = data_generation_dir / 'clusterization'
    split_file = clusterization_dir / args.input
    
    if not split_file.exists():
        parser.error(f'Split file not found: {split_file}')
    
    with open(split_file, 'r') as f:
        split_data = json.load(f)
    
    print(f"Source folder: {args.source}")
    print(f"Output folder: {output_folder}")
    print(f"Output dimensions: {args.width}x{args.height}")
    print(f"Augmentations per image: {args.augmentations}")
    print(f"Minimum size ratio: {args.min_size} (ignored if shape doesn't fit)")
    print(f"Random seed: {args.seed}")
    
    process_dataset(
        data_generation_dir, 
        split_data, 
        args.source, 
        output_folder,
        args.augmentations, 
        args.width,
        args.height,
        args.min_size, 
        args.seed
    )
    
    print(f"\n✓ Dataset creation complete")
    print(f"Output: {data_generation_dir / output_folder / 'train'}")
    print(f"        {data_generation_dir / output_folder / 'test'}")


if __name__ == '__main__':
    main()
