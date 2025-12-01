#!/usr/bin/env python3
"""
Create train/test split from clustering results.

Usage:
    python create_train_test_split.py --test-split 0.1
    python create_train_test_split.py --test-split 20
"""

import json
import random
import argparse
from pathlib import Path


def extract_representatives(clustering_data):
    """Extract representative image filenames without extension from clusters."""
    representatives = []
    for cluster in clustering_data['clusters']:
        filename = Path(cluster['representative_image']).stem
        representatives.append(filename)
    return representatives


def create_split(representatives, test_split, seed):
    """Split representatives into train and test sets."""
    random.seed(seed)
    
    # Calculate test size (percentage or absolute)
    if isinstance(test_split, float):
        test_size = int(len(representatives) * test_split)
    else:
        test_size = test_split
    
    shuffled = representatives.copy()
    random.shuffle(shuffled)
    
    return {
        'train': sorted(shuffled[test_size:]),
        'test': sorted(shuffled[:test_size])
    }


def process_clustering_files(clusterization_dir, test_split, seed):
    """Process all clustering JSON files and create train/test splits."""
    result = {
        'metadata': {
            'test_split_parameter': test_split,
            'random_seed': seed,
        },
        'polygon_types': {}
    }
    
    clustering_files = sorted(clusterization_dir.glob('polygon_*_clustering.json'))
    
    for filepath in clustering_files:
        with open(filepath, 'r') as f:
            clustering_data = json.load(f)
        
        polygon_type = clustering_data['polygon_type']
        representatives = extract_representatives(clustering_data)
        split = create_split(representatives, test_split, seed)
        
        result['polygon_types'][polygon_type] = {
            'total_clusters': len(representatives),
            'train_size': len(split['train']),
            'test_size': len(split['test']),
            'train': split['train'],
            'test': split['test']
        }
        
        print(f"{polygon_type}: {len(split['train'])} train, {len(split['test'])} test")
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Create train/test split from clustering results')
    
    parser.add_argument(
        '--test-split',
        type=str,
        required=True,
        help='Test set size: float (0.0-1.0) for percentage or integer for absolute number'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='train_test_split.json',
        help='Output JSON filename (default: train_test_split.json)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Parse test_split as float or int
    if '.' in args.test_split:
        test_split = float(args.test_split)
    else:
        test_split = int(args.test_split)
    
    clusterization_dir = Path(__file__).parent
    output_path = clusterization_dir / args.output
    
    result = process_clustering_files(clusterization_dir, test_split, args.seed)
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nCreated {output_path}")


if __name__ == '__main__':
    main()
