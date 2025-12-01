import os
import json
import argparse
from pathlib import Path

def points_to_svg_string(points):
    """Convert list of [x, y] points to SVG polygon points string"""
    if not points:
        return ""
    return " ".join([f"{x},{y}" for x, y in points])

def create_svg(points, width=64, height=64):
    """Create SVG string from polygon points"""
    points_str = points_to_svg_string(points)
    
    svg = f'''<?xml version='1.0' encoding='utf-8'?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><rect width="{width}" height="{height}" fill="white" /><polygon points="{points_str}" fill="black" stroke="none" /></svg>'''
    
    return svg

def generate_svgs_from_json(json_path, output_dir):
    """Generate SVG files from JSON data"""
    
    # Load JSON data
    print(f"Loading JSON data from: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Create main output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    stats = {
        'total': 0,
        'successful': 0,
        'failed': 0,
        'by_class': {}
    }
    
    # Group data by polygon type
    by_polygon_type = {}
    for entry in data:
        polygon_type = entry['polygon_type']
        if polygon_type not in by_polygon_type:
            by_polygon_type[polygon_type] = []
        by_polygon_type[polygon_type].append(entry)
    
    # Process each polygon class
    for polygon_type in sorted(by_polygon_type.keys()):
        print(f"\nProcessing {polygon_type}...")
        
        # Create output directory for this class
        class_output_dir = os.path.join(output_dir, polygon_type)
        Path(class_output_dir).mkdir(parents=True, exist_ok=True)
        
        entries = by_polygon_type[polygon_type]
        
        stats['by_class'][polygon_type] = {
            'total': len(entries),
            'successful': 0,
            'failed': 0
        }
        
        for entry in entries:
            try:
                filename = entry['filename']
                points = entry['points']
                
                # Create SVG content
                svg_content = create_svg(points)
                
                # Output path (in class-specific subdirectory)
                svg_path = os.path.join(class_output_dir, filename)
                
                # Write SVG file
                with open(svg_path, 'w', encoding='utf-8') as f:
                    f.write(svg_content)
                
                stats['successful'] += 1
                stats['by_class'][polygon_type]['successful'] += 1
                
            except Exception as e:
                print(f"  Error processing {entry.get('filename', 'unknown')}: {e}")
                stats['failed'] += 1
                stats['by_class'][polygon_type]['failed'] += 1
            
            stats['total'] += 1
        
        print(f"  Processed {len(entries)} files")
        print(f"  Successful: {stats['by_class'][polygon_type]['successful']}")
        print(f"  Failed: {stats['by_class'][polygon_type]['failed']}")
        print(f"  Output: {class_output_dir}")
    
    return stats

def main():
    parser = argparse.ArgumentParser(description='Generate SVG files from JSON polygon data')
    parser.add_argument('--json_path', type=str, required=True,
                        help='Path to input JSON file containing polygon data')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for SVG files')
    parser.add_argument('--width', type=int, default=64,
                        help='SVG width (default: 64)')
    parser.add_argument('--height', type=int, default=64,
                        help='SVG height (default: 64)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("SVG GENERATION FROM JSON")
    print("="*60)
    
    print(f"\nInput JSON: {args.json_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"SVG dimensions: {args.width}x{args.height}")
    
    # Check if JSON file exists
    if not os.path.exists(args.json_path):
        print(f"\nError: JSON file not found: {args.json_path}")
        return
    
    # Generate SVGs
    stats = generate_svgs_from_json(args.json_path, args.output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nTotal files: {stats['total']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    
    if stats['total'] > 0:
        print(f"Success rate: {stats['successful']/stats['total']*100:.2f}%")
    
    print(f"\nOutput directory structure:")
    for polygon_type in sorted(stats['by_class'].keys()):
        count = stats['by_class'][polygon_type]['successful']
        print(f"  {args.output_dir}/{polygon_type}/ - {count} files")

if __name__ == '__main__':
    main()
