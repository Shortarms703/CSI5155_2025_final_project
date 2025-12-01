#!/usr/bin/env python3
"""
Convert SVG files to PNG format, maintaining folder structure.
Converts from vector/ folders to raster/ folders while preserving polygon classes.
"""

import argparse
from pathlib import Path
from PIL import Image
import cairosvg
from io import BytesIO


def convert_svg_to_png(svg_path, output_path, width=None, height=None, 
                       antialias=True, supersample=2, background='white'):
    """
    Convert an SVG file to PNG with optional antialiasing.
    
    Args:
        svg_path: Path to input SVG file
        output_path: Path to output PNG file
        width: Output width in pixels (None = use SVG dimensions)
        height: Output height in pixels (None = use SVG dimensions)
        antialias: Whether to use antialiasing
        supersample: Supersampling factor for antialiasing (e.g., 2 = render at 2x then downsample)
        background: Background color ('white', 'black', 'transparent', or hex color)
    """
    
    # Determine output dimensions
    if antialias and supersample > 1:
        # Render at higher resolution for antialiasing
        ss_width = width * supersample if width else None
        ss_height = height * supersample if height else None
        
        # Convert SVG to PNG at high resolution
        png_data = cairosvg.svg2png(
            url=str(svg_path),
            output_width=ss_width,
            output_height=ss_height,
            background_color=background if background != 'transparent' else None
        )
        
        # Load and downsample
        img = Image.open(BytesIO(png_data))
        
        if width and height:
            img = img.resize((width, height), Image.LANCZOS)
        
    else:
        # Direct rendering without supersampling
        png_data = cairosvg.svg2png(
            url=str(svg_path),
            output_width=width,
            output_height=height,
            background_color=background if background != 'transparent' else None
        )
        
        img = Image.open(BytesIO(png_data))
    
    # Save the result
    img.save(output_path)


def convert_folder_structure(input_base, output_base, width=None, height=None,
                            antialias=True, supersample=2, background='white'):
    """
    Convert all SVG files in the input folder structure to PNG in the output structure.
    
    Args:
        input_base: Base input folder (e.g., "64x64/test/vector")
        output_base: Base output folder (e.g., "64x64/test/raster")
        width: Output width in pixels (None = preserve original)
        height: Output height in pixels (None = preserve original)
        antialias: Whether to use antialiasing
        supersample: Supersampling factor for antialiasing
        background: Background color
    """
    
    input_path = Path(input_base)
    output_path = Path(output_base)
    
    if not input_path.exists():
        print(f"Error: Input path '{input_path}' does not exist!")
        return
    
    # Find all polygon_* directories
    polygon_dirs = sorted([d for d in input_path.iterdir() if d.is_dir() and d.name.startswith('polygon_')])
    
    if not polygon_dirs:
        print(f"Warning: No polygon_* directories found in '{input_path}'")
        return
    
    total_converted = 0
    
    for polygon_dir in polygon_dirs:
        polygon_class = polygon_dir.name  # e.g., "polygon_3"
        
        # Create corresponding output directory
        output_polygon_dir = output_path / polygon_class
        output_polygon_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all SVG files
        svg_files = sorted(polygon_dir.glob('*.svg'))
        
        if not svg_files:
            print(f"Warning: No SVG files found in '{polygon_dir}'")
            continue
        
        print(f"\nConverting {len(svg_files)} files from {polygon_class}...")
        
        for i, svg_file in enumerate(svg_files):
            # Create output filename (change .svg to .png)
            png_filename = output_polygon_dir / f"{svg_file.stem}.png"
            
            try:
                convert_svg_to_png(
                    svg_file, 
                    png_filename,
                    width=width,
                    height=height,
                    antialias=antialias,
                    supersample=supersample,
                    background=background
                )
                total_converted += 1
                
                if (i + 1) % 50 == 0:
                    print(f"  Converted {i + 1}/{len(svg_files)} files...")
                    
            except Exception as e:
                print(f"  Error converting {svg_file.name}: {e}")
        
        print(f"  Completed {polygon_class}: {len(svg_files)} files")
    
    print(f"\nDone! Total files converted: {total_converted}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert SVG files to PNG, maintaining polygon class folder structure',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output paths
    parser.add_argument(
        '-i', '--input',
        type=str,
        default='64x64/test/vector',
        help='Input folder containing polygon_* directories with SVG files'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='64x64/test/raster',
        help='Output folder for PNG files (will mirror polygon_* structure)'
    )
    
    # Conversion parameters
    parser.add_argument(
        '--width',
        type=int,
        default=None,
        help='Output width in pixels (None = preserve original SVG dimensions)'
    )
    
    parser.add_argument(
        '--height',
        type=int,
        default=None,
        help='Output height in pixels (None = preserve original SVG dimensions)'
    )
    
    parser.add_argument(
        '--antialias',
        action='store_true',
        default=True,
        help='Enable antialiasing via supersampling'
    )
    
    parser.add_argument(
        '--no-antialias',
        action='store_false',
        dest='antialias',
        help='Disable antialiasing'
    )
    
    parser.add_argument(
        '--supersample',
        type=int,
        default=8,
        help='Supersampling factor for antialiasing (2 = render at 2x resolution)'
    )
    
    parser.add_argument(
        '--background',
        type=str,
        default='white',
        choices=['white', 'black', 'transparent'],
        help='Background color for PNG output'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"Converting SVG to PNG...")
    print(f"  Input:  {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Dimensions: {args.width or 'auto'}x{args.height or 'auto'}")
    print(f"  Antialiasing: {args.antialias} (supersample={args.supersample})")
    print(f"  Background: {args.background}")
    
    convert_folder_structure(
        input_base=args.input,
        output_base=args.output,
        width=args.width,
        height=args.height,
        antialias=args.antialias,
        supersample=args.supersample,
        background=args.background
    )


if __name__ == "__main__":
    main()
