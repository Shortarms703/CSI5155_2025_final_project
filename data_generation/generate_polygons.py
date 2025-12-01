import argparse
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import svgwrite
from scipy.spatial import ConvexHull
import math


def parse_args():
    parser = argparse.ArgumentParser(description='Generate polygon images in SVG and PNG formats')
    parser.add_argument('--width', type=int, default=64, help='Image width')
    parser.add_argument('--height', type=int, default=64, help='Image height')
    parser.add_argument('--sides', type=str, required=True, 
                        help='Number of sides (e.g., "5" or "3-6" for range)')
    parser.add_argument('--count', type=int, required=True, help='Number of images per polygon type')
    parser.add_argument('--threshold', type=float, default=3.0, 
                        help='Min area of the polygon as a percentage of the image (0-100)')
    parser.add_argument('--maximize', action='store_true', 
                        help='The polygon size will fit the image without cropping')
    parser.add_argument('--min_dist', type=float, default=5.0, 
                        help='Min distance between vertices (% of min dim)')
    parser.add_argument('--convex_only', action='store_true', 
                        help='Generate only convex polygons')
    parser.add_argument('--antialias', action='store_true', default=True, 
                        help='Enable antialiasing for PNGs (default: True)')
    parser.add_argument('--supersample', type=int, default=8, choices=[1, 2, 4, 8], 
                        help='Supersampling factor for antialiasing (default: 8)')
    parser.add_argument('--allow_colinear_vertices', action='store_true', default=False,
                        help='Allow colinear vertices')
    parser.add_argument('--output', type=str, default='.', 
                        help='Output folder path (default: current directory)')
    parser.add_argument('--min_angle', type=float, default=None,
                        help='Minimum interior angle in degrees')
    parser.add_argument('--max_angle', type=float, default=None,
                        help='Maximum interior angle in degrees')
    
    return parser.parse_args()


def parse_sides_range(sides_str):
    """Parse sides argument, returns list of side counts"""
    if '-' in sides_str:
        start, end = map(int, sides_str.split('-'))
        return list(range(start, end + 1))
    else:
        return [int(sides_str)]


def is_simple_polygon(vertices):
    """Check if polygon is simple (non-self-intersecting)"""
    n = len(vertices)
    
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    
    def segments_intersect(A, B, C, D):
        # Check if line segment AB intersects CD
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
    
    # Check all non-adjacent edge pairs
    for i in range(n):
        for j in range(i + 2, n):
            # Skip adjacent edges
            if j == (i + 1) % n or i == (j + 1) % n:
                continue
            
            p1, p2 = vertices[i], vertices[(i + 1) % n]
            p3, p4 = vertices[j], vertices[(j + 1) % n]
            
            if segments_intersect(p1, p2, p3, p4):
                return False
    
    return True


def points_are_colinear(p1, p2, p3, tolerance=1e-6):
    """Check if three points are colinear"""
    # Using cross product
    cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
    return abs(cross) < tolerance


def has_colinear_vertices(vertices):
    """Check if any three consecutive vertices are colinear"""
    n = len(vertices)
    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n]
        p3 = vertices[(i + 2) % n]
        if points_are_colinear(p1, p2, p3):
            return True
    return False


def calculate_polygon_area(vertices):
    """Calculate polygon area using shoelace formula"""
    n = len(vertices)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]
    return abs(area) / 2.0


def is_convex(vertices):
    """Check if polygon is convex"""
    n = len(vertices)
    if n < 3:
        return False
    
    sign = None
    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n]
        p3 = vertices[(i + 2) % n]
        
        # Cross product
        cross = (p2[0] - p1[0]) * (p3[1] - p2[1]) - (p2[1] - p1[1]) * (p3[0] - p2[0])
        
        if abs(cross) < 1e-6:  # Nearly colinear
            continue
            
        if sign is None:
            sign = cross > 0
        elif (cross > 0) != sign:
            return False
    
    return True


def calculate_interior_angles(vertices):
    """Calculate all interior angles of the polygon in degrees"""
    n = len(vertices)
    angles = []
    
    for i in range(n):
        p1 = np.array(vertices[(i - 1) % n])
        p2 = np.array(vertices[i])
        p3 = np.array(vertices[(i + 1) % n])
        
        v1 = p1 - p2
        v2 = p3 - p2
        
        # Calculate angle using atan2
        angle = np.arctan2(np.cross(v1, v2), np.dot(v1, v2))
        angle_deg = np.degrees(angle)
        
        # Ensure positive angle
        if angle_deg < 0:
            angle_deg += 360
            
        angles.append(angle_deg)
    
    return angles


def check_angle_constraints(vertices, min_angle, max_angle):
    """Check if polygon satisfies angle constraints"""
    if min_angle is None and max_angle is None:
        return True
    
    angles = calculate_interior_angles(vertices)
    
    if min_angle is not None:
        if any(angle < min_angle for angle in angles):
            return False
    
    if max_angle is not None:
        if any(angle > max_angle for angle in angles):
            return False
    
    return True


def generate_polygon(num_sides, width, height, min_dist_percent, convex_only, 
                     allow_colinear, min_angle, max_angle, threshold_percent, maximize):
    """Generate a random non-self-intersecting polygon"""
    
    max_attempts = 10000
    min_dim = min(width, height)
    min_dist = (min_dist_percent / 100.0) * min_dim
    min_area = (threshold_percent / 100.0) * width * height
    
    # Calculate margin for centering
    margin = 0.1 if not maximize else 0.05
    
    for attempt in range(max_attempts):
        if convex_only:
            # Generate convex polygon using random points and convex hull
            if maximize:
                # Generate on a circle for maximum size
                center_x, center_y = width / 2, height / 2
                max_radius = min(center_x, center_y) * 0.9
                
                angles = sorted(np.random.uniform(0, 2 * np.pi, num_sides))
                vertices = []
                for angle in angles:
                    x = center_x + max_radius * np.cos(angle)
                    y = center_y + max_radius * np.sin(angle)
                    vertices.append((x, y))
            else:
                # Generate random points within bounds
                num_candidates = max(num_sides * 3, 20)
                x_min, x_max = width * margin, width * (1 - margin)
                y_min, y_max = height * margin, height * (1 - margin)
                
                points = np.column_stack([
                    np.random.uniform(x_min, x_max, num_candidates),
                    np.random.uniform(y_min, y_max, num_candidates)
                ])
                
                # Compute convex hull
                hull = ConvexHull(points)
                hull_indices = hull.vertices
                
                # If we have more points than needed, randomly sample
                if len(hull_indices) >= num_sides:
                    # Sample num_sides points from hull
                    selected = np.random.choice(hull_indices, num_sides, replace=False)
                    # Sort by angle from centroid
                    centroid = points[selected].mean(axis=0)
                    angles = np.arctan2(points[selected][:, 1] - centroid[1], 
                                       points[selected][:, 0] - centroid[0])
                    sorted_indices = selected[np.argsort(angles)]
                    vertices = [tuple(points[i]) for i in sorted_indices]
                else:
                    continue
        else:
            # Generate non-convex polygon
            if maximize:
                center_x, center_y = width / 2, height / 2
                max_radius = min(center_x, center_y) * 0.9
                min_radius = max_radius * 0.3
                
                angles = sorted(np.random.uniform(0, 2 * np.pi, num_sides))
                vertices = []
                for angle in angles:
                    radius = np.random.uniform(min_radius, max_radius)
                    x = center_x + radius * np.cos(angle)
                    y = center_y + radius * np.sin(angle)
                    vertices.append((x, y))
            else:
                # Generate random vertices
                x_min, x_max = width * margin, width * (1 - margin)
                y_min, y_max = height * margin, height * (1 - margin)
                
                vertices = []
                for _ in range(num_sides):
                    x = np.random.uniform(x_min, x_max)
                    y = np.random.uniform(y_min, y_max)
                    vertices.append((x, y))
                
                # Sort by angle to reduce self-intersection likelihood
                centroid_x = sum(v[0] for v in vertices) / len(vertices)
                centroid_y = sum(v[1] for v in vertices) / len(vertices)
                vertices.sort(key=lambda v: np.arctan2(v[1] - centroid_y, v[0] - centroid_x))
        
        # Check minimum distance between vertices
        valid_distances = True
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                dist = np.sqrt((vertices[i][0] - vertices[j][0])**2 + 
                              (vertices[i][1] - vertices[j][1])**2)
                if dist < min_dist:
                    valid_distances = False
                    break
            if not valid_distances:
                break
        
        if not valid_distances:
            continue
        
        # Check for colinear vertices
        if not allow_colinear and has_colinear_vertices(vertices):
            continue
        
        # Check if simple (non-self-intersecting)
        if not is_simple_polygon(vertices):
            continue
        
        # Check convexity if required
        if convex_only and not is_convex(vertices):
            continue
        
        # Check area threshold
        area = calculate_polygon_area(vertices)
        if area < min_area:
            continue
        
        # Check angle constraints
        if not check_angle_constraints(vertices, min_angle, max_angle):
            continue
        
        return vertices
    
    return None


def draw_png(vertices, width, height, antialias, supersample):
    """Draw polygon as PNG with optional antialiasing - white background, black polygon"""
    
    if antialias and supersample > 1:
        # Render at higher resolution
        ss_width = width * supersample
        ss_height = height * supersample
        
        # Scale vertices
        ss_vertices = [(x * supersample, y * supersample) for x, y in vertices]
        
        # Create high-res image with white background
        img = Image.new('L', (ss_width, ss_height), color=255)
        draw = ImageDraw.Draw(img)
        draw.polygon(ss_vertices, fill=0, outline=0)
        
        # Downsample
        img = img.resize((width, height), Image.LANCZOS)
    else:
        # Direct rendering - white background, black polygon
        img = Image.new('L', (width, height), color=255)
        draw = ImageDraw.Draw(img)
        draw.polygon(vertices, fill=0, outline=0)
    
    return img


def draw_svg(vertices, width, height, filename):
    """Draw polygon as SVG - white background, black polygon"""
    dwg = svgwrite.Drawing(filename, size=(width, height), profile='tiny')
    
    # Create polygon points
    points = [(x, y) for x, y in vertices]
    
    # Add white background and black polygon
    dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill='white'))
    dwg.add(dwg.polygon(points=points, fill='black', stroke='black'))
    
    dwg.save()


def main():
    args = parse_args()
    
    # Parse sides range
    sides_list = parse_sides_range(args.sides)
    
    # Create output directory structure
    output_base = Path(args.output)
    dim_folder = f"{args.width}x{args.height}"
    
    for num_sides in sides_list:
        print(f"\nGenerating {args.count} polygons with {num_sides} sides...")
        
        # Create directories
        raster_dir = output_base / dim_folder / "raster" / f"polygon_{num_sides}"
        vector_dir = output_base / dim_folder / "vector" / f"polygon_{num_sides}"
        raster_dir.mkdir(parents=True, exist_ok=True)
        vector_dir.mkdir(parents=True, exist_ok=True)
        
        successful = 0
        attempts = 0
        max_total_attempts = args.count * 100
        
        while successful < args.count and attempts < max_total_attempts:
            attempts += 1
            
            vertices = generate_polygon(
                num_sides, 
                args.width, 
                args.height, 
                args.min_dist,
                args.convex_only,
                args.allow_colinear_vertices,
                args.min_angle,
                args.max_angle,
                args.threshold,
                args.maximize
            )
            
            if vertices is None:
                continue
            
            # Generate filenames
            png_filename = raster_dir / f"polygon_{num_sides}_{successful:04d}.png"
            svg_filename = vector_dir / f"polygon_{num_sides}_{successful:04d}.svg"
            
            # Draw and save PNG
            img = draw_png(vertices, args.width, args.height, args.antialias, args.supersample)
            img.save(png_filename)
            
            # Draw and save SVG
            draw_svg(vertices, args.width, args.height, svg_filename)
            
            successful += 1
            
            if successful % 100 == 0:
                print(f"  Generated {successful}/{args.count} polygons...")
        
        if successful < args.count:
            print(f"  Warning: Only generated {successful}/{args.count} valid polygons after {attempts} attempts")
        else:
            print(f"  Successfully generated {successful} polygons")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
