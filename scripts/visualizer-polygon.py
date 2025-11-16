# This python file takes argv[1] as path to polygon file to visualization
# Polygon file format is as same as test/polygons

import matplotlib.pyplot as plt
import numpy as np
import sys
import math
import os
from typing import List, Dict, Any
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.patches import Circle

def patchify(polys):
    """Returns a PathPatch that uses even-odd filling for multiple loops."""

    vertices = []
    codes = []

    for poly in polys:
        arr = np.asarray(poly, dtype=float)

        if arr.ndim == 1 or arr.size == 0:
            continue

        # Accept either shape (2, N) or (N, 2)
        if arr.shape[0] == 2 and arr.shape[1] != 2:
            arr = arr.T
        elif arr.shape[1] != 2:
            continue

        if arr.shape[0] < 2:
            continue

        if not np.allclose(arr[0], arr[-1]):
            arr = np.vstack([arr, arr[0]])

        vertices.extend(arr.tolist())
        if arr.shape[0] == 2:
            codes.extend([Path.MOVETO, Path.CLOSEPOLY])
        else:
            codes.extend([Path.MOVETO] + [Path.LINETO] * (arr.shape[0] - 2) + [Path.CLOSEPOLY])

    if not vertices:
        return None

    path = Path(vertices, codes)
    patch = PathPatch(path)
    return patch

def plot_polygon(ax, loops_list, label, color, closed=True, linestyle='-', marker='o', markersize=5, linewidth=1.5, show_indices=False, index_offset=0.2, fill_polygon=False, fill_alpha=0.3):
    """
    Plot a polygon with multiple loops (outer boundary + holes).
    
    Parameters:
    - ax: matplotlib axes object
    - loops_list: list of simple loops (outer boundaries and/or holes), each loop is
                  a list of points [(x1,y1), (x2,y2), ...]
    - label: label for the legend
    - color: color for the polygon
    - closed: whether to close each loop
    - linestyle: line style for plotting
    - marker: marker style for vertices
    - markersize: size of vertex markers
    - linewidth: width of the lines
    - show_indices: whether to show vertex indices
    - index_offset: offset distance for index labels from vertices
    - fill_polygon: whether to fill the polygon using patchify (handles holes correctly)
    - fill_alpha: alpha transparency for the fill
    """
    if not loops_list or len(loops_list) == 0:
        return
    
    # Fill the polygon with holes if requested
    if fill_polygon:
        polys = []
        for loop in loops_list:
            if not loop:
                continue
            polys.append(np.array(loop, dtype=float))

        patch = patchify(polys)
        if patch is not None:
            patch.set_facecolor(color)
            patch.set_alpha(fill_alpha)
            patch.set_edgecolor('none')
            patch.set_label(label)
            ax.add_patch(patch)
            label = None  # Avoid duplicate legend entries when drawing outlines
    
    # Track global index for continuous numbering across all loops
    global_index = 0
    
    # Plot the outlines
    for loop_idx, points_list in enumerate(loops_list):
        if not points_list:
            continue
            
        # Use numpy array for easier vector calculations
        points = np.array(points_list, dtype=float)
        
        # Prepare coordinates for plotting
        plot_coords = np.vstack([points, points[0]]) if closed and len(points) > 1 else points
        
        # Determine line style for holes (you can customize this)
        current_linestyle = linestyle
        # Only label the outer boundary, and only if we didn't already label the fill
        current_label = label if loop_idx == 0 and not fill_polygon else None
        
        # Plot the loop
        ax.plot(plot_coords[:, 0], plot_coords[:, 1], 
                marker=marker, linestyle=current_linestyle, label=current_label, 
                color=color, markersize=markersize, linewidth=linewidth)
        
        # Plot vertex indices if requested
        if show_indices:
            num_points = len(points)
            if num_points == 0:
                continue
            
            # Get a colormap
            colormap = plt.get_cmap('viridis')
            
            for i in range(num_points):
                p_current = points[i]
                
                # Determine previous and next points to calculate the edges
                p_prev = points[i - 1] if closed else (points[i - 1] if i > 0 else None)
                p_next = points[(i + 1) % num_points] if closed else (points[i + 1] if i < num_points - 1 else None)
                
                # Calculate the normal vector at the vertex
                normal_vec = np.array([0.0, 0.0])
                
                if p_prev is not None:
                    # Vector from previous to current point
                    edge_in = p_current - p_prev
                    # Add the normal of the incoming edge
                    normal_vec += np.array([edge_in[1], -edge_in[0]])
                
                if p_next is not None:
                    # Vector from current to next point
                    edge_out = p_next - p_current
                    # Add the normal of the outgoing edge
                    normal_vec += np.array([edge_out[1], -edge_out[0]])
                
                # Normalize the resulting vector to get a direction
                norm_magnitude = np.linalg.norm(normal_vec)
                if norm_magnitude > 1e-9:  # Avoid division by zero
                    unit_normal = normal_vec / norm_magnitude
                else:
                    unit_normal = np.array([0.0, 0.0])
                
                # Calculate the final text position
                text_pos = p_current + unit_normal * index_offset
                
                # Use global index for continuous numbering
                total_points = sum(len(loop) for loop in loops_list if loop)
                text_color = colormap(global_index / (total_points - 1)) if total_points > 1 else colormap(0.0)
                
                # Add a bounding box to the text to prevent visual overlap with lines
                ax.text(text_pos[0], text_pos[1], str(global_index), fontsize=9, color=text_color, 
                        ha='center', va='center',
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.1'))
                
                global_index += 1


def update_bounds(bounds: Dict[str, float], loops_list):
    """Expand bounds with a list of loops (simple polygons)."""
    if not loops_list:
        return

    for loop in loops_list:
        if not loop:
            continue
        arr = np.asarray(loop, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            continue

        bounds['min_x'] = min(bounds['min_x'], np.min(arr[:, 0]))
        bounds['max_x'] = max(bounds['max_x'], np.max(arr[:, 0]))
        bounds['min_y'] = min(bounds['min_y'], np.min(arr[:, 1]))
        bounds['max_y'] = max(bounds['max_y'], np.max(arr[:, 1]))


def update_bounds_with_points(bounds: Dict[str, float], points):
    if not points:
        return

    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return

    bounds['min_x'] = min(bounds['min_x'], np.min(arr[:, 0]))
    bounds['max_x'] = max(bounds['max_x'], np.max(arr[:, 0]))
    bounds['min_y'] = min(bounds['min_y'], np.min(arr[:, 1]))
    bounds['max_y'] = max(bounds['max_y'], np.max(arr[:, 1]))


def compute_axis_limits(bounds: Dict[str, float], padding_ratio: float = 0.05):
    if (not math.isfinite(bounds['min_x']) or not math.isfinite(bounds['max_x']) or
            not math.isfinite(bounds['min_y']) or not math.isfinite(bounds['max_y'])):
        return None

    width = bounds['max_x'] - bounds['min_x']
    height = bounds['max_y'] - bounds['min_y']
    span = max(width, height, 1.0)
    padding = span * padding_ratio

    return (
        bounds['min_x'] - padding,
        bounds['max_x'] + padding,
        bounds['min_y'] - padding,
        bounds['max_y'] + padding,
    )


def read_input_file(filename: str) -> List[Dict[str, Any]]:
    """
    Reads an input file generated by the C++ SavePolygons function.

    Expected format:
    Line 1: Header (filename 1) - SKIPPED
    Line 2: N_polygons
    For each polygon:
      Line 3: N_points
      Line 4...: x y
    
    Returns data in the same format as the old read_polygon_results_file:
    [{"name": "Input", "polygons": [ [poly1_points, poly2_points, ...] ]}]
    (One "test" with one "cross-section" containing N polygons)
    """
    print(f"Reading input file: {filename}")
    try:
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        line_idx = 0
        
        # Line 1: Header (e.g., "input.txt 1"), skip
        header_name = lines[line_idx].split()[0]
        line_idx += 1

        # Line 2: Number of polygons
        n_polygons = int(lines[line_idx])
        line_idx += 1
        print(f"  Found {n_polygons} input polygon(s).")

        current_cross_section_paths = []
        for i in range(n_polygons):
            # Line 3: Number of points in this polygon
            n_points = int(lines[line_idx])
            line_idx += 1

            if n_points == 0:
                print(f"  Skipping input polygon {i} (0 points).")
                continue

            current_polygon = []
            for _ in range(n_points):
                point_coords = lines[line_idx].split()
                x = float(point_coords[0])
                y = float(point_coords[1])
                current_polygon.append([x, y])
                line_idx += 1
            
            current_cross_section_paths.append(current_polygon)
            print(f"  Added input polygon {i} with {n_points} points.")

        # Wrap in the expected return format
        current_test = {
            "name": f"Input ({header_name})",
            "polygons": [current_cross_section_paths] # One cross-section
        }
        return [current_test]

    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return []
    except (ValueError, IndexError) as e:
        print(f"Error parsing input file at line {line_idx + 1}: {e}. The file might be malformed.")
        return []

def read_circles_and_results(filename: str) -> (float, List[List[float]], List[List[float]], List[Dict[str, Any]]):
    """
    Reads a result file containing circles and SaveCrossSection data.

    Expected format:
    1. radius (float)
    2. N_removed_circles
    3. List of (x y) for removed circle centers
    4. N_fillet_circles
    5. List of (x y) for fillet circle centers
    6. N_CrossSections (from SaveCrossSection)
    For each CrossSection:
      7. N_polygons_in_CrossSection
      For each polygon:
        8. N_points_in_polygon
        9. List of (x y) for polygon vertices
        
    Returns:
    - radius: (float) The radius read from the file, or None on error
    - removed_circles: List of [x, y] centers
    - fillet_circles: List of [x, y] centers
    - polygon_data: List[Dict] in the format expected by plotters
    """
    print(f"Reading result file: {filename}")
    radius = None
    removed_circles = []
    fillet_circles = []
    polygon_data = []

    try:
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        line_idx = 0
        if not lines:
            print("Error: File is empty.")
            return None, [], [], []

        # --- 1. Read Radius (NEW) ---
        try:
            radius = float(lines[line_idx])
            line_idx += 1
            print(f"  Read radius: {radius}")
        except (ValueError, IndexError):
            print(f"Error: Could not read radius from first line of {filename}")
            return None, [], [], [] # Return None for radius to signal error

        # --- 2. Read Removed Circles ---
        n_removed_circles = int(lines[line_idx])
        line_idx += 1
        print(f"  Found {n_removed_circles} removed circle(s).")
        for _ in range(n_removed_circles):
            x, y = map(float, lines[line_idx].split())
            removed_circles.append([x, y])
            line_idx += 1

        # --- 3. Read Fillet Circles ---
        n_fillet_circles = int(lines[line_idx])
        line_idx += 1
        print(f"  Found {n_fillet_circles} fillet circle(s).")
        for _ in range(n_fillet_circles):
            x, y = map(float, lines[line_idx].split())
            fillet_circles.append([x, y])
            line_idx += 1

        # --- 4. Read Result Polygons (SaveCrossSection format) ---
        n_cross_sections = int(lines[line_idx])
        line_idx += 1
        print(f"  Found {n_cross_sections} cross-section(s).")

        all_cross_sections = []
        for cs_idx in range(n_cross_sections):
            n_polygons = int(lines[line_idx])
            line_idx += 1
            print(f"    Cross-section {cs_idx} has {n_polygons} polygon(s).")
            
            current_cross_section_paths = []
            for poly_idx in range(n_polygons):
                n_points = int(lines[line_idx])
                line_idx += 1
                
                if n_points == 0:
                    print(f"    Skipping result polygon {poly_idx} in CS {cs_idx} (0 points).")
                    continue
                
                current_polygon = []
                for _ in range(n_points):
                    point_coords = lines[line_idx].split()
                    x = float(point_coords[0])
                    y = float(point_coords[1])
                    current_polygon.append([x, y])
                    line_idx += 1
                
                current_cross_section_paths.append(current_polygon)
                print(f"    Added result polygon {poly_idx} (CS {cs_idx}) with {n_points} points.")
            
            all_cross_sections.append(current_cross_section_paths)

        # Wrap in the expected return format
        current_test = {
            "name": f"Result ({filename})",
            "polygons": all_cross_sections # Multiple cross-sections
        }
        polygon_data = [current_test]

        return radius, removed_circles, fillet_circles, polygon_data

    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return None, [], [], []
    except (ValueError, IndexError) as e:
        print(f"Error parsing result file at line {line_idx + 1}: {e}. The file might be malformed.")
        return radius, removed_circles, fillet_circles, [] # Return partial data if possible

def plot_circles(ax, centers: List[List[float]], radius: float, **kwargs):
    """
    Plots a list of circles on the given axes.
    'kwargs' are passed to matplotlib.patches.Circle (e.g., color, label, linestyle)
    """
    if not centers:
        return
        
    # Make sure label is only applied once
    label = kwargs.pop('label', None)
    
    for i, center in enumerate(centers):
        current_label = label if i == 0 else None
        circle = Circle(center, radius, fill=False, label=current_label, **kwargs)
        ax.add_patch(circle)


def plot_circle_centers(ax, centers: List[List[float]], color: str, label: str, marker: str):
    if not centers:
        return

    centers_arr = np.asarray(centers, dtype=float)
    if centers_arr.ndim != 2 or centers_arr.shape[1] != 2:
        return

    ax.scatter(centers_arr[:, 0], centers_arr[:, 1], color=color, label=label,
               marker=marker, s=30, edgecolor='black', linewidth=0.5, zorder=4)


if __name__ == "__main__":
    input_file = "Testing/Fillet/input.txt" # Hardcoded as per request
    max_file_index = 10      # Hardcoded scan from 0 to 10

    # --- Read Input Data ONCE ---
    print(f"Reading base input file: {input_file}")
    input_data = read_input_file(input_file)
    if not input_data:
        print("Error: Could not read input file. Exiting.")
        sys.exit(1)
    
    # --- Collect all valid result files first ---
    results_to_plot = []
    print("Scanning for result files (0.txt to 10.txt)...")
    for i in range(max_file_index + 1):
        result_file = f"Testing/Fillet/{i}.txt"
        radius, removed, fillet, res_data = read_circles_and_results(result_file)
        
        # If radius is None, file wasn't found or was invalid
        if radius is not None:
            results_to_plot.append((result_file, radius, removed, fillet, res_data))
        else:
            print(f"Skipping {result_file} (could not be read or radius not found).")
            
    if not results_to_plot:
        print("No valid result files found to plot. Exiting.")
        sys.exit(1)

    print(f"\nFound {len(results_to_plot)} valid result files. Creating combined plot...")

    bounds = {
        'min_x': math.inf,
        'max_x': -math.inf,
        'min_y': math.inf,
        'max_y': -math.inf,
    }

    if input_data and input_data[0]["polygons"]:
        for cross_section in input_data[0]["polygons"]:
            update_bounds(bounds, cross_section)

    for _, _, removed_circles, fillet_circles, result_data in results_to_plot:
        if result_data and result_data[0].get("polygons"):
            for cross_section in result_data[0]["polygons"]:
                update_bounds(bounds, cross_section)

        update_bounds_with_points(bounds, removed_circles)
        update_bounds_with_points(bounds, fillet_circles)

    axis_limits = compute_axis_limits(bounds)

    # --- Calculate Grid Size ---
    num_plots = len(results_to_plot)
    cols = 4  # You can adjust this for a different layout (e.g., 3 or 5)
    # Calculate rows needed to fit all plots
    rows = (num_plots + cols - 1) // cols 

    # --- Plot Creation (one big figure) ---
    # Create a figure with rows x cols subplots.
    # squeeze=False ensures 'axes' is always a 2D array, even if rows=1
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 7, rows * 7), squeeze=False)
    
    # Flatten the 2D array of axes into a 1D list for easy iteration
    flat_axes = axes.flatten()
    result_cmap = plt.get_cmap('tab10')

    # --- Process Each Result File into a subplot ---
    for plot_index, (result_file, radius, removed_circles, fillet_circles, result_data) in enumerate(results_to_plot):
        
        ax = flat_axes[plot_index] # Get the current subplot
        
        print(f"Plotting {result_file} on subplot {plot_index}...")

        ax.grid(True, linestyle=':', alpha=0.7)
        ax.set_aspect('equal', adjustable='box')

        if axis_limits:
            ax.set_xlim(axis_limits[0], axis_limits[1])
            ax.set_ylim(axis_limits[2], axis_limits[3])

        # --- Plot Input (on every subplot) ---
        if input_data and input_data[0]["polygons"]:
            input_polys = input_data[0]["polygons"][0]
            plot_polygon(ax, input_polys, label="Input Polygon", color='blue', 
                         linestyle='--', marker='.', markersize=4, linewidth=1, 
                         show_indices=False, fill_polygon=True, fill_alpha=0.1)

        # --- Plot Result ---
        if result_data and result_data[0]["polygons"]:
            all_result_cross_sections = result_data[0]["polygons"]
            for i, cross_section in enumerate(all_result_cross_sections):
                color = result_cmap(i % result_cmap.N)
                plot_polygon(ax, cross_section, label=f"Result Polygon (CS {i})", color=color, 
                             linestyle='-', marker='o', markersize=5, linewidth=2, 
                              show_indices=True, fill_polygon=False)

        # --- Plot Circles ---
        plot_circles(ax, removed_circles, radius, color='red', linestyle='--', alpha=0.35, linewidth=1)
        plot_circles(ax, fillet_circles, radius, color='green', linestyle=':', alpha=0.35, linewidth=1)
        plot_circle_centers(ax, removed_circles, color='red', label='Removed center', marker='x')
        plot_circle_centers(ax, fillet_circles, color='green', label='Fillet center', marker='o')

        # --- Subplot Finalization ---
        ax.set_title(f"{os.path.basename(result_file)} (r={radius:.3f})")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")

        # De-duplicate legend entries for this subplot
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize='x-small')
            
    # --- Turn off any unused subplots ---
    for i in range(num_plots, rows * cols):
        flat_axes[i].axis('off')
        
    # --- Final Figure Saving ---
    fig.suptitle(f"Polygon Visualization (Input: {input_file})", fontsize=20)
    # Adjust layout to prevent title overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    
    save_name = "all_results.png"
    try:
        plt.savefig(save_name)
        print(f"\nSuccessfully saved combined plot to {save_name}")
    except Exception as e:
        print(f"Error saving plot {save_name}: {e}")
    
    # Close the figure to free up memory
    plt.close(fig)

    print("\nAll result files processed.")
