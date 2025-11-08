# This python file takes argv[1] as path to polygon file to visualization
# Polygon file format is as same as test/polygons

import matplotlib.pyplot as plt
import numpy as np
import sys
from typing import List, Dict, Any
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.patches import Circle

def patchify(polys):
    """Returns a matplotlib patch representing the polygon with holes.
    
    polys is an iterable (i.e list) of polygons, each polygon is a numpy array 
    of shape (2, N), where N is the number of points in each polygon.
    The first polygon is assumed to be the exterior polygon and the rest are holes.
    The first and last points of each polygon may or may not be the same.
    
    This is inspired by https://sgillies.net/2010/04/06/painting-punctured-polygons-with-matplotlib.html
    """
    
    def reorder(poly, cw=True):
        """Reorders the polygon to run clockwise or counter-clockwise according to the value of cw.
        It calculates whether a polygon is cw or ccw by summing (x2-x1)*(y2+y1) for all edges 
        of the polygon, see https://stackoverflow.com/a/1165943/898213.
        """
        # Close polygon if not closed
        if not np.allclose(poly[:, 0], poly[:, -1]):
            poly = np.c_[poly, poly[:, 0]]
        
        direction = ((poly[0] - np.roll(poly[0], 1)) * (poly[1] + np.roll(poly[1], 1))).sum() < 0
        if direction == cw:
            return poly
        else:
            return poly[:, ::-1]
    
    def ring_coding(n):
        """Returns a list of len(n) of this format: 
        [MOVETO, LINETO, LINETO, ..., LINETO, LINETO, CLOSEPOLY]
        """
        codes = [Path.LINETO] * n
        codes[0] = Path.MOVETO
        codes[-1] = Path.CLOSEPOLY
        return codes
    
    # First polygon (exterior) should be CCW, holes should be CW
    ccw = [True] + ([False] * (len(polys) - 1))
    polys = [reorder(poly, c) for poly, c in zip(polys, ccw)]
    codes = np.concatenate([ring_coding(p.shape[1]) for p in polys])
    vertices = np.concatenate(polys, axis=1)
    return PathPatch(Path(vertices.T, codes))

def plot_polygon(ax, loops_list, label, color, closed=True, linestyle='-', marker='o', markersize=5, linewidth=1.5, show_indices=False, index_offset=0.2, fill_polygon=False, fill_alpha=0.3):
    """
    Plot a polygon with multiple loops (outer boundary + holes).
    
    Parameters:
    - ax: matplotlib axes object
    - loops_list: list of loops, where each loop is a list of points [(x1,y1), (x2,y2), ...]
                  First loop is the outer boundary, subsequent loops are holes
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
        # Convert loops to the format expected by patchify (numpy arrays of shape (2, N))
        polys = []
        for loop in loops_list:
            if not loop:
                continue
            loop_array = np.array(loop).T  # Convert [(x1,y1), (x2,y2), ...] to [[x1,x2,...], [y1,y2,...]]
            polys.append(loop_array)
        
        if polys:
            # Create and add the filled patch
            patch = patchify(polys)
            patch.set_facecolor(color)
            patch.set_alpha(fill_alpha)
            # Use the main label for the filled patch if it's the primary representation
            patch.set_label(label) 
            ax.add_patch(patch)
    
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

    # --- Process Each Result File into a subplot ---
    for plot_index, (result_file, radius, removed_circles, fillet_circles, result_data) in enumerate(results_to_plot):
        
        ax = flat_axes[plot_index] # Get the current subplot
        
        print(f"Plotting {result_file} on subplot {plot_index}...")

        ax.grid(True, linestyle=':', alpha=0.7)
        ax.set_aspect('equal', adjustable='box')

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
                plot_polygon(ax, cross_section, label=f"Result Polygon (CS {i})", color='purple', 
                             linestyle='-', marker='o', markersize=5, linewidth=2, 
                             show_indices=True, fill_polygon=False)

        # --- Plot Circles ---
        plot_circles(ax, removed_circles, radius, color='red', label='Removed Circle', linestyle='--')
        plot_circles(ax, fillet_circles, radius, color='green', label='Fillet Circle', linestyle=':')

        # --- Subplot Finalization ---
        ax.set_title(f"{result_file} (r={radius})")
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