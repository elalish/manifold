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
            patch.set_label(f"{label} (filled)")
            ax.add_patch(patch)
    
    # Track global index for continuous numbering across all loops
    global_index = 0
    
    for loop_idx, points_list in enumerate(loops_list):
        if not points_list:
            continue
            
        # Use numpy array for easier vector calculations
        points = np.array(points_list, dtype=float)
        
        # Prepare coordinates for plotting
        plot_coords = np.vstack([points, points[0]]) if closed and len(points) > 1 else points
        
        # Determine line style for holes (you can customize this)
        current_linestyle = linestyle
        current_label = label if loop_idx == 0 and not fill_polygon else None  # Only label the outer boundary
        
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


def read_polygon_results_file(filename: str) -> List[Dict[str, Any]]:
    """
    Reads a test file generated by the C++ code.

    The file has the following structure:
    TestName NumCrossSections
    NumPaths_in_CS1
    NumPoints_in_Path1
    x1 y1
    x2 y2
    ...
    """
    all_tests = []
    try:
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        line_idx = 0
        while line_idx < len(lines):
            # --- Read Test Header ---
            # Reads "TestName NumCrossSections"
            header_parts = lines[line_idx].split()
            test_name = header_parts[0]
            num_cross_sections = int(header_parts[1])
            line_idx += 1

            current_test = {
                "name": test_name,
                "polygons": []
            }

            # --- Read CrossSections for the Current Test ---
            for _ in range(num_cross_sections):
                # Reads the number of paths (polygons) in this cross-section
                num_paths = int(lines[line_idx])
                line_idx += 1
                
                current_cross_section_paths = []
                for _ in range(num_paths):
                    # Reads the number of points in the current path
                    num_points = int(lines[line_idx])
                    line_idx += 1

                    current_polygon = []
                    # Reads all points for the current polygon
                    for _ in range(num_points):
                        point_coords = lines[line_idx].split()
                        x = float(point_coords[0])
                        y = float(point_coords[1])
                        current_polygon.append([x, y])
                        line_idx += 1
                    
                    current_cross_section_paths.append(current_polygon)
                
                current_test["polygons"].append(current_cross_section_paths)
            
            all_tests.append(current_test)

    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return []
    except (ValueError, IndexError) as e:
        print(f"Error parsing file at line {line_idx + 1}: {e}. The file might be malformed.")
        return []
        
    return all_tests
def read_and_draw_circles(filename, axes_array, colors=None):
    """
    Reads circle data from a text file and draws them on each subplot in an array.

    The file should contain a single radius on the first line, followed by
    x y center coordinates on subsequent lines.

    Parameters:
    filename (str): Path to the text file.
    axes_array (np.ndarray): A NumPy array of Matplotlib Axes objects to draw on.
    colors (list): Optional list of colors for the circles.
    """
    # --- 1. Read and Parse Circle Data (This part is done only once) ---
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
            if not lines:
                print(f"Error: File '{filename}' is empty.")
                return
            
            radius = float(lines[0].strip())
            
            centers = []
            for line in lines[1:]:
                if line.strip():  # Skip empty lines
                    parts = line.split()
                    if len(parts) >= 2:
                        centers.append((float(parts[0]), float(parts[1])))
            
            if not centers:
                print(f"Warning: No center points found in '{filename}'.")
                return

            # Create a list of circle properties: (x, y, radius)
            circles_to_draw = [(center[0], center[1], radius) for center in centers]

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return
    except (ValueError, IndexError) as e:
        print(f"Error processing file '{filename}': {e}")
        return

    # --- 2. Draw the Circles on Each Subplot ---
    # Use a clear name for the individual subplot in the loop.
    for single_ax in axes_array.flatten():
        
        # This inner loop draws ALL circles onto the CURRENT subplot.
        for i, (x, y, r) in enumerate(circles_to_draw):
            # Use a default color cycle if none is provided.
            color = colors[i % len(colors)] if colors else plt.cm.tab10(i)

            # Create a NEW patch for each circle on each subplot.
            circle_patch = Circle((x, y), r, linewidth=2, edgecolor=color,
                                  facecolor=color, alpha=0.25)
            
            # Use the individual subplot 'single_ax' for all drawing commands.
            single_ax.add_patch(circle_patch)
            single_ax.plot(x, y, 'o', color=color, markersize=5, label=f'Center {i+1}')
            single_ax.annotate(f'({x:.2f}, {y:.2f})', (x, y),
                             xytext=(5, 5), textcoords='offset points',
                             fontsize=8, color='black')


# CORRECTED FUNCTION: It now uses the 'polygons' argument passed to it.
def draw_polygon_file(axes, polygons_data, rows, cols):
    """
    Draws a set of polygons onto a given set of axes.
    - axes: The matplotlib axes array to draw on.
    - polygons_data: The list of polygon data to plot (e.g., your 'input' or 'result').
    - rows, cols: The dimensions of this specific axes grid.
    """
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k'] # Removed 'w' (white) as it's usually invisible
    
    # Flatten the axes array for easy iteration, this works for 1D and 2D arrays.
    flat_axes = axes.flatten()

    # Loop over the data that was actually PASSED to the function
    for j in range(len(polygons_data)):
        ax = flat_axes[j]
        
        ax.grid(True)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(polygons_data[j]["name"])

        # Plot each polygon within the current item
        for i in range(len(polygons_data[j]["polygons"])):
            # Make sure we don't run out of colors
            color = colors[i % len(colors)]
            plot_polygon(ax, polygons_data[j]["polygons"][i], "", color, show_indices=True, fill_polygon=False)

    # Turn off any unused axes in this grid section
    for i in range(len(polygons_data), rows * cols):
        ax = flat_axes[i]
        ax.axis('off')

if __name__ == "__main__":
    # Read the input and result data files
    input_data = read_polygon_results_file(sys.argv[1])
    result_data = read_polygon_results_file(sys.argv[2])

    # --- Grid Calculation ---
    # The grid size should be determined by the larger of the two datasets
    # to ensure there's enough space for everything.
    num_items = max(len(input_data), len(result_data))
    cols = 5 if num_items > 4 else num_items
    if cols == 0: # Avoid division by zero if there's no data
        rows = 0
    else:
        rows = (num_items + cols - 1) // cols # This is a robust way to calculate rows
    
    if rows == 0:
        print("No data to plot. Exiting.")
        sys.exit()

    # --- Plot Creation ---
    # Create ONE figure with enough columns for both input and result
    fig, allAxes = plt.subplots(rows, 2 * cols, figsize=(15 * 2 * cols, 15 * rows))

    # Handle the case of a single row/column, where subplots doesn't return a 2D array
    if rows == 1:
        allAxes = allAxes.reshape(1, -1)

    # Slice the axes array for input (left) and result (right)
    inputAxes = allAxes[:, :cols]
    resultAxes = allAxes[:, cols:]

    # --- Drawing ---
    # Draw the circles if that function exists and is needed
    # read_and_draw_circles("circle.txt", inputAxes)

    # Call the FIXED function with the correct data
    print("Drawing input polygons...")
    draw_polygon_file(inputAxes, input_data, rows, cols)
    
    read_and_draw_circles("circle.txt", inputAxes)
    
    print("Drawing result polygons...")
    draw_polygon_file(resultAxes, result_data, rows, cols)
    
    # --- Finalization and Saving ---
    fig.suptitle('Input vs. Result Comparison', fontsize=24)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust rect for suptitle
    plt.savefig("r.png")
    print("Saved plot to r.png")
    # plt.show() # Uncomment to display the plot interactively