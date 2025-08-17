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

def read_and_draw_circles(filename, ax, colors=None):
    """
    Read circle data from a text file and draw them using matplotlib.
    
    File format:
    - First line: radius (float)
    - Following lines: center_x center_y (two floats separated by space)
    
    Parameters:
    filename (str): Path to the text file
    figsize (tuple): Figure size (width, height)
    colors (list): List of colors for circles (optional)
    
    Returns:
    fig, ax: matplotlib figure and axis objects
    """
    circles = []
    
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
            
            # Read radius from first line
            radius = float(lines[0].strip())
            
            # Read circle centers from remaining lines
            centers = []
            for line in lines[1:]:
                line = line.strip()
                if line:  # Skip empty lines
                    parts = line.split()
                    if len(parts) >= 2:
                        x = float(parts[0])
                        y = float(parts[1])
                        centers.append((x, y))
            
            circles = [(center[0], center[1], radius) for center in centers]
            
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None, None
    except ValueError as e:
        print(f"Error reading file: {e}")
        return None, None
    
    if not circles:
        print("No valid circle data found.")
        return None, None
    
    # Set default colors if not provided
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(circles)))
    
    # Draw circles
    for i, (x, y, r) in enumerate(circles):
        color = colors[i % len(colors)]
        circle = Circle((x, y), r, linewidth=2, edgecolor=color, 
                               facecolor=color, alpha=0.3, label=f'Circle {i+1}')
        ax.add_patch(circle)
        
        # Add center point
        ax.plot(x, y, 'ko', markersize=4)
        
        # Add text annotation
        ax.annotate(f'({x:.1f}, {y:.1f})', (x, y), 
                   xytext=(5, 5), textcoords='offset points', fontsize=8)

    return ax

if __name__ == "__main__":
    result = read_polygon_results_file(sys.argv[1])
    # data = read_polygon_results_file(sys.argv[2])

    # result.append(data[0])

    rows = int(len(result) / 5) + 1
    cols = len(result) if len(result) < 5 else 5

    fig, axes = plt.subplots(rows, cols, figsize=(15 * cols, 15 * rows))

    read_and_draw_circles("circle.txt", axes)

    # plot_polygon(ax1, result[0]["polygons"][0], "", "blue", show_indices=True)
    # plot_polygon(ax1, Poly, "", "blue", show_indices=False)

    # ax1.legend(); 
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    
    for j in range(0, len(result)):
      ax = axes if cols == 1 else axes.flatten()[j]
      
      ax.grid(True)
      ax.grid(True); 
      ax.set_aspect('equal', adjustable='box')
      ax.set_title(result[j]["name"])

      for i in range(0, len(result[j]["polygons"])):
        plot_polygon(ax, result[j]["polygons"][i], "", colors[i], show_indices=True, fill_polygon=False)
        ax.grid(True); 
        ax.set_aspect('equal', adjustable='box')

    for i in range(len(result), rows * cols):
      ax = axes.flatten()[i]
      ax.axis('off')

    plt.tight_layout(rect=[0, 1, 0, 0.96])
    plt.savefig("r.png")