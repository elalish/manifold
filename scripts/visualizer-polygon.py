# This python file takes argv[1] as path to polygon file to visualization
# Polygon file format is as same as test/polygons

import matplotlib.pyplot as plt
import numpy as np
import sys
from typing import List, Dict, Any
from matplotlib.path import Path
from matplotlib.patches import PathPatch

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
                
                # For holes (loop_idx > 0), we might want to reverse the normal direction
                # so indices appear on the "inside" of the hole
                if loop_idx > 0:
                    normal_vec = -normal_vec
                
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


def read_polygon_tests_file(filename: str) -> List[Dict[str, Any]]:
    all_tests = []
    try:
        with open(filename, 'r') as f:
            # Read the entire file and split by whitespace to get a flat list of tokens
            tokens = f.read().split()
            
            token_idx = 0
            while token_idx < len(tokens):
                # --- Read Test Header ---
                if token_idx + 4 > len(tokens):
                    print(f"Warning: Truncated test header at end of file.")
                    break
                
                test_name = tokens[token_idx]
                expected_num_tri = int(tokens[token_idx + 1])
                epsilon = float(tokens[token_idx + 2])
                num_polys = int(tokens[token_idx + 3])
                token_idx += 4
                
                test_data = {
                    "name": test_name,
                    "expectedNumTri": expected_num_tri,
                    "epsilon": epsilon,
                    "polygons": []
                }

                # --- Read Polygons for the Current Test ---
                for _ in range(num_polys):
                    if token_idx >= len(tokens):
                        print(f"Warning: Expected more polygon data for test '{test_name}'.")
                        break
                        
                    num_points = int(tokens[token_idx])
                    token_idx += 1
                    
                    current_polygon = []
                    
                    # Check if there are enough tokens for all points
                    if token_idx + num_points * 2 > len(tokens):
                        print(f"Warning: Truncated point data for a polygon in test '{test_name}'.")
                        token_idx = len(tokens) # Force outer loop to terminate
                        break
                    
                    for _ in range(num_points):
                        x = float(tokens[token_idx])
                        y = float(tokens[token_idx + 1])
                        current_polygon.append([x, y])
                        token_idx += 2
                    
                    test_data["polygons"].append(current_polygon)
                
                all_tests.append(test_data)

    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return []
    except (ValueError, IndexError) as e:
        print(f"Error parsing file: {e}. The file might be malformed.")
        return []
        
    return all_tests

if __name__ == "__main__":
    outer_boundary = [(0, 0), (6, 0), (6, 4), (0, 4)]
    hole1 = [(1, 1), (2, 1), (2, 2), (1, 2)]
    hole2 = [(3, 1), (5, 1), (5, 3), (3, 3)]
    
    loops = [outer_boundary, hole1, hole2]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original style - outline only
    plot_polygon(ax1, loops, "Polygon with Holes", "blue", show_indices=True)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_title('Original Style (Outline)')
    
    # New style - with fill
    plot_polygon(ax2, loops, "Polygon with Holes", "red", show_indices=True, 
                fill_polygon=True, fill_alpha=0.4)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_title('With Fill (Handles Holes Correctly)')
    
    plt.tight_layout()
    plt.savefig("r.png")



    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 20))

    # result = read_polygon_tests_file(sys.argv[1])

    # plot_polygon(ax1, result[0]["polygons"][0], "", "blue", show_indices=True)
    # # plot_polygon(ax1, Poly, "", "blue", show_indices=False)

    # # ax1.legend(); 
    # ax1.grid(True); 
    # ax1.set_aspect('equal', adjustable='box')
    # colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    
    # for i in range(1,len(result[0]["polygons"])):
    #   plot_polygon(ax2, result[0]["polygons"][i], "", colors[i], show_indices=True)
    #   # ax2.legend(); 
    #   ax2.grid(True); 
    #   ax2.set_aspect('equal', adjustable='box')
    
    # ax2.set_xlim(ax1.get_xlim())
    # ax2.set_ylim(ax1.get_ylim())
    

    # # plot_polygon(ax2, result[0]["polygons"][2], "", "blue", show_indices=True)
    # # # ax2.legend(); 
    # # ax2.grid(True); 
    # # ax2.set_aspect('equal', adjustable='box')

    # # plot_polygon(ax2, result[0]["polygons"][4], "", "blue", show_indices=True)
    # # # ax2.legend(); 
    # # ax2.grid(True); 
    # # ax2.set_aspect('equal', adjustable='box')

    # # plot_polygon(ax3, result[2]["polygons"][0], "", "blue", show_indices=True)
    # # # ax2.legend(); 
    # # ax3.grid(True); 
    # # ax3.set_aspect('equal', adjustable='box')

    # # plot_polygon(ax3, result[3]["polygons"][0], "", "blue", show_indices=True)
    # # # ax2.legend(); 
    # # ax3.grid(True); 
    # # ax3.set_aspect('equal', adjustable='box')


    # # plt.tight_layout(rect=[0, 1, 0, 0.96])
    # plt.savefig("r.png")