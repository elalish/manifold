import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any



def plot_polygon(ax, points_list, label, color, closed=True, linestyle='-', marker='o', markersize=5, linewidth=1.5, show_indices=False, index_offset=0.2):
    if not points_list:
        return

    # Use numpy array for easier vector calculations
    points = np.array(points_list, dtype=float)
    
    # Prepare coordinates for plotting
    plot_coords = np.vstack([points, points[0]]) if closed and len(points) > 1 else points
    
    ax.plot(plot_coords[:, 0], plot_coords[:, 1], marker=marker, linestyle=linestyle, label=label, color=color, markersize=markersize, linewidth=linewidth)
    
    # Plot vertex indices if requested
    if show_indices:
        num_points = len(points)
        if num_points == 0:
            return

        # Get a colormap. 'viridis' is a perceptually uniform colormap,
        # making it excellent for identifying distinct colors, even for viewers with color vision deficiency.
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
                # Add the normal of the incoming edge (points "outward" for CCW polygons)
                normal_vec += np.array([edge_in[1], -edge_in[0]])
                
            if p_next is not None:
                # Vector from current to next point
                edge_out = p_next - p_current
                # Add the normal of the outgoing edge
                normal_vec += np.array([edge_out[1], -edge_out[0]])

            # Normalize the resulting vector to get a direction
            norm_magnitude = np.linalg.norm(normal_vec)
            if norm_magnitude > 1e-9: # Avoid division by zero for zero-length vectors
                unit_normal = normal_vec / norm_magnitude
            else:
                unit_normal = np.array([0.0, 0.0]) # Default case, should not happen with valid polygons

            # Calculate the final text position
            text_pos = p_current + unit_normal * index_offset
            
            # Determine the color for the text based on its index.
            # This maps the index to a color in the chosen colormap.
            text_color = colormap(i / (num_points - 1)) if num_points > 1 else colormap(0.0)
            
            # Add a bounding box to the text to prevent visual overlap with lines
            ax.text(text_pos[0], text_pos[1], str(i), fontsize=9, color=text_color, ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.1'))



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
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 15))

    result = read_polygon_tests_file("result.txt")
    
    Poly = [
    [0, 0] ,[0, 5] ,[5, 5] ,[5, 0] ,[1, 4.5] ,[0.975528, 4.65451] ,[0.904508, 4.79389] ,[0.793893, 4.90451] ,[0.654508, 4.97553] ,[0.5, 5] ,[0.345492, 4.97553] ,[0.206107, 4.90451] ,[0.0954915, 4.79389] ,[0.0244717, 4.65451] ,[0, 4.5] ,[0.0244717, 4.34549] ,[0.0954915, 4.20611] ,[0.206107, 4.09549] ,[0.345492, 4.02447] ,[0.5, 4] ,[0.654508, 4.02447] ,[0.793893, 4.09549] ,[0.904508, 4.20611] ,[0.975528, 4.34549] ,[1, 4.5] ,[0.975528, 4.65451] ,[0.904508, 4.79389] ,[0.793893, 4.90451] ,[0.654508, 4.97553] ,[0.5, 5] ,[0.345492, 4.97553] ,[0.206107, 4.90451] ,[0.0954915, 4.79389] ,[0.0244717, 4.65451] ,[0, 4.5] ,[0.0244717, 4.34549] ,[0.0954915, 4.20611] ,[0.206107, 4.09549] ,[0.345492, 4.02447] ,[0.5, 4] ,[0.654508, 4.02447] ,[0.793893, 4.09549] ,[0.904508, 4.20611] ,[0.975528, 4.34549] ,[5, 4.5] ,[4.97553, 4.65451] ,[4.90451, 4.79389] ,[4.79389, 4.90451] ,[4.65451, 4.97553] ,[4.5, 5] ,[4.34549, 4.97553] ,[4.20611, 4.90451] ,[4.09549, 4.79389] ,[4.02447, 4.65451] ,[4, 4.5] ,[4.02447, 4.34549] ,[4.09549, 4.20611] ,[4.20611, 4.09549] ,[4.34549, 4.02447] ,[4.5, 4] ,[4.65451, 4.02447] ,[4.79389, 4.09549] ,[4.90451, 4.20611] ,[4.97553, 4.34549] ,[5, 4.5] ,[4.97553, 4.65451] ,[4.90451, 4.79389] ,[4.79389, 4.90451] ,[4.65451, 4.97553] ,[4.5, 5] ,[4.34549, 4.97553] ,[4.20611, 4.90451] ,[4.09549, 4.79389] ,[4.02447, 4.65451] ,[4, 4.5] ,[4.02447, 4.34549] ,[4.09549, 4.20611] ,[4.20611, 4.09549] ,[4.34549, 4.02447] ,[4.5, 4] ,[4.65451, 4.02447] ,[4.79389, 4.09549] ,[4.90451, 4.20611] ,[4.97553, 4.34549] ,[5, 0.5] ,[4.97553, 0.654508] ,[4.90451, 0.793893] ,[4.79389, 0.904508] ,[4.65451, 0.975528] ,[4.5, 1] ,[4.34549, 0.975528] ,[4.20611, 0.904508] ,[4.09549, 0.793893] ,[4.02447, 0.654508] ,[4, 0.5] ,[4.02447, 0.345492] ,[4.09549, 0.206107] ,[4.20611, 0.0954915] ,[4.34549, 0.0244717] ,[4.5, 1.11022e-16] ,[4.65451, 0.0244717] ,[4.79389, 0.0954915] ,[4.90451, 0.206107] ,[4.97553, 0.345492] ,[5, 0.5] ,[4.97553, 0.654508] ,[4.90451, 0.793893] ,[4.79389, 0.904508] ,[4.65451, 0.975528] ,[4.5, 1] ,[4.34549, 0.975528] ,[4.20611, 0.904508] ,[4.09549, 0.793893] ,[4.02447, 0.654508] ,[4, 0.5] ,[4.02447, 0.345492] ,[4.09549, 0.206107] ,[4.20611, 0.0954915] ,[4.34549, 0.0244717] ,[4.5, 1.11022e-16] ,[4.65451, 0.0244717] ,[4.79389, 0.0954915] ,[4.90451, 0.206107] ,[4.97553, 0.345492] ,[1, 0.5] ,[0.975528, 0.654508] ,[0.904508, 0.793893] ,[0.793893, 0.904508] ,[0.654508, 0.975528] ,[0.5, 1] ,[0.345492, 0.975528] ,[0.206107, 0.904508] ,[0.0954915, 0.793893] ,[0.0244717, 0.654508] ,[1.11022e-16, 0.5] ,[0.0244717, 0.345492] ,[0.0954915, 0.206107] ,[0.206107, 0.0954915] ,[0.345492, 0.0244717] ,[0.5, 0] ,[0.654508, 0.0244717] ,[0.793893, 0.0954915] ,[0.904508, 0.206107] ,[0.975528, 0.345492] ,[1, 0.5] ,[0.975528, 0.654508] ,[0.904508, 0.793893] ,[0.793893, 0.904508] ,[0.654508, 0.975528] ,[0.5, 1] ,[0.345492, 0.975528] ,[0.206107, 0.904508] ,[0.0954915, 0.793893] ,[0.0244717, 0.654508] ,[1.11022e-16, 0.5] ,[0.0244717, 0.345492] ,[0.0954915, 0.206107] ,[0.206107, 0.0954915] ,[0.345492, 0.0244717] ,[0.5, 0] ,[0.654508, 0.0244717] ,[0.793893, 0.0954915] ,[0.904508, 0.206107] ,[0.975528, 0.345492]
    ]

    plot_polygon(ax1, result[0]["polygons"][0], "", "blue", show_indices=True)
    # plot_polygon(ax1, Poly, "", "blue", show_indices=False)

    # ax1.legend(); 
    ax1.grid(True); 
    ax1.set_aspect('equal', adjustable='box')

    plot_polygon(ax2, result[1]["polygons"][0], "", "blue", show_indices=True)
    # ax2.legend(); 
    ax2.grid(True); 
    ax2.set_aspect('equal', adjustable='box')


    # plt.tight_layout(rect=[0, 1, 0, 0.96])
    plt.savefig("r.png")