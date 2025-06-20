import matplotlib.pyplot as plt

def plot_polygon(ax, points_list, label, color, closed=True, linestyle='-', marker='o', markersize=5, linewidth=1.5, show_indices=False):
    """Plots a single polygon."""
    if not points_list: return
    
    x_coords = [p[0] for p in points_list]
    y_coords = [p[1] for p in points_list]
    
    if closed and len(points_list) > 1:
        x_coords.append(points_list[0][0]) # Append the x-coord of the first point
        y_coords.append(points_list[0][1]) # Append the y-coord of the first point
    
    ax.plot(x_coords, y_coords, marker=marker, linestyle=linestyle, label=label, color=color, markersize=markersize, linewidth=linewidth)
    
    if show_indices:
        for i, p in enumerate(points_list):
            # Use p[0] and p[1] for text positioning
            ax.text(p[0] + 0.5, p[1] + 0.5, str(i), fontsize=9, color='darkslategrey')


if __name__ == "__main__":
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7.5))

    star_shape = [
        [100, 0], [120, 60], [180, 70], [140, 110],
        [150, 170], [100, 140], [50, 170], [60, 110],
        [20, 70], [80, 60]
    ]

    plot_polygon(ax1, star_shape, "", "blue", show_indices=True)
    ax1.legend(); ax1.grid(True); ax1.set_aspect('equal', adjustable='box')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()