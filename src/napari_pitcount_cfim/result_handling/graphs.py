from pathlib import Path

import matplotlib.pyplot as plt
# try optional adjustText import
from adjustText import adjust_text


# Assuming avg_vals, pct_vals, and graph_groups are defined as lists of floats and labels:
# avg_vals = [...]  # average pits per cell
# pct_vals = [...]  # percent of cells with pits
# graph_groups = [...]  # labels for each data point


def modern_graph(graph_label, avg_vals, pct_vals, graph_groups, output_folder):
    """
    Create a modern scatter plot with enhanced readability and aesthetics.

    Parameters:
        graph_label (str): Label for the graph title.
        avg_vals (list): List of average pits per cell.
        pct_vals (list): List of percentage of cells with pits.
        graph_groups (list): List of labels for each data point.
        output_folder (str): Folder where the graph image will be saved.
    """
    # Ensure input lists are of the same length
    if not (len(avg_vals) == len(pct_vals) == len(graph_groups)):
        raise ValueError("Input lists must have the same length.")
    # Use a clean, modern style with gridlines
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 6))

    # Optionally color-code points by group (using prefix of label before the first space, if applicable)
    groups = [label.split()[0] for label in graph_groups]
    unique_groups = []
    for g in groups:
        if g not in unique_groups:
            unique_groups.append(g)
    cmap = plt.get_cmap('tab10')  # qualitative colormap for up to 10 distinct groups
    color_map = {group: cmap(i) for i, group in enumerate(unique_groups)}
    colors = [color_map[g] for g in groups]

    # Plot the scatter with larger markers and an outline for clarity
    ax.scatter(avg_vals, pct_vals, c=colors, s=100, edgecolors='black', linewidth=0.7)

    # Add data point labels, then adjust to prevent overlap
    texts = [ax.text(x, y, label, fontsize=12) for x, y, label in zip(avg_vals, pct_vals, graph_groups)]
    # adjust labels if adjustText is installed

    adjust_text(texts, ax=ax, expand_points=(2, 2),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1))

    # Increase font sizes for axes labels and title
    ax.set_xlabel('Average pits per cell', fontsize=14)
    ax.set_ylabel('Cells with pits (%)', fontsize=14)
    ax.set_title(f'Statistics ({graph_label})', fontsize=16)

    # Improve tick label readability and add dashed gridlines
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)

    # Save the figure as a high-resolution PNG
    graph_path = Path(output_folder) / f"stats_graph_{graph_label}.png"
    plt.tight_layout()
    plt.savefig(graph_path, dpi=300)
    plt.close(fig)


def simple_graph(graph_label, avg_vals, pct_vals, graph_groups, output_folder):
    """
    Basic scatter plot of avg pits vs percent with pits.
    """
    plt.figure()
    plt.scatter(avg_vals, pct_vals)
    for i, label in enumerate(graph_groups):
        plt.annotate(label, (avg_vals[i], pct_vals[i]))
    plt.xlabel('Average pits per cell')
    plt.ylabel('Cells with pits (%)')
    plt.title(f'Statistics ({graph_label})')
    graph_path = Path(output_folder) / f'stats_graph_{graph_label}.png'
    plt.savefig(graph_path)
    plt.close()
    print(f"Graph saved to {graph_path}")

