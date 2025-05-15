#!/usr/bin/env python3
"""
Visualization utilities for the Facebook ego network data
"""

import os
import networkx as nx
import matplotlib.pyplot as plt


def visualize_ego_network(G, ego_id, output_file=None):
    """
    Visualize the ego network centered around the ego node

    Args:
        G: NetworkX graph
        ego_id: ID of the ego node
        output_file: Path to save the visualization (optional)
    """
    plt.figure(figsize=(12, 12))

    # Position nodes using spring layout
    pos = nx.spring_layout(G, seed=42)

    # Draw nodes
    ego_nodes = [ego_id]
    other_nodes = [n for n in G.nodes() if n != ego_id]

    # Draw ego node with larger size
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=ego_nodes,
        node_size=500,
        node_color="red",
        alpha=0.8,
        label="Ego Node",
    )

    # Draw other nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=other_nodes,
        node_size=200,
        node_color="blue",
        alpha=0.6,
        label="Friend Nodes",
    )

    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.3)

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10)

    plt.title(f"Ego Network for Node {ego_id}")
    plt.legend()
    plt.axis("off")

    if output_file:
        plt.savefig(output_file)
        print(f"Saved ego network visualization to {output_file}")
    else:
        plt.show()


if __name__ == "__main__":
    print(
        "This is a utility module. Please import and use these functions from your main script."
    )
