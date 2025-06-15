#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import networkx as nx
import torch
import pyro
from matplotlib.colors import LinearSegmentedColormap
import random
import networkx as nx


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


def get_prediction_probabilities_and_uncertainty(model, guide, X, num_samples=1000):
    predictive = pyro.infer.Predictive(model, guide=guide, num_samples=num_samples)
    samples = predictive(X)
    all_samples = samples["obs"].float()

    mean_probs = all_samples.mean(dim=0)
    std_probs = all_samples.std(dim=0)

    return mean_probs, std_probs, all_samples


def visualize_link_prediction_network(
    G,
    model,
    guide,
    loader,
    ego_id,
    num_predictions=50,
    output_file=None,
    use_embeddings=True,
    embedding_dim=16,
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    embeddings = (
        loader.compute_node2vec_embeddings(dimensions=embedding_dim)
        if use_embeddings
        else None
    )
    extractor = lambda u, v: (
        loader.extract_link_features_with_embeddings(u, v, embeddings)
        if use_embeddings
        else loader.extract_link_features
    )

    nodes = list(G.nodes())
    non_edges = []
    while len(non_edges) < num_predictions:
        u, v = random.sample(nodes, 2)
        if not G.has_edge(u, v) and u != v:
            non_edges.append((u, v))

    X_pred = torch.stack([extractor(u, v) for u, v in non_edges])
    mean_probs, std_probs, _ = get_prediction_probabilities_and_uncertainty(
        model, guide, X_pred
    )

    pos = nx.spring_layout(G, seed=42, k=1, iterations=50)

    ax1.set_title(
        f"Original Ego Network (Node {ego_id})", fontsize=14, fontweight="bold"
    )

    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color="lightgray", ax=ax1)

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=[ego_id],
        node_size=800,
        node_color="red",
        alpha=0.8,
        ax=ax1,
        label="Ego Node",
    )

    other_nodes = [n for n in G.nodes() if n != ego_id]
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=other_nodes,
        node_size=200,
        node_color="lightblue",
        alpha=0.7,
        ax=ax1,
        label="Friend Nodes",
    )

    ax1.legend()
    ax1.axis("off")

    ax2.set_title(
        "Network + Predicted Links\n(Color: Probability, Width: Confidence)",
        fontsize=14,
        fontweight="bold",
    )

    nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color="lightgray", ax=ax2)

    nx.draw_networkx_nodes(
        G, pos, nodelist=[ego_id], node_size=800, node_color="red", alpha=0.8, ax=ax2
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=other_nodes,
        node_size=200,
        node_color="lightblue",
        alpha=0.7,
        ax=ax2,
    )

    high_prob_edges = []
    edge_colors = []
    edge_widths = []

    for i, (u, v) in enumerate(non_edges):
        prob = mean_probs[i].item()
        uncertainty = std_probs[i].item()

        if prob > 0.1:
            high_prob_edges.append((u, v))
            edge_colors.append(prob)
            edge_widths.append(max(0.5, 5 * (1 - uncertainty)))

    if high_prob_edges:
        colors = ["yellow", "orange", "red", "darkred"]
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list("prob_cmap", colors, N=n_bins)

        edges = nx.draw_networkx_edges(
            G.edge_subgraph([]).copy(),
            pos,
            edgelist=high_prob_edges,
            edge_color=edge_colors,
            edge_cmap=cmap,
            edge_vmin=0,
            edge_vmax=1,
            width=edge_widths,
            alpha=0.8,
            ax=ax2,
        )

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax2, shrink=0.8)
        cbar.set_label("Link Probability", fontsize=12)

    ax2.axis("off")

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Saved link prediction network visualization to {output_file}")
    else:
        plt.show()


def plot_uncertainty_distribution(mean_probs, std_probs, output_file=None):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    mean_probs_np = mean_probs.cpu().numpy()
    std_probs_np = std_probs.cpu().numpy()

    ax1.hist(mean_probs_np, bins=30, alpha=0.7, color="skyblue", edgecolor="black")
    ax1.set_xlabel("Prediction Probability")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Distribution of Link Prediction Probabilities")
    ax1.grid(True, alpha=0.3)

    ax2.hist(std_probs_np, bins=30, alpha=0.7, color="lightcoral", edgecolor="black")
    ax2.set_xlabel("Prediction Uncertainty (Std Dev)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Distribution of Prediction Uncertainties")
    ax2.grid(True, alpha=0.3)

    ax3.scatter(mean_probs_np, std_probs_np, alpha=0.6, color="purple")
    ax3.set_xlabel("Prediction Probability")
    ax3.set_ylabel("Prediction Uncertainty")
    ax3.set_title("Probability vs Uncertainty")
    ax3.grid(True, alpha=0.3)

    z = np.polyfit(mean_probs_np, std_probs_np, 1)
    p = np.poly1d(z)
    ax3.plot(mean_probs_np, p(mean_probs_np), "r--", alpha=0.8)

    n_show = min(20, len(mean_probs_np))
    indices = np.random.choice(len(mean_probs_np), n_show, replace=False)
    indices = sorted(indices)

    x_pos = range(n_show)
    means_subset = mean_probs_np[indices]
    stds_subset = std_probs_np[indices]

    ax4.errorbar(
        x_pos, means_subset, yerr=stds_subset, fmt="o", capsize=5, capthick=2, alpha=0.7
    )
    ax4.set_xlabel("Prediction Index")
    ax4.set_ylabel("Probability ± Uncertainty")
    ax4.set_title(f"Confidence Intervals (Random {n_show} Predictions)")
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-0.1, 1.1)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Saved uncertainty distribution plots to {output_file}")
    else:
        plt.show()


def plot_posterior_samples_distribution(
    all_samples, indices_to_show=None, output_file=None
):
    all_samples_np = all_samples.cpu().numpy()

    if indices_to_show is None:
        indices_to_show = np.random.choice(
            all_samples_np.shape[1], min(6, all_samples_np.shape[1]), replace=False
        )

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, idx in enumerate(indices_to_show):
        if i >= len(axes):
            break

        samples = all_samples_np[:, idx]

        axes[i].hist(
            samples,
            bins=30,
            alpha=0.7,
            density=True,
            color="lightgreen",
            edgecolor="black",
        )

        mean_val = np.mean(samples)
        ci_lower = np.percentile(samples, 2.5)
        ci_upper = np.percentile(samples, 97.5)

        axes[i].axvline(
            mean_val,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_val:.3f}",
        )
        axes[i].axvline(
            ci_lower,
            color="orange",
            linestyle=":",
            alpha=0.8,
            label=f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]",
        )
        axes[i].axvline(ci_upper, color="orange", linestyle=":", alpha=0.8)

        axes[i].set_title(f"Posterior Distribution - Prediction {idx}")
        axes[i].set_xlabel("Probability")
        axes[i].set_ylabel("Density")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    for i in range(len(indices_to_show), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Saved posterior samples visualization to {output_file}")
    else:
        plt.show()


def create_uncertainty_heatmap(
    G,
    model,
    guide,
    loader,
    ego_id,
    grid_size=20,
    output_file=None,
    use_embeddings=True,
    embedding_dim=16,
):
    pos = nx.spring_layout(G, seed=42)

    x_coords = [pos[node][0] for node in G.nodes()]
    y_coords = [pos[node][1] for node in G.nodes()]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    x_grid = np.linspace(x_min, y_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)

    nodes = list(G.nodes())

    embeddings = (
        loader.compute_node2vec_embeddings(dimensions=embedding_dim)
        if use_embeddings
        else None
    )
    extractor = lambda u, v: (
        loader.extract_link_features_with_embeddings(u, v, embeddings)
        if use_embeddings
        else loader.extract_link_features
    )

    sample_pairs = []
    for _ in range(min(100, len(nodes) * (len(nodes) - 1) // 2)):
        u, v = random.sample(nodes, 2)
        if not G.has_edge(u, v):
            sample_pairs.append((u, v))

    if sample_pairs:
        X_sample = torch.stack([extractor(u, v) for u, v in sample_pairs])
        mean_probs, std_probs, _ = get_prediction_probabilities_and_uncertainty(
            model, guide, X_sample
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        nx.draw_networkx_nodes(G, pos, node_size=300, node_color="lightblue", ax=ax1)
        nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax1)

        high_prob_indices = mean_probs > 0.3
        if high_prob_indices.any():
            high_prob_pairs = [
                sample_pairs[i]
                for i in range(len(sample_pairs))
                if high_prob_indices[i]
            ]
            high_prob_values = mean_probs[high_prob_indices]

            colors = plt.cm.viridis(high_prob_values.cpu().numpy())
            for (u, v), color in zip(high_prob_pairs, colors):
                ax1.plot(
                    [pos[u][0], pos[v][0]],
                    [pos[u][1], pos[v][1]],
                    color=color,
                    linewidth=3,
                    alpha=0.8,
                )

        ax1.set_title("Network with High-Probability Predictions")
        ax1.axis("off")

        ax2.scatter(
            [pos[u][0] for u, v in sample_pairs],
            [pos[u][1] for u, v in sample_pairs],
            c=std_probs.cpu().numpy(),
            cmap="Reds",
            s=100,
            alpha=0.7,
        )

        nx.draw_networkx_nodes(
            G, pos, node_size=100, node_color="lightgray", alpha=0.5, ax=ax2
        )
        nx.draw_networkx_edges(G, pos, alpha=0.2, ax=ax2)

        cbar = plt.colorbar(plt.cm.ScalarMappable(cmap="Reds"), ax=ax2)
        cbar.set_label("Prediction Uncertainty")
        ax2.set_title("Prediction Uncertainty Heatmap")
        ax2.axis("off")

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            print(f"Saved uncertainty heatmap to {output_file}")
        else:
            plt.show()


def comprehensive_link_prediction_analysis(
    loader, model, guide, ego_id, output_dir="./visualizations"
):

    print(f"Running comprehensive visualization analysis for ego {ego_id}...")

    G = loader.load_ego_network(ego_id)

    nodes = list(G.nodes())
    test_pairs = []
    for _ in range(100):
        u, v = random.sample(nodes, 2)
        if not G.has_edge(u, v):
            test_pairs.append((u, v))

    embeddings = loader.compute_node2vec_embeddings(dimensions=16)
    X_test = torch.stack(
        [
            loader.extract_link_features_with_embeddings(u, v, embeddings)
            for u, v in test_pairs
        ]
    )

    mean_probs, std_probs, all_samples = get_prediction_probabilities_and_uncertainty(
        model, guide, X_test, num_samples=1000
    )

    print("Creating network visualization...")
    visualize_link_prediction_network(
        G,
        model,
        guide,
        loader,
        ego_id,
        output_file=f"{output_dir}/network_predictions_ego_{ego_id}.png",
    )

    print("Creating uncertainty distribution plots...")
    plot_uncertainty_distribution(
        mean_probs,
        std_probs,
        output_file=f"{output_dir}/uncertainty_distribution_ego_{ego_id}.png",
    )

    print("Creating posterior samples visualization...")
    plot_posterior_samples_distribution(
        all_samples, output_file=f"{output_dir}/posterior_samples_ego_{ego_id}.png"
    )

    print("Creating uncertainty heatmap...")
    create_uncertainty_heatmap(
        G,
        model,
        guide,
        loader,
        ego_id,
        output_file=f"{output_dir}/uncertainty_heatmap_ego_{ego_id}.png",
    )

    print(f"All visualizations saved to {output_dir}/")

    summary = {
        "ego_id": ego_id,
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "num_predictions": len(mean_probs),
        "mean_probability": mean_probs.mean().item(),
        "mean_uncertainty": std_probs.mean().item(),
        "high_confidence_predictions": (std_probs < 0.1).sum().item(),
        "high_probability_predictions": (mean_probs > 0.7).sum().item(),
    }

    print("\nSummary Statistics:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    return summary
