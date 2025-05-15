#!/usr/bin/env python3
"""
Example script for testing
"""

import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from main import FacebookEgoNetwork, BayesianLogisticRegression
from visualization import (
    visualize_ego_network,
)


def run_example(data_dir="./data/facebook", ego_id=0):
    """
    Run a  example of our link prediction model

    Args:
        data_dir: Directory containing the Facebook ego network data
        ego_id: ID of the ego node to analyze
    """

    # Step 1: Load the data
    print("\n--- Step 1: Loading the ego network data ---")
    loader = FacebookEgoNetwork(data_dir)

    try:
        G = loader.load_ego_network(ego_id)
        print(f"Successfully loaded ego network with ID {ego_id}")
        print(f"Network statistics:")
        print(f"  - Number of nodes: {G.number_of_nodes()}")
        print(f"  - Number of edges: {G.number_of_edges()}")
        print(
            f"  - Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}"
        )

        # Save a visualization of the ego network
        print("Visualizing the ego network...")
        vis_output = f"ego_network_{ego_id}.png"
        visualize_ego_network(G, ego_id, output_file=vis_output)

        # Step 2: Generate training and testing data
        print("\n--- Step 2: Generating training and testing data ---")
        X_train, y_train, X_test, y_test = loader.generate_training_data(test_size=0.2)
        print(
            f"Generated {len(X_train)} training examples and {len(X_test)} testing examples"
        )
        print(f"Feature dimensionality: {X_train.shape[1]}")
        print(
            f"Class balance in training: {y_train.sum().item()}/{len(y_train)} positive examples"
        )
        print(
            f"Class balance in testing: {y_test.sum().item()}/{len(y_test)} positive examples"
        )

        # TODO

        # Define feature names
        feature_names = [
            "Common Neighbors",
            "Jaccard Coefficient",
            "Adamic-Adar Index",
            "Preferential Attachment",
        ]

        # Step 3: Train the Bayesian logistic regression model

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(
            "Please check that the Facebook ego network data files are in the correct directory."
        )
        return None, None, None, None, None, None, None, None


def main():
    # Get data directory from command line arguments or use default
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./data/facebook"

    # Run the  example
    run_example(data_dir=data_dir)


if __name__ == "__main__":
    main()
