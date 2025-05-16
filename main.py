#!/usr/bin/env python3
"""
CS267 Project: Using PPLs to Model Social Ego-Networks
Main file for loading data, training models, and evaluating link prediction
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
import networkx as nx
from sklearn.model_selection import train_test_split
from pyro.nn import PyroModule, PyroSample
from pyro.infer import SVI

# For sanity check without training:
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import Trace_ELBO

# Set random seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
pyro.set_rng_seed(RANDOM_SEED)


class FacebookEgoNetwork:
    """Loader for Facebook ego network data"""

    def __init__(self, data_dir):
        """
        Initialize the data loader

        Args:
            data_dir: Directory containing the Facebook ego network data
        """
        self.data_dir = data_dir
        self.graph = None
        self.features = {}
        self.ego_features = {}
        self.feature_names = {}
        self.circles = {}

    def load_ego_network(self, ego_id):
        """
        Load the ego network for a specific ego node

        Args:
            ego_id: ID of the ego node

        Returns:
            NetworkX graph of the ego network
        """
        print(f"Loading ego network for node {ego_id}")

        # Load edges
        edges_file = os.path.join(self.data_dir, f"{ego_id}.edges")
        if not os.path.exists(edges_file):
            raise FileNotFoundError(f"Edge file not found: {edges_file}")

        # Create graph
        G = nx.Graph()

        # Add ego node to graph
        G.add_node(ego_id)

        # Read edges
        with open(edges_file, "r") as f:
            for line in f:
                node1, node2 = map(int, line.strip().split())
                G.add_edge(node1, node2)

                # Also add edges from ego node to all other nodes
                G.add_edge(ego_id, node1)
                G.add_edge(ego_id, node2)

        # Load node features
        features_file = os.path.join(self.data_dir, f"{ego_id}.feat")
        self.features[ego_id] = {}

        if os.path.exists(features_file):
            with open(features_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    node_id = int(parts[0])
                    features = list(map(int, parts[1:]))
                    self.features[ego_id][node_id] = features

        # Load ego features
        ego_feat_file = os.path.join(self.data_dir, f"{ego_id}.egofeat")
        if os.path.exists(ego_feat_file):
            with open(ego_feat_file, "r") as f:
                line = f.readline().strip()
                self.ego_features[ego_id] = list(map(int, line.split()))

        # Load feature names
        featnames_file = os.path.join(self.data_dir, f"{ego_id}.featnames")
        self.feature_names[ego_id] = []

        if os.path.exists(featnames_file):
            with open(featnames_file, "r") as f:
                for line in f:
                    self.feature_names[ego_id].append(line.strip())

        # Load circles
        circles_file = os.path.join(self.data_dir, f"{ego_id}.circles")
        self.circles[ego_id] = []

        if os.path.exists(circles_file):
            with open(circles_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    circle_name = parts[0]
                    circle_nodes = list(map(int, parts[1:]))
                    self.circles[ego_id].append((circle_name, circle_nodes))

        self.graph = G
        return G

    def extract_link_features(self, node1, node2):
        """
        Extract features for a potential link between two nodes

        Args:
            node1: First node ID
            node2: Second node ID

        Returns:
            Feature vector for the potential link
        """
        if self.graph is None:
            raise ValueError("No graph loaded. Call load_ego_network first.")

        G = self.graph

        # Common neighbors
        common_neighbors = len(list(nx.common_neighbors(G, node1, node2)))

        # Jaccard coefficient
        neighbors1 = set(G.neighbors(node1))
        neighbors2 = set(G.neighbors(node2))
        jaccard = 0
        if len(neighbors1) > 0 and len(neighbors2) > 0:
            jaccard = len(neighbors1 & neighbors2) / len(neighbors1 | neighbors2)

        # Adamic-Adar index
        aa_index = 0
        for common_neighbor in nx.common_neighbors(G, node1, node2):
            aa_index += 1 / np.log(G.degree(common_neighbor))

        # Preferential attachment
        pref_attachment = G.degree(node1) * G.degree(node2)

        return torch.tensor(
            [common_neighbors, jaccard, aa_index, pref_attachment], dtype=torch.float
        )

    def generate_training_data(self, test_size=0.2):
        """
        Generate training and testing data for link prediction

        Args:
            test_size: Fraction of edges to use for testing

        Returns:
            X_train, y_train, X_test, y_test: Features and labels for training and testing
        """
        if self.graph is None:
            raise ValueError("No graph loaded. Call load_ego_network first.")

        G = self.graph

        # Generate positive examples (existing edges)
        positive_examples = list(G.edges())

        # Generate negative examples (non-existing edges)
        nodes = list(G.nodes())
        negative_examples = []

        # Sample random non-edges
        num_positive = len(positive_examples)
        while len(negative_examples) < num_positive:
            i = random.randint(0, len(nodes) - 1)
            j = random.randint(0, len(nodes) - 1)

            if i != j and not G.has_edge(nodes[i], nodes[j]):
                negative_examples.append((nodes[i], nodes[j]))

        # Split into train and test sets
        pos_train, pos_test = train_test_split(
            positive_examples, test_size=test_size, random_state=RANDOM_SEED
        )
        neg_train, neg_test = train_test_split(
            negative_examples, test_size=test_size, random_state=RANDOM_SEED
        )

        # Create feature vectors
        X_train = []
        y_train = []

        for edge in pos_train:
            X_train.append(self.extract_link_features(edge[0], edge[1]))
            y_train.append(1)

        for edge in neg_train:
            X_train.append(self.extract_link_features(edge[0], edge[1]))
            y_train.append(0)

        X_test = []
        y_test = []

        for edge in pos_test:
            X_test.append(self.extract_link_features(edge[0], edge[1]))
            y_test.append(1)

        for edge in neg_test:
            X_test.append(self.extract_link_features(edge[0], edge[1]))
            y_test.append(0)

        # Convert to tensors
        X_train = torch.stack(X_train)
        y_train = torch.tensor(y_train, dtype=torch.float)
        X_test = torch.stack(X_test)
        y_test = torch.tensor(y_test, dtype=torch.float)

        return X_train, y_train, X_test, y_test


class BayesianLogisticRegression(PyroModule):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = PyroModule[nn.Linear](input_dim, 1)
        self.linear.weight = PyroSample(dist.Normal(0., 1.).expand([1, input_dim]).to_event(2))
        self.linear.bias = PyroSample(dist.Normal(0., 10.).expand([1]).to_event(1))

    def forward(self, x, y=None):
        logits = self.linear(x).squeeze(-1)
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Bernoulli(logits=logits), obs=y)
        return logits

def train_bayesian_lr(model, X_train, y_train, num_steps=5000):
    guide = pyro.infer.autoguide.AutoDiagonalNormal(model)
    optim = pyro.optim.Adam({"lr": 0.01})
    svi = pyro.infer.SVI(model, guide, optim, loss=pyro.infer.Trace_ELBO())

    pyro.clear_param_store()
    for step in range(num_steps):
        loss = svi.step(X_train, y_train) / X_train.shape[0]
        if step % 500 == 0:
            print(f"[Step {step}] Loss: {loss:.4f}")
    return guide

def evaluate(model, guide, X_test, y_test):
    predictive = pyro.infer.Predictive(model, guide=guide, num_samples=1000)
    samples = predictive(X_test)
    probs = samples["obs"].float().mean(dim=0)

    y_pred = (probs > 0.5).float()
    accuracy = (y_pred == y_test).float().mean().item()
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy

def main():
    # Path to the downloaded Facebook data
    data_dir = "./data/facebook"

    # Initialize data loader
    loader = FacebookEgoNetwork(data_dir)

    # Load a specific ego network (using 0 as an example)
    ego_id = 3437 #0
    try:
        G = loader.load_ego_network(ego_id)
        print(
            f"Loaded ego network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges"
        )

        # Generate training and testing data
        X_train, y_train, X_test, y_test = loader.generate_training_data(test_size=0.2)
        print(
            f"Generated {len(X_train)} training examples and {len(X_test)} testing examples"
        )

        # Initialize Bayesian logistic regression model (sanity check without training)
        model = BayesianLogisticRegression(input_dim=X_train.shape[1])
        posterior = train_bayesian_lr(model, X_train, y_train)
        accuracy = evaluate(model=model, guide=posterior, X_test=X_test, y_test=y_test)
        print(f"Accuracy for ego network node {ego_id}: {accuracy:.4f}")

        # # approximate posterior
        # guide = AutoDiagonalNormal(model)

        # optimizer = pyro.optim.Adam({"lr": 0.01})
        # elbo = Trace_ELBO()
        # svi = SVI(model, guide, optimizer, loss=elbo)

        # # perform one step of inference
        # try:
        #     for i in range(100):
        #         loss = svi.step(X_train, y_train)
        #         if i % 10 == 0:
        #             print(f"[Sanity Check] Initial ELBO loss after {i } steps: {loss:.4f}")
            
        # except Exception as e:
        #     print(f"[Sanity Check] Failed with error: {e}")


        # uncomment to train and evaluate
        # guide = train_bayesian_lr(model, X_train, y_train, num_steps=1000)
        # evaluate(model, guide, X_test, y_test)

    except Exception as e:
        print(f"Error: {e}")
        print(
            "Please check that the Facebook ego network data files are in the correct directory."
        )


if __name__ == "__main__":
    main()
