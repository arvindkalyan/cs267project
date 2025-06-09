#!/usr/bin/env python3
"""
CS267 Project: Using PPLs to Model Social Ego-Networks
Main file for loading data, training models, and evaluating link prediction
"""

import os
import copy
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
from node2vec import Node2Vec
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier



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

    def compute_node2vec_embeddings(self, dimensions=16):
        if self.graph is None:
            raise ValueError("Graph not loaded. Call load_ego_network first.")
        
        print("Computing Node2Vec embeddings...")
        node2vec = Node2Vec(self.graph, dimensions=dimensions, walk_length=20, num_walks=100, workers=2)
        model = node2vec.fit(window=10, min_count=1, batch_words=4)

        # Store embeddings as {node_id: torch.tensor}
        embeddings = {int(node): torch.tensor(model.wv[node], dtype=torch.float) for node in model.wv.index_to_key}
        return embeddings

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

    def extract_link_features_with_embeddings(self, node1, node2, embeddings):
        base_features = self.extract_link_features(node1, node2)

        emb1 = embeddings.get(node1, torch.zeros(embeddings[next(iter(embeddings))].shape))
        emb2 = embeddings.get(node2, torch.zeros(emb1.shape))

        # Concatenate embedding features: [emb1, emb2, |emb1 - emb2|]
        emb_features = torch.cat([emb1, emb2, torch.abs(emb1 - emb2)])
        return torch.cat([base_features, emb_features])


    # def generate_training_data(self, test_size=0.2):
    def generate_training_data(self, test_size=0.2, use_embeddings=True, embedding_dim=16):
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
        # Use embeddings
        embeddings = self.compute_node2vec_embeddings(dimensions=embedding_dim) if use_embeddings else None

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
        test_edges = pos_test + neg_test

        # Create feature vectors
        # X_train = []
        # y_train = []

        # for edge in pos_train:
        #     X_train.append(self.extract_link_features(edge[0], edge[1]))
        #     y_train.append(1)

        # for edge in neg_train:
        #     X_train.append(self.extract_link_features(edge[0], edge[1]))
        #     y_train.append(0)

        # X_test = []
        # y_test = []

        # for edge in pos_test:
        #     X_test.append(self.extract_link_features(edge[0], edge[1]))
        #     y_test.append(1)

        # for edge in neg_test:
        #     X_test.append(self.extract_link_features(edge[0], edge[1]))
        #     y_test.append(0)

        # # Convert to tensors
        # X_train = torch.stack(X_train)
        # y_train = torch.tensor(y_train, dtype=torch.float)
        # X_test = torch.stack(X_test)
        # y_test = torch.tensor(y_test, dtype=torch.float)

        # return X_train, y_train, X_test, y_test
        # Create feature vectors
        X_train, y_train, X_test, y_test = [], [], [], []

        extractor = (
            lambda u, v: self.extract_link_features_with_embeddings(u, v, embeddings)
            if use_embeddings else
            self.extract_link_features
        )

        for edge in pos_train:
            X_train.append(extractor(edge[0], edge[1]))
            y_train.append(1)
        for edge in neg_train:
            X_train.append(extractor(edge[0], edge[1]))
            y_train.append(0)

        for edge in pos_test:
            X_test.append(extractor(edge[0], edge[1]))
            y_test.append(1)
        for edge in neg_test:
            X_test.append(extractor(edge[0], edge[1]))
            y_test.append(0)

        return (
            torch.stack(X_train),
            torch.tensor(y_train, dtype=torch.float),
            torch.stack(X_test),
            torch.tensor(y_test, dtype=torch.float),
            test_edges
        )


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

def get_bayesian_model_scores(model, guide, X_test):
    print("\n--- Evaluating Bayesian Model ---")
    predictive = pyro.infer.Predictive(model, guide=guide, num_samples=1000)
    samples = predictive(X_test)
    return samples["obs"].float().mean(dim=0) # probs in 'evaluate'

def get_heuristic_scores(G, test_edges):
    print("\n--- Evaluating Heuristic Models ---")
    scores = {
        "Adamic-Adar Index": [s for _, _, s in nx.adamic_adar_index(G, test_edges)],
        "Jaccard Coefficient": [s for _, _, s in nx.jaccard_coefficient(G, test_edges)],
        "Preferential Attachment": [s for _, _, s in nx.preferential_attachment(G, test_edges)],
    }
    return scores

def get_classic_ml_scores(X_train, y_train, X_test):
    print("\n--- Evaluating Classic ML Models ---")
    X_train_np = X_train.cpu().numpy()
    y_train_np = y_train.cpu().numpy()
    X_test_np = X_test.cpu().numpy()

    models = {
        "SKlearn Logistic Regression": LogisticRegression(max_iter=5000, random_state=RANDOM_SEED),
        "SKlearn Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    }

    scores = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_np, y_train_np)
        scores[name] = model.predict_proba(X_test_np)[:, 1]
      
    return scores

def evaluate(y_true, y_probs, model_name="Model"):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    # if isinstance(y_probs, torch.Tensor):
    #     y_probs = y_probs.cpu().numpy()

    y_probs = np.asarray(y_probs.cpu() if isinstance(y_probs, torch.Tensor) else y_probs)
    
    auc = roc_auc_score(y_true, y_probs)
    y_pred = (y_probs > 0.5).astype(int)
    accuracy = accuracy_score(y_true, y_pred)
    
    print(f"{model_name:<28} | AUC: {auc:.4f} | Accuracy: {accuracy:.4f}")
    
    return auc, accuracy

# def evaluate(model, guide, X_test, y_test):
#     if isinstance(model, str):
#         pass
#     if isinstance(model, BayesianLogisticRegression):
#         pass
#     predictive = pyro.infer.Predictive(model, guide=guide, num_samples=1000)
#     samples = predictive(X_test)
#     probs = samples["obs"].float().mean(dim=0)

#     auc = roc_auc_score(y_test.cpu().numpy(), probs.cpu().numpy())
#     print(f"AUC Score: {auc:.4f}")

#     y_pred = (probs > 0.5).float()
#     accuracy = (y_pred == y_test).float().mean().item()
#     print(f"Accuracy: {accuracy:.4f}")
#     return accuracy

def benchmark(model, guide, X_train, y_train, X_test, y_test):
    heuristics = ['Adamic-Agar', 'Common Neighbors']
    for heur in heuristics:
        evaluate(heur, None, X_test, y_test)

    lr_model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    lr_model.fit(X_train, y_train)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    rf_model.fit(X_train, y_train)
    ml_models = [lr_model, rf_model]
    for model in ml_models:
        evaluate(model, None, X_test, y_test)

    bnn = None
    return

def edge_recovery_evaluation(loader, model, guide, ego_id, fraction=0.5, use_embeddings=True, embedding_dim=16):
    """
    Sever a fraction of edges in the given ego network, predict edges, and evaluate recovery.

    Args:
        loader: FacebookEgoNetwork instance
        model: Trained BayesianLogisticRegression model
        guide: Posterior guide from training
        ego_id: Ego node ID

    Returns:
        recovery_rate: Fraction of removed edges correctly predicted as present
    """

    if fraction <= 0 or fraction >= 1:
        raise ValueError("Fraction must be between 0 and 1.")
    
    # Load the ego network
    G = loader.load_ego_network(ego_id)
    edges = list(G.edges())
    num_remove = int(len(edges) * fraction)
    removed_edges = random.sample(edges, num_remove)

    # Create a copy of the graph and remove edges
    G_severed = copy.deepcopy(G)
    G_severed.remove_edges_from(removed_edges)
    loader.graph = G_severed  

    embeddings = loader.compute_node2vec_embeddings(dimensions=embedding_dim) if use_embeddings else None
    # Choose extractor
    extractor = (
        lambda u, v: loader.extract_link_features_with_embeddings(u, v, embeddings)
        if use_embeddings else
        loader.extract_link_features
    )

    # For each removed edge, extract features and predict
    X_removed = []
    for u, v in removed_edges:
        # Only predict if both nodes still exist in the graph
        if G_severed.has_node(u) and G_severed.has_node(v):
            X_removed.append(extractor(u, v))

    X_removed = torch.stack(X_removed)
    predictive = pyro.infer.Predictive(model, guide=guide, num_samples=500)
    samples = predictive(X_removed)
    probs = samples["obs"].float().mean(dim=0)
    y_pred = (probs > 0.5).float()

    # all are true edges so label is 1
    recovery_rate = y_pred.sum().item() / len(y_pred)
    print(f"Edge recovery rate for ego {ego_id}: {recovery_rate:.4f} ({int(y_pred.sum().item())}/{len(y_pred)})")

    # restore original graph
    loader.graph = G
    return recovery_rate


def main():
    # Path to the downloaded Facebook data
    data_dir = "./data/facebook"

    # Initialize data loader
    loader = FacebookEgoNetwork(data_dir)

    # Load a specific ego network (using 0 as an example)
    ego_id = 3980 #0
    try:
        G = loader.load_ego_network(ego_id)
        print(
            f"Loaded ego network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges"
        )

        # Generate training and testing data
        # X_train, y_train, X_test, y_test = loader.generate_training_data(test_size=0.2)
        X_train, y_train, X_test, y_test, test_edges = loader.generate_training_data(
            test_size=0.2, use_embeddings=True, embedding_dim=16
        )
        print(
            f"Generated {len(X_train)} training examples and {len(X_test)} testing examples"
        )


        # Initialize Bayesian logistic regression model (sanity check without training)
        model = BayesianLogisticRegression(input_dim=X_train.shape[1])
        posterior = train_bayesian_lr(model, X_train, y_train)
        bayesian_scores = get_bayesian_model_scores(model, posterior, X_test)
        evaluate(y_test, bayesian_scores, "Bayesian Logistic Regression")

        # --- Edge Recovery Evaluation ---
        print("\n--- Edge Recovery Evaluation ---")
        edge_recovery_evaluation(loader, model, posterior, ego_id, fraction=0.2)

        heuristic_scores = get_heuristic_scores(G, test_edges)
        for name, scores in heuristic_scores.items():
            evaluate(y_test, scores, name)
            
        # --- Benchmark Classic ML Models ---
        classic_ml_scores = get_classic_ml_scores(X_train, y_train, X_test)
        for name, scores in classic_ml_scores.items():
            evaluate(y_test, scores, name)

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
