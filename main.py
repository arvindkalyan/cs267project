#!/usr/bin/env python3
"""
CS267 Project: Using PPLs to Model Social Ego-Networks
Main file with integrated benchmarking and result visualization.
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
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from pyro.nn import PyroModule, PyroSample
from pyro.infer import SVI
from node2vec import Node2Vec
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from visualization import (
    visualize_ego_network,
    comprehensive_link_prediction_analysis,
)

# Set random seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
pyro.set_rng_seed(RANDOM_SEED)


class FacebookEgoNetwork:
    """Loader for Facebook ego network data"""

    def __init__(self, data_dir):
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
        node2vec = Node2Vec(
            self.graph, dimensions=dimensions, walk_length=20, num_walks=100, workers=2
        )
        model = node2vec.fit(window=10, min_count=1, batch_words=4)
        return {
            int(node): torch.tensor(model.wv[node], dtype=torch.float)
            for node in model.wv.index_to_key
        }

    def load_ego_network(self, ego_id):
        print(f"Loading ego network for node {ego_id}")
        edges_file = os.path.join(self.data_dir, f"{ego_id}.edges")
        if not os.path.exists(edges_file):
            raise FileNotFoundError(f"Edge file not found: {edges_file}")
        G = nx.Graph()
        G.add_node(ego_id)
        with open(edges_file, "r") as f:
            for line in f:
                node1, node2 = map(int, line.strip().split())
                G.add_edge(node1, node2)
                G.add_edge(ego_id, node1)
                G.add_edge(ego_id, node2)
        all_nodes = set(G.nodes())
        with open(edges_file, "r") as f:
            for line in f:
                node1, node2 = map(int, line.strip().split())
                if node1 not in all_nodes:
                    G.add_node(node1)
                if node2 not in all_nodes:
                    G.add_node(node2)
        self.graph = G
        return G

    def extract_link_features(self, node1, node2):
        if self.graph is None:
            raise ValueError("No graph loaded.")
        G = self.graph
        cn = len(list(nx.common_neighbors(G, node1, node2)))
        n1, n2 = set(G.neighbors(node1)), set(G.neighbors(node2))
        jaccard = 0
        if len(n1 | n2) > 0:
            jaccard = len(n1 & n2) / len(n1 | n2)
        aa_index = sum(
            1 / np.log(G.degree(u))
            for u in nx.common_neighbors(G, node1, node2)
            if G.degree(u) > 1
        )
        pa = G.degree(node1) * G.degree(node2)
        return torch.tensor([cn, jaccard, aa_index, pa], dtype=torch.float)

    def extract_link_features_with_embeddings(self, node1, node2, embeddings):
        base_features = self.extract_link_features(node1, node2)
        emb_shape = embeddings[next(iter(embeddings))].shape
        emb1 = embeddings.get(node1, torch.zeros(emb_shape))
        emb2 = embeddings.get(node2, torch.zeros(emb_shape))
        emb_features = torch.cat([emb1, emb2, torch.abs(emb1 - emb2)])
        return torch.cat([base_features, emb_features])

    def generate_training_data(
        self, test_size=0.2, use_embeddings=True, embedding_dim=16
    ):
        if self.graph is None:
            raise ValueError("No graph loaded.")
        G = self.graph
        embeddings = (
            self.compute_node2vec_embeddings(dimensions=embedding_dim)
            if use_embeddings
            else None
        )
        positive_examples = list(G.edges())
        nodes = list(G.nodes())
        negative_examples = []
        while len(negative_examples) < len(positive_examples):
            i, j = random.sample(nodes, 2)
            if i != j and not G.has_edge(i, j):
                negative_examples.append((i, j))

        pos_train, pos_test = train_test_split(
            positive_examples, test_size=test_size, random_state=RANDOM_SEED
        )
        neg_train, neg_test = train_test_split(
            negative_examples, test_size=test_size, random_state=RANDOM_SEED
        )

        X_train, y_train, X_test, y_test = [], [], [], []
        test_edges = pos_test + neg_test

        extractor = lambda u, v: (
            self.extract_link_features_with_embeddings(u, v, embeddings)
            if use_embeddings
            else self.extract_link_features(u, v)
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
            test_edges,
            embeddings,
        )


class BayesianLogisticRegression(PyroModule):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = PyroModule[nn.Linear](input_dim, 1)
        self.linear.weight = PyroSample(
            dist.Normal(0.0, 1.0).expand([1, input_dim]).to_event(2)
        )
        self.linear.bias = PyroSample(dist.Normal(0.0, 10.0).expand([1]).to_event(1))

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
    return samples["obs"].float().mean(dim=0)


def get_heuristic_scores(G, test_edges):
    print("\n--- Evaluating Heuristic Models ---")
    scores = {
        "Adamic-Adar Index": [s for _, _, s in nx.adamic_adar_index(G, test_edges)],
        "Jaccard Coefficient": [s for _, _, s in nx.jaccard_coefficient(G, test_edges)],
        "Preferential Attachment": [
            s for _, _, s in nx.preferential_attachment(G, test_edges)
        ],
    }
    return scores


def get_classic_ml_scores(X_train, y_train, X_test):
    print("\n--- Evaluating Classic ML Models ---")
    X_train_np = X_train.cpu().numpy()
    y_train_np = y_train.cpu().numpy()
    X_test_np = X_test.cpu().numpy()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_np)
    X_test_scaled = scaler.transform(X_test_np)

    models = {
        "SKlearn Logistic Regression": LogisticRegression(
            max_iter=5000, random_state=RANDOM_SEED
        ),
        "SKlearn Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_SEED
        ),
    }

    scores = {}
    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train_np)
        scores[name] = model.predict_proba(X_test_scaled)[:, 1]
        trained_models[name] = model, scaler  # Store model and its scaler

    return scores, trained_models


def evaluate(y_true, y_probs, model_name="Model"):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    y_probs = np.asarray(
        y_probs.cpu() if isinstance(y_probs, torch.Tensor) else y_probs
    )
    auc = roc_auc_score(y_true, y_probs)
    y_pred = (y_probs > 0.5).astype(int)
    accuracy = accuracy_score(y_true, y_pred)
    print(f"{model_name:<28} | AUC: {auc:.4f} | Accuracy: {accuracy:.4f}")
    return auc, accuracy


def plot_benchmarks(results):
    print("\n--- Generating Benchmark Plots with Seaborn ---")
    plot_data = []
    for name, (auc, acc) in results.items():
        if "Bayesian" in name:
            model_type = "Bayesian"
        elif "SKlearn" in name:
            model_type = "Classic ML"
        else:
            model_type = "Heuristic"
        plot_data.append(
            {"Model": name, "AUC": auc, "Accuracy": acc, "Type": model_type}
        )
    df = pd.DataFrame(plot_data)

    def _create_and_save_seaborn_plot(data_df, metric, filename):
        sorted_df = data_df.sort_values(by=metric, ascending=False)
        plt.figure(figsize=(12, 8))
        ax = sns.barplot(
            x=metric,
            y="Model",
            data=sorted_df,
            hue="Type",
            dodge=False,
            palette="viridis",
        )
        ax.set_title(
            f"Link Prediction Benchmark: {metric}",
            fontsize=18,
            fontweight="bold",
            pad=20,
        )
        ax.set_xlabel(f"{metric} Score", fontsize=14)
        ax.set_ylabel("Model", fontsize=14)
        max_score = data_df[metric].max()
        ax.set_xlim(left=0.75, right=max_score * 1.10)
        for p in ax.patches:
            width = p.get_width()
            if width > 0:
                ax.text(
                    width - 0.005,
                    p.get_y() + p.get_height() / 2,
                    f"{width:.4f}",
                    va="center",
                    ha="right",
                    color="white",
                    fontweight="bold",
                    fontsize=12,
                )
        plt.legend(title="Model Type", loc="lower right", fontsize=12)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {filename}")
        plt.show()

    _create_and_save_seaborn_plot(
        df, "AUC", os.path.join("./visualizations", "ROC_AUC.png")
    )
    _create_and_save_seaborn_plot(
        df, "Accuracy", os.path.join("./visualizations", "accuracy_benchmark.png")
    )


def run_edge_recovery_bayesian(
    loader, model, guide, ego_id, fraction=0.2, use_embeddings=True, embedding_dim=16
):
    """Calculates the edge recovery rate for the Bayesian model."""
    print(f"\nRunning Edge Recovery for [Bayesian Logistic Regression]...")
    original_graph = loader.load_ego_network(ego_id)

    # Sever edges
    edges = list(original_graph.edges())
    num_remove = int(len(edges) * fraction)
    removed_edges = random.sample(edges, num_remove)
    G_severed = original_graph.copy()
    G_severed.remove_edges_from(removed_edges)

    # Re-calculate features from the severed graph
    temp_loader = copy.deepcopy(loader)
    temp_loader.graph = G_severed

    embeddings = (
        temp_loader.compute_node2vec_embeddings(dimensions=embedding_dim)
        if use_embeddings
        else None
    )
    extractor = lambda u, v: (
        temp_loader.extract_link_features_with_embeddings(u, v, embeddings)
        if use_embeddings
        else temp_loader.extract_link_features(u, v)
    )

    X_removed = torch.stack([extractor(u, v) for u, v in removed_edges])

    # Get predictions
    predictive = pyro.infer.Predictive(model, guide=guide, num_samples=500)
    samples = predictive(X_removed)
    probs = samples["obs"].float().mean(dim=0)
    y_pred = (probs > 0.5).float()

    # Calculate and print recovery rate
    recovered_count = int(y_pred.sum().item())
    total_removed = len(y_pred)
    recovery_rate = recovered_count / total_removed if total_removed > 0 else 0
    print(
        f"{'[Recovery] Bayesian LR':<28} | Recovery Rate: {recovery_rate:.4f} ({recovered_count}/{total_removed})"
    )


def run_edge_recovery_heuristic(loader, ego_id, fraction=0.2):
    """Calculates the edge recovery rate for heuristic models."""
    print(f"\nRunning Edge Recovery for [Heuristic Models]...")
    original_graph = loader.load_ego_network(ego_id)

    # Sever edges
    edges = list(original_graph.edges())
    num_remove = int(len(edges) * fraction)
    removed_edges = random.sample(edges, num_remove)
    G_severed = original_graph.copy()
    G_severed.remove_edges_from(removed_edges)

    # Get scores for each heuristic on the severed graph
    heuristic_scores = {
        "Adamic-Adar Index": [
            s for _, _, s in nx.adamic_adar_index(G_severed, removed_edges)
        ],
        "Jaccard Coefficient": [
            s for _, _, s in nx.jaccard_coefficient(G_severed, removed_edges)
        ],
        "Preferential Attachment": [
            s for _, _, s in nx.preferential_attachment(G_severed, removed_edges)
        ],
    }

    # Evaluate each heuristic
    for name, scores in heuristic_scores.items():
        scores_np = np.asarray(scores)
        if len(scores_np) == 0:
            continue

        # Use the median score as a dynamic threshold for binary prediction
        threshold = np.median(scores_np)
        y_pred = (scores_np > threshold).astype(int)

        recovered_count = int(y_pred.sum())
        total_removed = len(y_pred)
        recovery_rate = recovered_count / total_removed if total_removed > 0 else 0
        print(
            f"{f'[Recovery] {name}':<28} | Recovery Rate: {recovery_rate:.4f} ({recovered_count}/{total_removed})"
        )


def run_edge_recovery_classic(
    loader, trained_models, ego_id, fraction=0.2, use_embeddings=True, embedding_dim=16
):
    """Calculates the edge recovery rate for classic ML models."""
    print(f"\nRunning Edge Recovery for [Classic ML Models]...")
    original_graph = loader.load_ego_network(ego_id)

    # Sever edges
    edges = list(original_graph.edges())
    num_remove = int(len(edges) * fraction)
    removed_edges = random.sample(edges, num_remove)
    G_severed = original_graph.copy()
    G_severed.remove_edges_from(removed_edges)

    # Re-calculate features from the severed graph
    temp_loader = copy.deepcopy(loader)
    temp_loader.graph = G_severed

    embeddings = (
        temp_loader.compute_node2vec_embeddings(dimensions=embedding_dim)
        if use_embeddings
        else None
    )
    extractor = lambda u, v: (
        temp_loader.extract_link_features_with_embeddings(u, v, embeddings)
        if use_embeddings
        else temp_loader.extract_link_features(u, v)
    )

    X_removed = torch.stack([extractor(u, v) for u, v in removed_edges])
    X_removed_np = X_removed.cpu().numpy()

    # Evaluate each trained model
    for name, (model, scaler) in trained_models.items():
        X_removed_scaled = scaler.transform(X_removed_np)
        scores = model.predict_proba(X_removed_scaled)[:, 1]
        y_pred = (scores > 0.5).astype(int)

        recovered_count = int(y_pred.sum())
        total_removed = len(y_pred)
        recovery_rate = recovered_count / total_removed if total_removed > 0 else 0
        print(
            f"{f'[Recovery] {name}':<28} | Recovery Rate: {recovery_rate:.4f} ({recovered_count}/{total_removed})"
        )


def main():
    # --- 1. Setup ---
    data_dir = "./data/facebook"
    ego_id = 3980
    loader = FacebookEgoNetwork(data_dir)
    # G is the original, full graph used for standard and heuristic evals
    G = loader.load_ego_network(ego_id)
    print(
        f"\nGraph loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges."
    )
    os.makedirs("./visualizations", exist_ok=True)
    vis_output = f"./visualizations/ego_network_{ego_id}.png"
    visualize_ego_network(G, ego_id, output_file=vis_output)

    benchmark_results = {}

    # --- 2. Data Generation & Standard Benchmark ---
    X_train, y_train, X_test, y_test, test_edges, _ = loader.generate_training_data(
        test_size=0.2, use_embeddings=True, embedding_dim=16
    )
    print(
        f"Generated {len(X_train)} training examples and {len(X_test)} testing examples."
    )

    print("\n" + "=" * 50)
    print("      PART 1: Standard Link Prediction Benchmark")
    print("=" * 50)

    # Bayesian Model
    model_name_b = "Bayesian Logistic Regression"
    model = BayesianLogisticRegression(input_dim=X_train.shape[1])
    posterior = train_bayesian_lr(model, X_train, y_train)
    bayesian_scores = get_bayesian_model_scores(model, posterior, X_test)
    benchmark_results[model_name_b] = evaluate(y_test, bayesian_scores, model_name_b)

    comprehensive_link_prediction_analysis(loader, model, posterior, ego_id)

    # Heuristic Models
    heuristic_scores = get_heuristic_scores(G, test_edges)
    for name, scores in heuristic_scores.items():
        benchmark_results[name] = evaluate(y_test, scores, name)

    # Classic ML Models
    classic_ml_scores, trained_classic_models = get_classic_ml_scores(
        X_train, y_train, X_test
    )
    for name, scores in classic_ml_scores.items():
        benchmark_results[name] = evaluate(y_test, scores, name)

    # Plotting Standard Benchmark Results
    plot_benchmarks(benchmark_results)

    # --- 3. Edge Recovery Benchmark ---
    print("\n" + "=" * 50)
    print("      PART 2: Edge Recovery Benchmark")
    print("=" * 50)

    # Run recovery evaluation for all model types
    # Note: These functions will print their own results in the requested format
    run_edge_recovery_bayesian(loader, model, posterior, ego_id, fraction=0.2)
    run_edge_recovery_heuristic(loader, ego_id, fraction=0.2)
    run_edge_recovery_classic(loader, trained_classic_models, ego_id, fraction=0.2)

    print("\nAll benchmarks complete.")


if __name__ == "__main__":
    main()
