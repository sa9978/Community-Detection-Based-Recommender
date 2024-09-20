import torch
from torch_geometric.datasets import Planetoid
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
import pandas as pd
import json

import torch
import numpy as np
import scipy.sparse as sp
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_scipy_sparse_matrix

def load_planetoid_data(dataset):
    """
    Loads Cora, Citeseer, or Pubmed datasets using PyTorch Geometric's Planetoid class.
    Returns the adjacency matrix (A), feature matrix (X), and community labels (Z).
    """
    # Load the dataset
    # dataset = Planetoid(root=f'./data/{dataset_name}', name=dataset_name)
    data = dataset[0]  # Get the graph object

    # Convert adjacency matrix to scipy sparse format
    A = to_scipy_sparse_matrix(data.edge_index).tocsr()

    # Extract feature matrix (node features) as a dense numpy array
    X = data.x.numpy()  # PyTorch tensor to NumPy

    # Extract community labels (node labels)
    Z = data.y.numpy()  # PyTorch tensor to NumPy

    # Remove self-loops by setting the diagonal to zero
    A = A.tolil() # LIL (List of Lists
    A.setdiag(0) # Sets them all diagonal elements to zero -> remove self loops
    A = A.tocsr() # Back to CSR format
    
    # Convert labels (Z) to one-hot encoding if necessary (optional step)
    Z = np.eye(max(Z) + 1)[Z]  # One-hot encode labels if needed

    # Convert the graph components into a dictionary
    graph = {
        'A': A,  # Adjacency matrix in sparse format
        'X': X,  # Feature matrix
        'Z': Z   # Community labels
    }
    print("Adjacency matrix shape:", A.shape)
    print("Feature matrix shape:", X.shape)
    print("Community labels shape:", Z.shape)
    return graph

if __name__ == "__main__":
    graph = load_planetoid_data('Cora')

    A = graph['A']
    X = graph['X']
    Z = graph['Z']

    print("Adjacency matrix shape:", A.shape)
    print("Feature matrix shape:", X.shape)
    print("Community labels shape:", Z.shape)

# # Load LastFM Dataset (from file)
# def load_lastfm_data(edges_path, target_path, features_path):
#     """
#     Loads LastFM dataset from given file paths.
#     Returns adjacency matrix, node features, and target labels.
#     """
#     # Load edges (node_1, node_2 pairs)
#     edges = pd.read_csv(edges_path)
#     num_nodes = max(max(edges['node_1']), max(edges['node_2'])) + 1
#     adj = sp.coo_matrix((np.ones(len(edges)), (edges['node_1'], edges['node_2'])), shape=(num_nodes, num_nodes))

#     # Convert to dense matrix
#     adj = adj.toarray()

#     # Load features (from JSON)
#     with open(features_path, 'r') as f:
#         features = json.load(f)

#     # Convert features to numpy array
#     features = np.array([features[str(i)] for i in range(len(features))])

#     # Load target labels
#     targets = pd.read_csv(target_path)['target'].values

#     return adj, features, targets

# # Preprocess Adjacency Matrix (Normalization and Adding Self-loops)
# def preprocess_adjacency(adj):
#     """
#     Normalizes the adjacency matrix and adds self-loops.
#     Returns the normalized adjacency matrix.
#     """
#     adj = adj + np.eye(adj.shape[0])  # Add self-loops
#     rowsum = np.array(adj.sum(1))
#     d_inv_sqrt = np.power(rowsum, -0.5).flatten()
#     d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
#     d_mat_inv_sqrt = np.diag(d_inv_sqrt)
#     return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

# # Preprocess Node Features
# def normalize_features(features):
#     """
#     Normalizes node features using standard scaling (mean=0, variance=1).
#     Returns the normalized feature matrix.
#     """
#     scaler = StandardScaler()
#     return scaler.fit_transform(features)

# # Common Data Loading and Preprocessing Function
# def load_data(dataset_name):
#     """
#     Loads and preprocesses the dataset based on its name.
#     Supports 'cora', 'citeseer', 'pubmed' (Planetoid) and 'lastfm'.
#     Returns preprocessed adjacency matrix and node features.
#     """
#     if dataset_name in ['cora', 'citeseer', 'pubmed']:
#         adj, features, labels = load_planetoid_data(dataset_name)
#     elif dataset_name == 'lastfm':
#         adj, features, labels = load_lastfm_data(
#             './data/lastfm_asia/lastfm_asia_edges.csv',
#             './data/lastfm_asia/lastfm_asia_target.csv',
#             './data/lastfm_asia/lastfm_asia_features.json'
#         )
#     else:
#         raise ValueError("Unsupported dataset!")

#     # Preprocess adjacency matrix and node features
#     adj_normalized = preprocess_adjacency(adj)
#     features_normalized = normalize_features(features)

#     return adj_normalized, features_normalized, labels

# Example Usage:
# adj, features, labels = load_data('cora')
# adj_lastfm, features_lastfm, labels_lastfm = load_data('lastfm')