import numpy as np
import torch
from scipy.optimize import minimize_scalar
import os
def gaussian_similarity(x1, x2, sigma=1.0):
    return torch.exp(-torch.norm(x1 - x2) ** 2 / (2 * sigma ** 2))

def find_optimal_sigma(node_features):
    def cost_function(sigma):
        pairwise_dists = np.linalg.norm(node_features[:, np.newaxis] - node_features[np.newaxis, :], axis=2)
        
        similarities = np.exp(-pairwise_dists**2 / (2 * sigma**2))
        upper_triangle_indices = np.triu_indices(len(node_features), k=1)
        
        relevant_similarities = similarities[upper_triangle_indices]
        
        min_sim = np.min(relevant_similarities)
        max_sim = np.max(relevant_similarities)
        normalized_similarities = (relevant_similarities - min_sim) / (max_sim - min_sim)

        hist, _ = np.histogram(normalized_similarities, bins=20, range=(0, 1))
        return np.var(hist)

    result = minimize_scalar(cost_function, bounds=(0.1, 10), method='bounded')
    return result.x

def prepare_pairs(features, community_data, node_idx):
    """
    Prepare positive and negative pairs for a specific node based on the community structure.

    Args:
        features (torch.Tensor): Node feature matrix.
        community_data (numpy.ndarray): Array of community labels for each node.
        node_idx (int): Index of the node for which to generate pairs.

    Returns:
        positive_pairs (list of tuple): List of positive pairs as (node_idx, positive_node_idx).
        negative_pairs (list of tuple): List of negative pairs as (node_idx, negative_node_idx).
    """
    positive_pairs = []
    negative_pairs = []

    current_node_feature = features[node_idx]
    current_community = community_data[node_idx]

    same_community_indices = np.where(community_data == current_community)[0]
    if len(same_community_indices) > 1:  # Ensure there are other nodes in the same community
        positive_indices = np.random.choice(
            same_community_indices[same_community_indices != node_idx], 
            size=min(len(same_community_indices) - 1, 10),  # Limit to a fixed number of samples to speed up training
            replace=False
        )
        positive_pairs.extend([(node_idx, pos_idx) for pos_idx in positive_indices])

    other_community_indices = np.where(community_data != current_community)[0]
    if len(other_community_indices) > 0:
        other_community_features = features[other_community_indices]

        distances = torch.cdist(current_node_feature.unsqueeze(0), other_community_features).squeeze(0)
        furthest_neg_idx = other_community_indices[torch.argmax(distances)]  # Choose the furthest node from other communities
        negative_pairs.append((node_idx, furthest_neg_idx))

    return positive_pairs, negative_pairs

def load_concatenated_features(dataset_name, model_name):
    """
    Load concatenated features from a single file in the 'ProcessedFeatures' directory.

    Args:
        dataset_name (str): The name of the dataset (e.g., 'citeseer').
        model_name (str): The name of the model (e.g., 'secomm').

    Returns:
        torch.FloatTensor: The concatenated feature tensor.
    """
    dataset_name = dataset_name.lower()
    model_name = model_name.lower()

    file_path = os.path.join('ProcessedFeatures', f'{dataset_name}_{model_name}_concatenated_features.npy')

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Concatenated feature file '{file_path}' not found in the 'ProcessedFeatures' directory.")

    concatenated_features = np.load(file_path)
    print(f"Loaded concatenated features from file: {file_path}")

    concatenated_features_tensor = torch.tensor(concatenated_features, dtype=torch.float32)

    print(f"Loaded concatenated features with shape: {concatenated_features_tensor.shape}")
    return concatenated_features_tensor