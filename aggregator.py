import torch
import torch.nn as nn
import numpy as np
import os
from torch_geometric.datasets import Planetoid
from torch.utils.data import DataLoader, TensorDataset
from data_loader import load_dataset

class CNNAggregator(nn.Module):
    def __init__(self, feature_size, kernel_size=3):
        super(CNNAggregator, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=feature_size, kernel_size=kernel_size, stride=1, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        x = x.unsqueeze(1) 
        x = self.conv1d(x)
        x = torch.relu(x)
        x = self.pool(x)
        return x.squeeze(-1) 

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.model(x)

def standardize_file_names(directory):
    """Convert all .npy file names in the specified directory to lowercase."""
    for filename in os.listdir(directory):
        if filename.endswith('.npy'):
            lowercase_filename = filename.lower()
            if filename != lowercase_filename:
                os.rename(os.path.join(directory, filename), os.path.join(directory, lowercase_filename))
                print(f"Renamed '{filename}' to '{lowercase_filename}'")

def load_community_data(dataset_name, model_name):
    dataset_name = dataset_name.lower()
    model_name = model_name.lower()

    filename = f'{dataset_name}-{model_name}.npy'
    filepath = os.path.join('Communities', filename)

    standardize_file_names('Communities')

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Community file '{filename}' not found in the 'Communities' folder.")

    community_data = np.load(filepath)
    return community_data

def save_features(features, filename):
    """Save the entire set of features to a single file."""
    output_filepath = os.path.join('ProcessedFeatures', filename)

    os.makedirs('ProcessedFeatures', exist_ok=True)


    np.save(output_filepath, features.detach().cpu().numpy())
    print(f"Features saved to {output_filepath}")

def cosine_similarity(x1, x2):
    """Calculate cosine similarity between two tensors."""
    return nn.functional.cosine_similarity(x1, x2, dim=1)

def aggregate_and_concatenate_features(dataset, dataset_name, model_name, batch_size=64):
    """Aggregate node features, compute concatenated features, and calculate similarity scores."""
    community_data = load_community_data(dataset_name, model_name)
    node_features = dataset[0].x

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_size = node_features.size(1)
    cnn_aggregator = CNNAggregator(feature_size).to(device)

    input_size = 2 * feature_size 
    hidden_size = 256 
    output_size = 128 
    mlp = MLP(input_size, hidden_size, output_size).to(device)

    node_features = node_features.to(device)

    all_concatenated_features = []
    all_similarity_scores = []

    unique_communities = np.unique(community_data)

    num_workers = 4 
    with torch.no_grad():
        for community in unique_communities:
            community_indices = torch.tensor(np.where(community_data == community)[0], dtype=torch.long, device=device)
            community_features = node_features[community_indices]

            dataset_loader = DataLoader(TensorDataset(community_features, community_indices), 
                                        batch_size=batch_size, shuffle=False, num_workers=num_workers)

            for community_batch, indices_batch in dataset_loader:
                with torch.cuda.amp.autocast(): 
                    aggregated_feature = cnn_aggregator(community_batch)
                    aggregated_feature_squeezed = aggregated_feature.squeeze(-1)

                    concatenated_features = torch.cat((node_features[indices_batch], aggregated_feature_squeezed), dim=1)
                    mlp_output = mlp(concatenated_features)

                    all_concatenated_features.append(mlp_output)

                    similarity_scores = cosine_similarity(aggregated_feature_squeezed, node_features[indices_batch])
                    all_similarity_scores.append(similarity_scores)

        concatenated_features_tensor = torch.cat(all_concatenated_features, dim=0)
        similarity_scores_tensor = torch.cat(all_similarity_scores, dim=0)


    save_features(concatenated_features_tensor, f'{dataset_name}_{model_name}_concatenated_features.npy')
    save_features(similarity_scores_tensor, f'soft-{dataset_name}_{model_name}_similarity_scores.npy')

    return concatenated_features_tensor, similarity_scores_tensor

if __name__ == "__main__":
    dataset_name = 'cora'
    model_name = 'SEComm'
    dataset = load_dataset(dataset_name)

    concatenated_features, similarity_scores = aggregate_and_concatenate_features(dataset, dataset_name, model_name, batch_size=64)

    print(f"Concatenated feature shape after MLP: {concatenated_features.shape}")
    print(f"Similarity scores shape: {similarity_scores.shape}")