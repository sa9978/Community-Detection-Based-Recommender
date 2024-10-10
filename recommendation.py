import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torch.optim as optim
from tqdm import tqdm

# Import custom utility functions
from utils import prepare_pairs, find_optimal_sigma, gaussian_similarity, load_concatenated_features
from data_loader import load_dataset
from aggregator import load_community_data

# Define the PinSAGEModel components
class LinearProjector(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dims)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, features):
        return self.linear(features)

class WeightedSAGEConv(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.Q = nn.Linear(input_dim, hidden_dim)
        self.W = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.Q.weight)
        nn.init.xavier_uniform_(self.W.weight)

    def forward(self, h, neighbors=None):
        if neighbors is None or neighbors.shape[0] == 0:
            return F.relu(self.W(torch.cat([h, torch.zeros_like(h)], dim=1)))

        h_neighbors = F.relu(self.Q(neighbors))
        agg_neighbors = torch.mean(h_neighbors, dim=0)
        h_combined = torch.cat([h, agg_neighbors], dim=1)
        return F.relu(self.W(h_combined))

class SAGENet(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([WeightedSAGEConv(input_dim, hidden_dim) for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class ItemToItemScorer(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(embedding_dim, 1))

    def forward(self, embeddings):
        scores = torch.matmul(embeddings, embeddings.t()) + self.bias
        return scores

class PinSAGEModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers):
        super().__init__()
        self.proj = LinearProjector(input_dim, hidden_dim)
        self.sage = SAGENet(hidden_dim, hidden_dim, n_layers)
        self.scorer = ItemToItemScorer(hidden_dim)

    def forward(self, node_features):
        projected_features = self.proj(node_features)
        node_embeddings = self.sage(projected_features)
        scores = self.scorer(node_embeddings)
        return scores

# Function to create community-based batches
def create_batches_by_community(community_data, batch_size):
    community_batches = []
    unique_communities = np.unique(community_data)

    for community in unique_communities:
        nodes_in_community = np.where(community_data == community)[0]
        for i in range(0, len(nodes_in_community), batch_size):
            batch_indices = nodes_in_community[i:i + batch_size]
            community_batches.append(batch_indices)

    return community_batches

# Revised training loop for PinSAGEModel with community-based batches
def train_recommender_system(model, community_data, features, sigma, dataset_name="citeseer", model_name="secomm", epochs=100, lr=0.001, batch_size=128, accumulate_steps=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    features = features.to(device, dtype=torch.float32)
    community_data = torch.tensor(community_data, device=device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    writer = SummaryWriter(log_dir=f'runs/MaxMarginRecommender_{dataset_name}_{model_name}')
    loss_fn = nn.MarginRankingLoss(margin=1.0)

    for epoch in range(epochs):
        total_loss = 0
        model.train()

        # Generate community-based mini-batches
        community_batches = create_batches_by_community(community_data.cpu().numpy(), batch_size)
        np.random.shuffle(community_batches)  # Shuffle the batches to ensure randomness

        for batch_idx, batch_indices in enumerate(tqdm(community_batches, desc=f'Epoch {epoch + 1}/{epochs}')):
            batch_positive_pairs = []
            batch_negative_pairs = []

            # Generate positive and negative pairs for each node in the batch
            for node_idx in batch_indices:
                positive_pairs, negative_pairs = prepare_pairs(features, community_data.cpu().numpy(), node_idx=node_idx)
                batch_positive_pairs.extend(positive_pairs)
                batch_negative_pairs.extend(negative_pairs)

            if not batch_positive_pairs or not batch_negative_pairs:
                continue  # Skip if no pairs found

            pos_sim = torch.tensor([gaussian_similarity(features[i], features[j], sigma) for i, j in batch_positive_pairs], device=device, dtype=torch.float32)
            neg_sim = torch.tensor([gaussian_similarity(features[i], features[j], sigma) for i, j in batch_negative_pairs], device=device, dtype=torch.float32)

            loss = loss_fn(pos_sim, neg_sim, torch.ones_like(pos_sim))
            loss = loss / accumulate_steps
            loss.backward()

            if (batch_idx + 1) % accumulate_steps == 0 or (batch_idx + 1) == len(community_batches):
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulate_steps

        avg_loss = total_loss / len(community_batches)
        writer.add_scalar('Loss/train', avg_loss, epoch)
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")

    writer.close()
    print("Training completed. You can visualize the results using TensorBoard.")

if __name__ == "__main__":
    # Data Loading Integration
    dataset_name = 'citeseer'
    model_name = 'secomm'
    dataset = load_dataset(dataset_name)

    # Load concatenated features and community data
    concatenated_features = load_concatenated_features(dataset_name, model_name)
    community_data = load_community_data(dataset_name, model_name)

    # Compute the number of unique communities and the optimal sigma value
    num_communities = len(np.unique(community_data))
    sigma = find_optimal_sigma(concatenated_features)

    # Initialize and train the PinSAGEModel
    input_dim = concatenated_features.size(1)
    hidden_dim = 128
    n_layers = 2
    model = PinSAGEModel(input_dim, hidden_dim, n_layers)

    # Start training with community-based batching
    train_recommender_system(model, community_data, concatenated_features, sigma, dataset_name=dataset, model_name=model_name, epochs=50, lr=0.001)