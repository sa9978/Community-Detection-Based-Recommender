import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

# Utility functions for data and feature loading
from utils import prepare_pairs, find_optimal_sigma, gaussian_similarity, load_concatenated_features
from data_loader import load_dataset
from aggregator import load_community_data

# Define the GraphSAGE layer
class GraphSAGEConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, aggregator_type='mean'):
        super(GraphSAGEConv, self).__init__()
        self.aggregator_type = aggregator_type
        self.linear = nn.Linear(input_dim * 2, hidden_dim)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, node_features, neighbor_features):
        if self.aggregator_type == 'mean':
            aggregated_neighbors = torch.mean(neighbor_features, dim=1)
        elif self.aggregator_type == 'sum':
            aggregated_neighbors = torch.sum(neighbor_features, dim=1)
        elif self.aggregator_type == 'max':
            aggregated_neighbors, _ = torch.max(neighbor_features, dim=1)
        else:
            raise ValueError("Invalid aggregator type. Choose from 'mean', 'sum', or 'max'.")

        combined = torch.cat([node_features, aggregated_neighbors], dim=1)
        return F.relu(self.linear(combined))

# GraphSAGE model with multiple layers
class GraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, aggregator_type='mean'):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList([GraphSAGEConv(input_dim if i == 0 else hidden_dim, hidden_dim, aggregator_type) for i in range(num_layers)])

    def forward(self, node_features, neighbor_features_list):
        for layer, neighbor_features in zip(self.layers, neighbor_features_list):
            node_features = layer(node_features, neighbor_features)
        return node_features

# Simple scorer for item-to-item recommendations
class GraphSAGEScorer(nn.Module):
    def __init__(self, embedding_dim):
        super(GraphSAGEScorer, self).__init__()
        self.bias = nn.Parameter(torch.zeros(embedding_dim, 1))

    def forward(self, embeddings):
        scores = torch.matmul(embeddings, embeddings.t()) + self.bias
        return scores

# Complete GraphSAGE-based model for recommendation
class GraphSAGEModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, aggregator_type='mean'):
        super(GraphSAGEModel, self).__init__()
        self.graphsage = GraphSAGE(input_dim, hidden_dim, num_layers, aggregator_type)
        self.scorer = GraphSAGEScorer(hidden_dim)

    def forward(self, node_features, neighbor_features_list):
        node_embeddings = self.graphsage(node_features, neighbor_features_list)
        scores = self.scorer(node_embeddings)
        return scores

def train_graphsage(model, features, neighbor_features_list, community_data, sigma, cd_model, dataset_name, epochs=50, lr=1e-6, batch_size=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    features = features.to(device, dtype=torch.float32)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    writer = SummaryWriter(log_dir=f'runs/light_Recommender-{cd_model}-{dataset_name}')
    loss_fn = nn.MarginRankingLoss(margin=1.0)

    print("Starting training...")
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        print(f"Starting Epoch {epoch + 1}/{epochs}")

        for i in tqdm(range(0, features.size(0), batch_size), desc=f'Epoch {epoch + 1}/{epochs}'):
            batch_features = features[i:i + batch_size]

            # Generate positive and negative pairs using community_data
            pos_pairs, neg_pairs = prepare_pairs(features, community_data, i)
            if not pos_pairs or not neg_pairs:
                print(f"Skipping batch {i // batch_size + 1} due to lack of pairs.")
                continue

            # Vectorized similarity computation
            pos_sim = torch.tensor([gaussian_similarity(features[pair[0]], features[pair[1]], sigma) for pair in pos_pairs], device=device)
            neg_sim = torch.tensor([gaussian_similarity(features[pair[0]], features[pair[1]], sigma) for pair in neg_pairs], device=device)

            # Ensure the loss function has valid input tensors
            pos_sim.requires_grad = True
            neg_sim.requires_grad = True

            # Compute the loss
            loss = loss_fn(pos_sim, neg_sim, torch.ones_like(pos_sim, device=device))
            optimizer.zero_grad()  # Clear gradients before the backward pass
            loss.backward()  # Compute gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients
            optimizer.step()  # Update weights

            total_loss += loss.item()

            if (i // batch_size + 1) % 10 == 0:  # Every 10 batches, print the current loss
                print(f"Batch {i // batch_size + 1}: Current loss = {loss.item():.4f}")

        avg_loss = total_loss / (features.size(0) // batch_size)
        writer.add_scalar('Loss/train', avg_loss, epoch)
        print(f"Epoch {epoch + 1}/{epochs} completed. Avg Loss: {avg_loss:.4f}")

        # Step the scheduler
        scheduler.step(avg_loss)

    writer.close()
    print("Training complete. Check TensorBoard for details.")

# Main section to set up and start training
if __name__ == "__main__":
    dataset_name = 'cora'
    model_name = 'secomm'
    dataset = load_dataset(dataset_name)

    concatenated_features = load_concatenated_features(dataset_name, model_name)
    community_data = load_community_data(dataset_name, model_name)
    # sigma = find_optimal_sigma(concatenated_features)

    input_dim = concatenated_features.size(1)
    hidden_dim = 128
    num_layers = 2
    model = GraphSAGEModel(input_dim, hidden_dim, num_layers, aggregator_type='mean')

    neighbor_features_list = [concatenated_features for _ in range(num_layers)]

        # Revise the training call with updated parameters
    train_graphsage(
        model,
        concatenated_features,
        neighbor_features_list,
        community_data,
        0.5,  # Use the calculated sigma value
        cd_model=model_name,
        dataset_name=dataset_name,
        epochs=1000,
        lr=1e-12,  # Lowered learning rate for stability
        batch_size=64  # Adjusted batch size
    )