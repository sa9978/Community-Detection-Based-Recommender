import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_scipy_sparse_matrix
import numpy as np
import scipy.sparse as sp
import random
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import ndcg_score
from utils import load_concatenated_features

class Args:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr = 0.0001
    epochs = 400
    batch_size = 128
    layer_size = '[64, 64]' 
    regs = [1e-1] 
    node_dropout = 0.2 

args = Args()

class NGCF(nn.Module):
    def __init__(self, input_features, norm_adj, args):
        super(NGCF, self).__init__()
        self.device = args.device
        self.norm_adj = norm_adj
        self.dropout = args.node_dropout

        self.weight_dict = nn.ParameterDict()
        layers = [input_features.shape[1]] + eval(args.layer_size)
        for k in range(len(layers) - 1):
            self.weight_dict[f'W_gc_{k}'] = nn.Parameter(nn.init.xavier_uniform_(torch.empty(layers[k], layers[k + 1])))
            self.weight_dict[f'b_gc_{k}'] = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1, layers[k + 1])))

    def forward(self, features):
        ego_embeddings = features
        all_embeddings = [ego_embeddings]

        for k in range(len(eval(args.layer_size))):
            ego_embeddings = F.dropout(ego_embeddings, p=self.dropout, training=self.training)
            side_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            sum_embeddings = torch.matmul(side_embeddings, self.weight_dict[f'W_gc_{k}']) + self.weight_dict[f'b_gc_{k}']
            ego_embeddings = F.leaky_relu(sum_embeddings)
            all_embeddings.append(ego_embeddings)

        all_embeddings = torch.cat(all_embeddings, 1)
        return all_embeddings

    def create_bpr_loss(self, item_embeddings, pos_items, neg_items):
        pos_scores = torch.sum(item_embeddings[pos_items], axis=1)
        neg_scores = torch.sum(item_embeddings[neg_items], axis=1)

        maxi = F.logsigmoid(pos_scores - neg_scores)
        bpr_loss = -torch.mean(maxi)

        reg_loss = torch.norm(item_embeddings) ** 2 * args.regs[0] / len(pos_items)

        return bpr_loss + reg_loss, bpr_loss, reg_loss

    def recommend(self, features, item, top_k=5):
        """Recommend top K items for a given item based on learned features."""
        with torch.no_grad():
            item_embeddings = self.forward(features)
            item_embedding = item_embeddings[item].unsqueeze(0) 

            similarities = F.cosine_similarity(item_embedding, item_embeddings)
            top_k_indices = torch.topk(similarities, top_k + 1).indices[1:] 

            return top_k_indices.cpu().numpy() 

import os
from sklearn.metrics import ndcg_score
import numpy as np

def precision_at_k(r, k):
    """Calculates precision at K"""
    r = np.asarray(r)[:k]  # Only consider the top-k predictions
    return np.mean(r)

def recall_at_k(r, k, num_relevant):
    """Calculates recall at K"""
    r = np.asarray(r)[:k]
    return np.sum(r) / num_relevant

def average_precision_at_k(r, k):
    """Calculates average precision at K"""
    r = np.asarray(r)[:k]
    precisions = [precision_at_k(r, i + 1) for i in range(k) if r[i]]
    if not precisions:
        return 0.
    return np.mean(precisions)

def ndcg_at_k(r, k):
    """Calculates NDCG at K"""
    return ndcg_score([r], [range(len(r))], k=k)

def evaluate_model(model, features, data, dataset_name, top_k=[5, 10]):
    """Evaluate the recommendation performance and save metrics to a file."""
    
    # Directory for saving metrics
    metric_dir = './metrics'
    os.makedirs(metric_dir, exist_ok=True)
    
    # File path for saving the metrics
    metric_filepath = os.path.join(metric_dir, f'cd-ngcf_SEComm-{dataset_name}.txt')
    
    # Initialize the metrics dictionary
    all_metrics = {}
    for k in top_k:
        all_metrics[f"P@{k}"] = []
        all_metrics[f"R@{k}"] = []
        all_metrics[f"MAP@{k}"] = []
        all_metrics[f"NDCG@{k}"] = []
    
    # item_embeddings = model(features)

    for node_id in range(data.num_nodes):
        true_label = data.y[node_id].item()  # True community or class
        for k in top_k:
            recommended_nodes = model.recommend(features, node_id, top_k=max(top_k))
            recommended_labels = data.y[recommended_nodes].cpu().numpy()

            relevant = (recommended_labels == true_label).astype(int)  # Relevant items in top K
            num_relevant = (data.y == true_label).sum().item()
            # Precision and Recall
            all_metrics[f"P@{k}"].append(precision_at_k(relevant, k))
            all_metrics[f"R@{k}"].append(recall_at_k(relevant, k, num_relevant))
            # MAP
            all_metrics[f"MAP@{k}"].append(average_precision_at_k(relevant, k))
            # NDCG
            all_metrics[f"NDCG@{k}"].append(ndcg_at_k(relevant, k))

    # Calculate mean for each metric
    mean_metrics = {metric: np.mean(all_metrics[metric]) for metric in all_metrics}

    # Save metrics to a file
    with open(metric_filepath, 'w') as f:
        f.write(f"Metrics for dataset: {dataset_name}\n")
        for metric, value in mean_metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    print(f"Metrics saved to {metric_filepath}")
    
    return mean_metrics

def main():
    community_detection_method = 'secomm'
    dataset_name = 'cora'

    dataset = Planetoid(root=f'./data/{dataset_name}', name=dataset_name)
    data = dataset[0]

    concatenated_features = load_concatenated_features(dataset_name, community_detection_method).to(args.device)

    edge_index = data.edge_index
    adj = to_scipy_sparse_matrix(edge_index, num_nodes=data.num_nodes)

    norm_adj = sp.coo_matrix(adj).tocsr()

    row, col = norm_adj.nonzero()
    values = norm_adj.data
    norm_adj_tensor = torch.sparse.FloatTensor(torch.LongTensor([row, col]),
                                               torch.FloatTensor(values),
                                               torch.Size(norm_adj.shape)).to(args.device)

    model = NGCF(input_features=concatenated_features, norm_adj=norm_adj_tensor, args=args).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5) 

    tensorboard_dir = f'./run/cd_ngcf_{community_detection_method}_{dataset_name}/'
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(tensorboard_dir)

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        loss, bpr_loss, reg_loss = 0., 0., 0.
        n_batch = concatenated_features.shape[0] // args.batch_size + 1

        for _ in range(n_batch):
            all_items = list(range(concatenated_features.shape[0]))
            pos_items = random.sample(all_items, args.batch_size)
            neg_items = random.sample([item for item in all_items if item not in pos_items], args.batch_size)

            item_embeddings = model(concatenated_features)
            batch_loss, batch_bpr_loss, batch_reg_loss = model.create_bpr_loss(item_embeddings, pos_items, neg_items)

            batch_loss.backward()
            optimizer.step()

            loss += batch_loss.item()
            bpr_loss += batch_bpr_loss.item()
            reg_loss += batch_reg_loss.item()

        scheduler.step() 

        writer.add_scalar('Loss/total_loss', loss, epoch)
        writer.add_scalar('Loss/bpr_loss', bpr_loss, epoch)
        writer.add_scalar('Loss/reg_loss', reg_loss, epoch)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{args.epochs}], Total Loss: {loss:.4f}, BPR Loss: {bpr_loss:.4f}, Reg Loss: {reg_loss:.4f}')

    writer.close()
    print("Training completed.")

    evaluate_model(model, concatenated_features, data)

if __name__ == "__main__":
    main()
