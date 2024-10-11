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
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class Args:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embed_size = 64
    node_dropout = [0.1] 
    lr = 0.00001
    epochs = 400
    batch_size = 64
    layer_size = '[64, 64]' 
    regs = [1e-1] 

args = Args()


class NGCF(nn.Module):
    def __init__(self, n_item, norm_adj, args):
        super(NGCF, self).__init__()
        self.n_item = n_item
        self.device = args.device
        self.emb_size = args.embed_size
        self.norm_adj = norm_adj


        self.embedding_dict = nn.ParameterDict({
            'item_emb': nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_item, self.emb_size)))
        })

        self.weight_dict = nn.ParameterDict()
        layers = [self.emb_size] + eval(args.layer_size)
        for k in range(len(layers) - 1):
            self.weight_dict[f'W_gc_{k}'] = nn.Parameter(nn.init.xavier_uniform_(torch.empty(layers[k], layers[k + 1])))
            self.weight_dict[f'b_gc_{k}'] = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1, layers[k + 1])))

    def forward(self):
        ego_embeddings = self.embedding_dict['item_emb']
        all_embeddings = [ego_embeddings]


        for k in range(len(eval(args.layer_size))):
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
        mf_loss = -torch.mean(maxi)


        regularizer = torch.norm(item_embeddings) ** 2
        emb_loss = regularizer * args.regs[0] / len(pos_items)

        return mf_loss + emb_loss, mf_loss, emb_loss

    def recommend(self, item, top_k=5):
        """Recommend top K items for a given item based on learned embeddings."""
        with torch.no_grad():
            item_embeddings = self.forward()
            item_embedding = item_embeddings[item].unsqueeze(0)  # Shape: (1, embed_size)


            similarities = F.cosine_similarity(item_embedding, item_embeddings)
            top_k_indices = torch.topk(similarities, top_k + 1).indices[1:] 

            return top_k_indices.cpu().numpy()

import os
import numpy as np

def precision_at_k(relevant, k):
    return np.sum(relevant[:k]) / k

def recall_at_k(relevant, k, num_relevant_items):
    return np.sum(relevant[:k]) / num_relevant_items if num_relevant_items > 0 else 0

def average_precision_at_k(relevant, k):
    relevant = relevant[:k]
    precisions = [precision_at_k(relevant, i + 1) for i in range(k) if relevant[i]]
    return np.mean(precisions) if precisions else 0

def ndcg_at_k(relevant, k):
    dcg = np.sum((2 ** relevant[:k] - 1) / np.log2(np.arange(2, k + 2)))
    ideal_relevant = np.sort(relevant)[::-1]
    idcg = np.sum((2 ** ideal_relevant[:k] - 1) / np.log2(np.arange(2, k + 2)))
    return dcg / idcg if idcg > 0 else 0

def evaluate_model(model, data, dataset_name, top_k=[5, 10]):
    """Evaluate the recommendation performance and save metrics to a file."""
    
    # Directory for saving metrics
    metric_dir = './metrics'
    os.makedirs(metric_dir, exist_ok=True)
    
    # File path for saving the metrics
    metric_filepath = os.path.join(metric_dir, f'ngcf_SEComm_{dataset_name}.txt')
    
    # Initialize the metrics dictionary
    all_metrics = {}
    for k in top_k:
        all_metrics[f"P@{k}"] = []
        all_metrics[f"R@{k}"] = []
        all_metrics[f"MAP@{k}"] = []
        all_metrics[f"NDCG@{k}"] = []
    
    # item_embeddings = model()

    for node_id in range(data.num_nodes):
        true_label = data.y[node_id].item()  # True community or class
        for k in top_k:
            recommended_nodes = model.recommend(node_id, top_k=max(top_k))
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

    edge_index = data.edge_index
    adj = to_scipy_sparse_matrix(edge_index, num_nodes=data.num_nodes)

    norm_adj = sp.coo_matrix(adj).tocsr() 

    row, col = norm_adj.nonzero()
    values = norm_adj.data

    norm_adj_tensor = torch.sparse.FloatTensor(torch.LongTensor([row, col]),
                                               torch.FloatTensor(values),
                                               torch.Size(norm_adj.shape)).to(args.device)

    n_item = data.num_nodes
    model = NGCF(n_item=n_item, norm_adj=norm_adj_tensor, args=args).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    tensorboard_dir = f'./run/ngcf_{community_detection_method}_{dataset_name}/'
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(tensorboard_dir)

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        loss, mf_loss, emb_loss = 0., 0., 0.
        n_batch = n_item // args.batch_size + 1

        for _ in range(n_batch):
            all_items = list(range(n_item))
            pos_items = random.sample(all_items, args.batch_size)
            neg_items = random.sample([item for item in all_items if item not in pos_items], args.batch_size)

            item_embeddings = model()
            batch_loss, batch_mf_loss, batch_emb_loss = model.create_bpr_loss(item_embeddings, pos_items, neg_items)

            batch_loss.backward()
            optimizer.step()

            loss += batch_loss.item()
            mf_loss += batch_mf_loss.item()
            emb_loss += batch_emb_loss.item()

        writer.add_scalar('Loss/total_loss', loss, epoch)
        writer.add_scalar('Loss/mf_loss', mf_loss, epoch)
        writer.add_scalar('Loss/emb_loss', emb_loss, epoch)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{args.epochs}], Total Loss: {loss:.4f}, MF Loss: {mf_loss:.4f}, Emb Loss: {emb_loss:.4f}')

    writer.close()
    print("Training completed.")

    evaluate_model(model, data, top_k=5)

if __name__ == "__main__":
    main()
