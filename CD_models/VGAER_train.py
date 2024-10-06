from __future__ import division
from __future__ import print_function

import argparse
import time
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from matplotlib import pyplot as plt
from torch import optim
import pandas as pd
from torch_geometric.utils import to_dense_adj
from sklearn.metrics import accuracy_score
from torch_geometric.datasets import Planetoid
import os
from CD_models.VGAER.model import VGAERModel
import torch.nn.functional as F
from CD_models.VGAER.cluster import community
from CD_models.VGAER.NMI import NMI
from CD_models.VGAER.Qvalue import Q
from torch.utils.tensorboard import SummaryWriter

def vgaer(graph, dataset):
    # Variables instead of CLI arguments

    if dataset == "cora":
        model = 'gcn_vae'
        seed = 42
        epochs = 25000
        hidden1 = 8
        hidden2 = 2
        lr = 0.05
        dropout = 0.0
        cluster = 7

    # Set device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # TensorBoard writer initialization
    writer = SummaryWriter(log_dir=f'runs/{dataset}/VGAER')

    graph = graph[0]

    # Load graph adjacency matrix
    A = to_dense_adj(graph.edge_index).squeeze(0)
    A_orig = A.detach().numpy()
    A_orig_ten = A.to(device)
    label_orig = graph.y.detach().numpy()

    # Compute B matrix
    K = 1 / (A.sum().item()) * (A.sum(dim=1).reshape(A.shape[0], 1) @ A.sum(dim=1).reshape(1, A.shape[0]))
    B = A - K
    B = B.to(device)

    # Compute A_hat matrix
    A = A + torch.eye(A.shape[0])
    D = torch.diag(torch.pow(A.sum(dim=1), -0.5))
    A_hat = D @ A @ D
    A_hat = A_hat.to(device)
    A = A.to(device)

    # Extract features
    feats = B
    in_dim = feats.shape[-1]

    # Create VGAER model
    vgaer_model = VGAERModel(in_dim, hidden1, hidden2, device).to(device)

    # Create optimizer
    optimizer = torch.optim.Adam(vgaer_model.parameters(), lr=lr)
    print('Total Parameters:', sum([p.nelement() for p in vgaer_model.parameters()]))

    if dataset == "citeseer":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    def compute_loss_para(adj):
        pos_weight = ((adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum())
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
        weight_mask = adj.view(-1) == 1
        weight_tensor = torch.ones(weight_mask.size(0)).to(device)
        weight_tensor[weight_mask] = pos_weight
        return weight_tensor, norm

    weight_tensor, norm = compute_loss_para(A)

    for epoch in range(epochs):
        epoch_start = time.perf_counter()  # Start time tracking

        # Training mode
        vgaer_model.train()
        recovered = vgaer_model.forward(A_hat, feats)
        logits = recovered[0]
        hidemb = recovered[1]
        loss = norm * F.binary_cross_entropy(logits.view(-1), A_orig_ten.view(-1), weight=weight_tensor)
        kl_divergence = 0.5 / logits.size(0) * (
                1 + 2 * vgaer_model.log_std - vgaer_model.mean ** 2 - torch.exp(vgaer_model.log_std) ** 2).sum(1).mean()
        loss -= kl_divergence

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if dataset == "citeseer":
            scheduler.step()

        # TensorBoard logging for loss
        writer.add_scalar('VGAER/Training_Loss', loss.item(), epoch)

        if epoch == epochs - 1:
            os.makedirs('models', exist_ok=True)
        
            # Save the model with parameters and dataset name
            model_path = f"models/VGAER_{dataset}_hidden{hidden1}_{hidden2}_dropout{dropout}_lr{lr}.pth"
            torch.save(vgaer_model.state_dict(), model_path)
            print(f"Model saved to {model_path}")

            # Final epoch: calculate community and log results
            hidemb = hidemb.cpu()
            commu_pred = community(hidemb, cluster)
            Q(A_orig, np.eye(cluster)[commu_pred])
            nmi = NMI(commu_pred, label_orig)
            accuracy = accuracy_score(label_orig, commu_pred)

            # Save community prediction results
            np.save(os.path.join('Communities', f'{dataset}-VGAER.npy'), commu_pred)

            # TensorBoard logging for NMI and accuracy
            writer.add_scalar('VGAER/Final_NMI', nmi, epoch)
            writer.add_scalar('VGAER/Final_Accuracy', accuracy, epoch)

            print(f"Acc={accuracy}, NMI={nmi}")

        # TensorBoard logging for epoch time
        epoch_time = time.perf_counter() - epoch_start
        writer.add_scalar('VGAER/Epoch_Time', epoch_time, epoch)

        print(f"Epoch: {epoch + 1}, train_loss={loss.item():.5f}, epoch_time={epoch_time:.2f}s")

    # Close TensorBoard writer
    writer.close()


if __name__ == '__main__':
    dataset = Planetoid(root='/tmp/' + "pubmed", name="pubmed")
    vgaer(dataset, "pubmed")