import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from CD_models.NOCD.model import GCN, BerpoDecoder, get_edge_sampler, NoImprovementStopping, ModelSaver, to_sparse_tensor, l2_reg_loss
from sklearn.metrics import accuracy_score
from CD_models.NOCD.data import load_planetoid_data
from CD_models.NOCD.metrics import get_nmi
from sklearn.preprocessing import normalize
from torch.utils.tensorboard import SummaryWriter
import time

def nocd(dataset, dataset_name):
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=f'runs/{dataset_name}/NOCD')

    print("Load Data...")
    # Load dataset and graph
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph = load_planetoid_data(dataset)

    A = graph['A']
    X = torch.FloatTensor(graph['X'])  # Node features
    adj_norm = GCN.normalize_adj(A)    # Normalized adjacency matrix
    Z_gt = torch.FloatTensor(graph['Z'])  # Ground truth communities
    N, K = Z_gt.shape
    print("*** Processed ***")

    # Model, optimizer, and parameters
    hidden_sizes = [128]    # hidden sizes of the GNN
    weight_decay = 1e-2     # strength of L2 regularization on GNN weights
    dropout = 0.5           # whether to use dropout
    batch_norm = True       # whether to use batch norm
    lr = 1e-3               # learning rate
    max_epochs = 500        # number of epochs to train
    display_step = 25       # how often to compute validation loss
    balance_loss = True     # whether to use balanced loss
    stochastic_loss = True  # whether to use stochastic or full-batch training
    batch_size = 20000      # batch size (only for stochastic training)

    # Normalize and convert X to sparse tensor for PyTorch
    x_norm = normalize(X)
    x_norm = torch.FloatTensor(x_norm)
    x_norm = to_sparse_tensor(x_norm).to(device)

    # Get edge sampler and set up model
    sampler = get_edge_sampler(A, batch_size, batch_size, num_workers=5)
    gnn = GCN(x_norm.shape[1], hidden_sizes, K, batch_norm=batch_norm, dropout=dropout).to(device)
    adj_norm = gnn.normalize_adj(A)
    decoder = BerpoDecoder(N, A.nnz, balance_loss=balance_loss)
    opt = torch.optim.Adam(gnn.parameters(), lr=lr)

    # Training loop
    val_loss = np.inf
    validation_fn = lambda: val_loss
    early_stopping = NoImprovementStopping(validation_fn, patience=10)
    model_saver = ModelSaver(gnn)

    for epoch, batch in enumerate(sampler):
        if epoch > max_epochs:
            break

        # Start time measurement for epoch
        epoch_start = time.perf_counter()

        if epoch % display_step == 0:
            with torch.no_grad():
                gnn.eval()
                Z = F.relu(gnn(x_norm, adj_norm))
                pred_labels = torch.argmax(Z, dim=1).cpu().numpy()
                true_labels = torch.argmax(Z_gt, dim=1).cpu().numpy()

                # Accuracy and NMI metrics
                accuracy = accuracy_score(true_labels, pred_labels)
                nmi_score = get_nmi(gnn, x_norm, adj_norm, Z_gt)

                # Validation loss
                val_loss = decoder.loss_full(Z, A)
                print(f'Epoch {epoch:4d}, loss.full = {val_loss:.4f}, nmi = {nmi_score:.2f}, accuracy = {accuracy:.4f}')

                # TensorBoard logging
                writer.add_scalar('NOCD/Validation_Loss', val_loss.item(), epoch)
                writer.add_scalar('NOCD/Accuracy', accuracy, epoch)
                writer.add_scalar('NOCD/NMI', nmi_score, epoch)

                # Early stopping and model saving
                early_stopping.next_step()
                if early_stopping.should_save():
                    model_saver.save()
                if early_stopping.should_stop():
                    print(f'Breaking due to early stopping at epoch {epoch}')
                    break
        
        # Training step
        gnn.train()
        opt.zero_grad()
        Z = F.relu(gnn(x_norm, adj_norm))
        ones_idx, zeros_idx = batch
        if stochastic_loss:
            loss = decoder.loss_batch(Z, ones_idx, zeros_idx)
        else:
            loss = decoder.loss_full(Z, A)
        loss += l2_reg_loss(gnn, scale=weight_decay)
        loss.backward()
        opt.step()

        # TensorBoard logging for training loss
        writer.add_scalar('NOCD/Training_Loss', loss.item(), epoch)

        # End time measurement for epoch
        epoch_time = time.perf_counter() - epoch_start
        writer.add_scalar('NOCD/Epoch_Time', epoch_time, epoch)

    # Save the final predicted labels
    pred_labels = torch.argmax(Z, dim=1).cpu().numpy()
    np.save(os.path.join('Communities', f'{dataset_name}-NOCD.npy'), pred_labels)

    # Close the TensorBoard writer
    writer.close()