import argparse
import os.path as osp
import random
from time import perf_counter as t
import yaml
from yaml import SafeLoader
from sklearn.preprocessing import normalize
from sklearn import metrics
import math
import numpy as np
import os
import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import Planetoid, Coauthor
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GCNConv
from torch.utils.tensorboard import SummaryWriter

from CD_models.SEComm.model import Encoder, Model, drop_feature, SelfExpr, ClusterModel
from CD_models.SEComm.eval import label_classification
from CD_models.SEComm.utils import enhance_sim_matrix, post_proC, err_rate, best_map, load_wiki

from sklearn.decomposition import PCA

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Divise is", device)

def secomm(dataset, dataset_name):
    printvals = False

    def test(x, edge_index, y):
        gracemodel.eval()
        z = gracemodel(x, edge_index)
        label_classification(z, y, ratio=0.1)
        torch.cuda.empty_cache()


    def test_spectral(c, y_train, n_class):
        y_train_x, _ = post_proC(c, n_class, 4, 1)
        print("Spectral Clustering Done.. Finding Best Fit..")
        scores = err_rate(y_train.detach().cpu().numpy(), y_train_x)
        return scores




    def train(x, edge_index, batch_size, cfg):
        drop_edge_rate_1 = cfg['drop_edge_rate_1']
        drop_edge_rate_2 = cfg['drop_edge_rate_2']
        drop_feature_rate_1 = cfg['drop_feature_rate_1']
        drop_feature_rate_2 = cfg['drop_feature_rate_2']
        gracemodel.train()
        graceoptimizer.zero_grad()
        edge_index_1 = dropout_adj(edge_index, p=drop_edge_rate_1)[0]
        edge_index_2 = dropout_adj(edge_index, p=drop_edge_rate_2)[0]
        x_1 = drop_feature(x, drop_feature_rate_1)
        x_2 = drop_feature(x, drop_feature_rate_2)
        z1 = gracemodel(x_1, edge_index_1)
        z2 = gracemodel(x_2, edge_index_2)
        loss = gracemodel.loss(z1, z2, batch_size=batch_size)
        loss.backward()

        # TensorBoard logging
        writer.add_scalar('Pretraining/GRACE_Loss', loss.item(), epoch)

        graceoptimizer.step()
        torch.cuda.empty_cache()
        return loss.item()

    def self_expressive_train(x_train, cfg, n_class):
        max_epoch = cfg['se_epochs']
        alpha = cfg['se_loss_reg']
        patience = cfg['patience']
        x1 = x_train
        best_loss = 1e9
        bad_count = 0
        for epoch in range(max_epoch):
            semodel.train()
            seoptimizer.zero_grad()
            c, x2 = semodel(x1)
            se_loss = torch.norm(x1-x2)
            reg_loss = torch.norm(c)
            loss = se_loss + alpha*reg_loss

            # TensorBoard logging
            writer.add_scalar('SelfExpressive/SE_Loss', se_loss.item(), epoch)
            writer.add_scalar('SelfExpressive/Reg_Loss', reg_loss.item(), epoch)
            writer.add_scalar('SelfExpressive/Full_Loss', loss.item(), epoch)

            loss.backward()
            seoptimizer.step()
            torch.cuda.empty_cache()
            print('se_loss: {:.9f}'.format(se_loss.item()), 'reg_loss: {:.9f}'.format(reg_loss.item()), end=' ')
            print('full_loss: {:.9f}'.format(loss.item()), flush=True)
            if loss.item()<best_loss:
                if torch.cuda.is_available():
                    best_c = c.to(device)
                else:
                    best_c = c
                bad_count = 0
                best_loss = loss.item()
            else:
                bad_count += 1
                if bad_count == patience:
                    break
        
        C = best_c
        C = C.cpu().detach().numpy()
        L = enhance_sim_matrix(C, n_class, 4, 1)
        torch.cuda.empty_cache()
        return L


    def batch_cluster_train(x, edge_index, from_list, to_list, val_list,
                            cfg, n_class, batch_size, MODEL_PATH):
        max_epoch = cfg['cluster_epochs']
        alpha = cfg['cluster_loss_reg']
        beta = cfg['final_loss_reg']
        patience = cfg['patience']
        drop_edge_rate_1 = cfg['drop_edge_rate_1']
        drop_edge_rate_2 = cfg['drop_edge_rate_2']
        drop_feature_rate_1 = cfg['drop_feature_rate_1']
        drop_feature_rate_2 = cfg['drop_feature_rate_2']
        best_loss = 1e9
        bad_count = 0
        for epoch in range(max_epoch+1):
            epoch_start = t()  # Start timing the epoch

            gracemodel.train()
            clustermodel.train()
            fulloptimizer.zero_grad()

            edge_index_1 = dropout_adj(edge_index, p=drop_edge_rate_1)[0]
            edge_index_2 = dropout_adj(edge_index, p=drop_edge_rate_2)[0]
            x_1 = drop_feature(x, drop_feature_rate_1)
            x_2 = drop_feature(x, drop_feature_rate_2)
            z1 = gracemodel(x_1, edge_index_1)
            z2 = gracemodel(x_2, edge_index_2)
            grace_loss = gracemodel.loss(z1, z2, batch_size=4000)
            
            z_full = clustermodel(gracemodel(x, edge_index))
            z_from = z_full[from_list]
            z_to = z_full[to_list]
            pred_similarity = torch.sum(z_from*z_to, dim=1)

            numer2 = torch.mm(z_full.T, z_full)
            denom2 = torch.norm(numer2)
            identity_mat = torch.eye(n_class)
            if torch.cuda.is_available():
                identity_mat = identity_mat.cuda()
            B = identity_mat/math.sqrt(n_class)
            C = numer2/denom2
            
            loss1 = F.mse_loss(pred_similarity, val_list)
            loss2 = torch.norm(B-C)
            loss = beta*grace_loss + loss1 + alpha*loss2

            # TensorBoard logging for epoch time
            writer.add_scalar('Time/Final_Training_Epoch_Time', t() - epoch_start, epoch)

            # TensorBoard logging
            writer.add_scalar('FinalTraining/Full_Loss', loss.item(), epoch)
            writer.add_scalar('FinalTraining/GRACE_Loss', grace_loss.item(), epoch)
            writer.add_scalar('FinalTraining/Loss1', loss1.item(), epoch)
            writer.add_scalar('FinalTraining/Loss2', loss2.item(), epoch)


            loss.backward()
            fulloptimizer.step()
            torch.cuda.empty_cache()
            print('Epoch:', epoch, 'full_loss: {:.5f}'.format(loss.item()), 'grace_loss: {:.5f}'.format(grace_loss.item()), end=' ')
            print('loss1: {:.5f}'.format(loss1.item()), 'loss2: {:.5f}'.format(loss2.item()), flush=True)

            # At the end of final training
            cluster_time = t() - start_cluster
            writer.add_scalar('Time/Final_Clustering_Training_Time', cluster_time)

            print("Final model training done.. time:", cluster_time)

            if loss2.item()<best_loss:
                bad_count = 0
                best_loss = loss2.item()
                torch.save(gracemodel.state_dict(), MODEL_PATH+"gracemodel_boosted")
                torch.save(clustermodel.state_dict(), MODEL_PATH+"clustermodel")
            else:
                bad_count += 1
                print("Model not improved for", bad_count, "consecutive epochs..")
                if bad_count == patience:
                    print("Early stopping Cluster Train...")
                    break

            if epoch%10 == 0:
                if torch.cuda.is_available():
                    y_pred = torch.argmax(z_full, dim=1).cpu().detach().numpy()
                    y_true = data.y.cpu().numpy()
                else:
                    y_pred = torch.argmax(z_full, dim=1).detach().numpy()
                    y_true = data.y.numpy()
                y_pred = best_map(y_true,y_pred)
                acc = metrics.accuracy_score(y_true, y_pred)
                nmi = metrics.normalized_mutual_info_score(y_true, y_pred)

                # TensorBoard logging
                writer.add_scalar('FinalTraining/Accuracy', acc, epoch)
                writer.add_scalar('FinalTraining/NMI', nmi, epoch)
                # writer.add_scalar('FinalTraining/F1_Macro', f1_macro, epoch)
                # writer.add_scalar('FinalTraining/F1_Micro', f1_micro, epoch)

                f1_macro = metrics.f1_score(y_true, y_pred, average='macro')
                f1_micro = metrics.f1_score(y_true, y_pred, average='micro')
                print("\n\nAc:", acc, "NMI:", nmi, "F1Ma:", f1_macro, "F1Mi:", f1_micro)

        return best_loss


    # Replace the argument parser with direct variable assignments
    # dataset_name = 'cora'  # Set your dataset name here
    gpu_id = 0
    config_path = os.path.join(os.path.dirname(__file__), 'SEComm', 'config.yaml')
    pretrain = 'T'
    args_path = './'  # Set your model path here

    # TensorBoard writer initialization
    writer = SummaryWriter(log_dir=f'runs/{dataset_name}/SEComm')

    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
    MODEL_PATH = "./Saved_Models/"+dataset_name+"/"
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH) 
    with open("CD_models/SEComm/config.yaml", 'r') as file:
        cfg = yaml.load(file, Loader=SafeLoader)[dataset_name]
        
    torch.manual_seed(cfg['seed'])
    random.seed(cfg['seed'])
    activation = ({'relu': F.relu, 'prelu': nn.PReLU()})[cfg['activation']]
    base_model = ({'GCNConv': GCNConv})[cfg['base_model']]

    # def get_dataset(path, name):
    #     assert name in ['Cora', 'CiteSeer', 'PubMed', 'Physics']

    #     if name == 'Physics':
    #         return Coauthor(path, name)
    #     else:
    #         return Planetoid(path, name)

    if dataset_name != "Wiki":
        path = osp.join(osp.expanduser('~'), 'datasets', dataset_name)
        # dataset = get_dataset(path, dataset)
        data = dataset[0]
        num_features = dataset.num_features
        if dataset_name == "Physics":
            pca = PCA(n_components=128)
            feats = pca.fit_transform(dataset[0].x)
            np.save("Saved_Models/Physics/Physics_PCA_Feats", feats)
            #feats = np.load("Saved_Models/Physics/Physics_PCA_Feats.npy")
            data.x = torch.FloatTensor(feats)
            num_features = data.x.shape[1]
    else:
        data, num_features = load_wiki(args_path)
    data = data.to(device)

    n_nodes = data.x.shape[0]
    n_class = max(data.y).item()+1
    batch_size = cfg['batch_size']
    if batch_size == 0:
        batch_size = n_nodes

    print("========", dataset_name, "========")
    print("Nodes    :", n_nodes)
    print("Features :", num_features)
    print("Classes  :", n_class)
    print("Distribution :", end=' ')
    for i in range(n_class-1):
        print(torch.sum(data.y==i).item(), end=', ')
    print(torch.sum(data.y==(n_class-1)).item(), end='\n')
    print("======================")

    encoder = Encoder(num_features, cfg['num_hidden'], activation,
                      base_model=base_model, k=cfg['num_layers']).to(device)
    gracemodel = Model(encoder, cfg['num_hidden'], cfg['num_proj_hidden'], cfg['tau']).to(device)
    graceoptimizer = torch.optim.Adam(gracemodel.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    semodel          = SelfExpr(batch_size).to(device)
    seoptimizer      = torch.optim.Adam(semodel.parameters(), lr=cfg['se_lr'], weight_decay=cfg['weight_decay'])
    clustermodel     = ClusterModel(cfg['num_hidden'], cfg['num_cl_hidden'], n_class, cfg['dropout']).to(device)
    clusteroptimizer = torch.optim.Adam(clustermodel.parameters(), lr=cfg['se_lr'], weight_decay=cfg['weight_decay'])

    #============== Pre-training Module ================#
    grace_time = 0
    if pretrain=='T':
        print("Pre-training GRACE model to get baseline embedding for Self Expressive Layer")
        start = t()
        prev = start
        for epoch in range(1, cfg['num_epochs'] + 1):
            epoch_start = t()
            loss = train(data.x, data.edge_index, batch_size, cfg)
            now = t()

            # TensorBoard logging for GRACE pre-training time per epoch
            writer.add_scalar('Time/GRACE_Pretraining_Epoch_Time', now - epoch_start, epoch)
            # TensorBoard logging
            writer.add_scalar('Pretraining/GRACE_Loss_Epoch', loss, epoch)

            print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, '
                  f'this epoch {now - prev:.4f}, total {now - start:.4f}', flush=True)
            prev = now
        grace_time = t()-start
        print("Saving pre-trained GRACE Model")
        torch.save(gracemodel.state_dict(), MODEL_PATH+"gracemodel")
    
    #============== Self-Expressive Layer training Module ================#
    se_time=0
    if pretrain=='T' or pretrain=='T1':
        print("Loading pre-trained GRACE model")
        gracemodel.load_state_dict(torch.load(MODEL_PATH+"gracemodel"))

        print("=== Supervised Accuracy test for GRACE Embeddings Generated ===")
        test(data.x, data.edge_index, data.y)

        start_se = t()
        X = gracemodel(data.x, data.edge_index).detach()
        if cfg['normalize']:
            print("Normalizing embeddings before Self Expressive layer training")
            X = normalize(X.cpu().numpy())
            X = torch.tensor(X).to(device)
        from_list = []
        to_list = []
        val_list = []
        for iters in range(cfg['iterations']):
            train_labels = random.sample(list(range(n_nodes)), batch_size)
            x_train = X[train_labels]
            y_train = data.y[train_labels]
            print("\n\n\nStarting self expressive train iteration:", iters+1)
            S = self_expressive_train(x_train, cfg, n_class)

            print("Performance of Spectral Clustering on the Similarity Matrix")
            scores = test_spectral(S, y_train, n_class)
            print(" Ac:", scores[0], "NMI:", scores[1], "F1:", scores[2])

            print("\nRetriving similarity values for point pairs")
            count = 0
            #TODO: Optimize this for faster runtime
            threshold = cfg['threshold']
            for i in range(batch_size):
                for j in range(batch_size):
                    if i == j:
                        continue
                    if S[i,j]>=(1-threshold) or (S[i,j]<=threshold and S[i,j]>=0):
                        from_list.append(train_labels[i])
                        to_list.append(train_labels[j])
                        val_list.append(S[i,j])
                        count+=1
            print("Included values for", count, "points out of", batch_size*batch_size)
        se_time = t()-start_se

        # TensorBoard logging for self-expressive training time
        writer.add_scalar('Time/SelfExpressive_Training_Time', se_time)

        print("Self Expressive Layer training done.. time:", se_time)
        np.save(MODEL_PATH+"from_list.npy", from_list)
        np.save(MODEL_PATH+"to_list.npy", to_list)
        np.save(MODEL_PATH+"val_list.npy",val_list)
    
    #============== Final full training Module ================#
    print("\n\n\nStarting final full training module")
    gracemodel.load_state_dict(torch.load(MODEL_PATH+"gracemodel"))
    from_list = np.load(MODEL_PATH+"from_list.npy")
    to_list = np.load(MODEL_PATH+"to_list.npy")
    val_list = np.load(MODEL_PATH+"val_list.npy")
    start_cluster = t()
    fulloptimizer = torch.optim.Adam((list(gracemodel.parameters()) + list(clustermodel.parameters())), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    convergence_loss = batch_cluster_train(data.x, data.edge_index, from_list, to_list, torch.FloatTensor(val_list).to(device),
                        cfg, n_class, batch_size, MODEL_PATH)
    cluster_time = t()-start_cluster

    # Calculate total time
    total_training_time = grace_time + se_time + cluster_time

    print("Final model training done.. time:", cluster_time)
    print("Total training time:", total_training_time)
    
    # TensorBoard logging for total training time
    writer.add_scalar('Time/Total_Training_Time', total_training_time)
    torch.cuda.empty_cache()
    # TensorBoard writer closing
    writer.close()

    print("\n=== Final Testing===")
    gracemodel.load_state_dict(torch.load(MODEL_PATH+"gracemodel_boosted"))
    clustermodel.load_state_dict(torch.load(MODEL_PATH+"clustermodel"))
    clustermodel.eval()
    gracemodel.eval()
    z = clustermodel(gracemodel(data.x, data.edge_index))
    if torch.cuda.is_available():
        y_pred = torch.argmax(z, dim=1).cpu().detach().numpy()
        y_true = data.y.cpu().numpy()
    else:
        y_pred = torch.argmax(z, dim=1).detach().numpy()
        y_true = data.y.numpy()
    y_pred = best_map(y_true,y_pred)
    acc = metrics.accuracy_score(y_true, y_pred)
    nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
    f1_macro = metrics.f1_score(y_true, y_pred, average='macro')
    print("Convergence Loss:", convergence_loss)
    print("Ac:", acc, "NMI:", nmi, "F1:", f1_macro)
    np.save(os.path.join('Communities', f'{dataset_name}-SEComm.npy'), y_pred)

    file_name = f'{dataset_name.lower()}_secomm_metrics.txt'
    file_path = os.path.join('Metrics', file_name)

    # Create the directory if it doesn't exist
    os.makedirs('Metrics', exist_ok=True)

    # Write the metrics to the file
    with open(file_path, 'w') as file:
        file.write(f"Convergence Loss: {convergence_loss}\n")
        file.write(f"Accuracy: {acc}\n")
        file.write(f"NMI: {nmi}\n")
        file.write(f"F1 Score (Macro): {f1_macro}\n")
    print(f"Metrics saved to '{file_path}'")
