# Community Detection in Graphs with Neural Networks and Its Application in Recommender Systems

This repository contains the official codebase and experimental framework for my B.Sc. thesis, titled **"Community Detection in Graphs with Neural Networks and its Application in Recommender Systems"**, conducted at the **Department of Computer Engineering, Amirkabir University of Technology (Tehran Polytechnic)**.

## Overview & Architecture

The project investigates whether integrating unsupervised graph community detection and community-aware feature aggregation into graph neural networks can improve node-to-node collaborative recommendations on attributed graphs. 

The framework unifies unsupervised community discovery, structural/neighborhood feature aggregation, and collaborative ranking into a three-stage pipeline:

```
Input Attributed Graph G = (V, E, X)
                 │
                 ▼
┌───────────────────────────────────────────────────────────┐
│ 1. Community Detection Module                             │
│    • SEComm (Self-Expressive GNN + InfoNCE Contrastive)   │
│    • NOCD (Bernoulli-Poisson Overlapping GNN)             │
│    • VGAER (Modularity-Reconstructed Variational GAE)     │
└───────────────────────────────────────────────────────────┘
                 │ Community Memberships / Assignments (C)
                 ▼
┌───────────────────────────────────────────────────────────┐
│ 2. Neighborhood-Community Aggregator                      │
│    • 1D-CNN over community-level feature matrices         │
│    • Adaptive Max-Pooling & ReLU activation               │
│    • 2-Layer Multi-Layer Perceptron (MLP)                 │
│    • Feature Concatenation: [Original Node || Aggregated] │
└───────────────────────────────────────────────────────────┘
                 │ Enriched Node Embeddings (H')
                 ▼
┌───────────────────────────────────────────────────────────┐
│ 3. Collaborative Recommendation Layer                     │
│    • CD-based NGCF (Neural Graph Collaborative Filtering) │
│    • Pairwise BPR Loss with Community Hard Negatives      │
│    • Comparative Baselines: LightGCN, PinSAGE, GraphSAGE  │
└───────────────────────────────────────────────────────────┘
```

---

## Methodology Details

### 1. Community Detection Module

The system extracts macro-level relational structures by grouping nodes that exhibit dense internal connectivity:

* **SEComm (Self-Expressive Community Detection - Primary Model):**  
  Combines graph corruptions with self-supervised contrastive learning (InfoNCE loss) to learn robust node representations $Z$. It enforces self-expressiveness where each node's representation is reconstructed as a sparse linear combination of other nodes ($Z = ZC$). The learned coefficient matrix $C$ guides a multi-layer network with soft membership constraints to discover balanced, high-cohesion communities.
* **NOCD (Neural Overlapping Community Detection - Complementary Baseline):**  
  Combines graph convolutional encoders with a Bernoulli-Poisson probabilistic model to detect overlapping community memberships by minimizing edge reconstruction loss.
* **VGAER (Variational Graph Autoencoder based Reconstruction):**  
  Employs a variational graph autoencoder tailored to reconstruct modularity matrices rather than adjacency matrices, preserving intra-community density.

### 2. Neighborhood-Community Aggregator

To bridge detected community assignments and downstream collaborative filtering, a custom aggregation pipeline fuses micro-level node attributes with macro-level community characteristics:

1. **1D-CNN Layer:** Processes community-level feature representations through 1D convolutions to extract local dependencies and feature interactions within community members.
2. **Adaptive Max-Pooling & Non-Linearity:** Compresses dimension size while retaining the most salient topological features, followed by ReLU activation to introduce non-linear modeling capacity.
3. **MLP Dimensionality Refinement:** A two-layer fully connected network projects aggregated representations into a compact latent space.
4. **Feature Fusion:** The resulting community representations are concatenated with the original node features, producing enriched node embeddings ($H'$) that preserve both individual attributes and structural group context.

### 3. Community-Aware Recommendation (CD-based NGCF)

The enriched embeddings are used to initialize node representations in **Neural Graph Collaborative Filtering (NGCF)**:

* **High-Order Propagation:** Message-passing layers recursively propagate latent features across user-item interaction paths, capturing collaborative signal interactions across multi-hop neighborhoods.
* **Pairwise Optimization (BPR Loss):** Model parameters are optimized using Bayesian Personalized Ranking (BPR) loss:
  $$\mathcal{L}_{\text{BPR}} = -\sum_{(u, i, j) \in \mathcal{D}} \ln \sigma(\hat{y}_{ui} - \hat{y}_{uj}) + \lambda \|\Theta\|_2^2$$
* **Community-Guided Negative Sampling:** Negative items ($j$) are dynamically sampled by computing feature distances to neighboring communities, prioritizing hard negative instances to enhance ranking discrimination and recommendation explainability.
* **Architectural Baselines:** Includes standard graph collaborative filtering implementations for comparative analysis, including **LightGCN** (linearized neighborhood summation), **PinSAGE** (importance-based random walk convolutions), and **lightweight GraphSAGE**.

---

## Repository Structure

```text
├── CD_models/             # Community detection implementations
│   ├── NOCD/              # Bernoulli-Poisson neural overlapping community detection
│   ├── SEComm/            # Self-expressive GNN implementation and config.yaml
│   └── VGAER/             # Variational graph autoencoder with modularity reconstruction
├── data_loader.py         # PyTorch Geometric dataset loader (Cora, Citeseer, Pubmed, etc.)
├── aggregator.py          # 1D-CNN + MLP neighborhood-community feature aggregator
├── recommendation.py      # PinSAGE-style recommendation pipeline
├── recommendation_lighter.py # Lightweight GraphSAGE baseline
├── ngcf.py                # Standard Neural Graph Collaborative Filtering baseline
├── cd_ngcf.py             # Community-Aware NGCF pipeline (CD-based NGCF)
├── main.py                # Command-line entry point for community detection models
└── requirements.txt       # Python package dependencies
```

---

## Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Community-Detection-Recommender.git
   cd Community-Detection-Recommender
   ```

2. Set up environment and install PyTorch with appropriate CUDA support:
   ```bash
   pip install torch torchvision torchaudio
   pip install torch_geometric
   pip install -r requirements.txt
   ```

---

## Execution Guide

### Step 1: Run Community Detection
Execute the unsupervised community detection module (e.g., SEComm on Cora):
```bash
python main.py --method secomm --dataset cora
```
* Supported methods: `secomm`, `nocd`, `vgaer`
* Supported datasets: `cora`, `citeseer`, `pubmed`, `lastfmasia`, `amazon_photos`, `amazon_computers`

Community assignments are exported to `Communities/`.

### Step 2: Extract Community-Aware Aggregated Features
Process detected community assignments through the 1D-CNN and MLP aggregation pipeline:
```bash
python aggregator.py
```
This builds and stores enriched embeddings combining individual node features and community-level priors.

### Step 3: Run Recommender Systems
Train the community-enhanced collaborative filtering model:
```bash
python cd_ngcf.py
```

To run baseline architectures for comparison:
* Standard NGCF: `python ngcf.py`
* PinSAGE: `python recommendation.py`
* Lightweight GraphSAGE: `python recommendation_lighter.py`

---

## Hyperparameter Optimization

Hyperparameters for community detection and ranking models (learning rates, layer sizes, regularizations, dropout rates, and InfoNCE temperature) are tuned using **Optuna** with Tree-Structured Parzen Estimators (TPE) across training iterations.

---
## References & Codebases

This project integrates and builds upon the official implementations of the following models:

* **SEComm:** [sambaranban/SEComm](https://github.com/sambaranban/SEComm)
* **NOCD:** [shchur/overlapping-community-detection](https://github.com/shchur/overlapping-community-detection)
* **VGAER:** [ChenyangQiu/VGAER](https://github.com/ChenyangQiu/VGAER)
* **NGCF:** [xiangwang1223/neural_graph_collaborative_filtering](https://github.com/xiangwang1223/neural_graph_collaborative_filtering)
* **LightGCN:** [gusye1234/LightGCN-PyTorch](https://github.com/gusye1234/LightGCN-PyTorch)
* **PinSAGE:** [dmlc/dgl (PinSAGE implementation)](https://github.com/dmlc/dgl/tree/master/examples/pytorch/pinsage)
---

## Citation

```bibtex
@thesis{rouhani2024community,
  title     = {Community Detection in Graphs with Neural Networks and its Application in Recommender Systems},
  author    = {Rouhani, Sara},
  school    = {Department of Computer Engineering, Amirkabir University of Technology (Tehran Polytechnic)},
  year      = {2024},
  type      = {Bachelor's Thesis}
}
```