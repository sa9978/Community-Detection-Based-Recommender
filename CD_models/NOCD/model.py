import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import torch.distributions as td
from copy import deepcopy


import torch
import torch.nn.functional as F

def sparse_or_dense_dropout(x, p=0.5, training=True):
    if x.is_sparse:
        # For sparse tensors (CPU or GPU):
        new_values = F.dropout(x.values(), p=p, training=training)
        
        # Check if the tensor is on GPU or CPU and return accordingly
        if x.is_cuda:
            return torch.cuda.sparse.FloatTensor(x.indices(), new_values, x.size())
        else:
            return torch.sparse.FloatTensor(x.indices(), new_values, x.size())
    else:
        # For dense tensors, simply apply standard dropout to the entire tensor
        return F.dropout(x, p=p, training=training)
    

def to_sparse_tensor(matrix: Union[sp.spmatrix, torch.Tensor], cuda: bool = False) -> torch.sparse.FloatTensor:
    """Convert a scipy sparse matrix or PyTorch tensor to a sparse PyTorch tensor.
    
    Args:
        matrix: Sparse matrix or dense tensor to convert.
        cuda: Whether to move the resulting tensor to GPU. (default is False for CPU)
    
    Returns:
        sparse_tensor: Resulting sparse tensor (on CPU or on GPU).
    """
    if sp.issparse(matrix):
        coo = matrix.tocoo()
        indices = torch.LongTensor(np.vstack([coo.row, coo.col]))
        values = torch.FloatTensor(coo.data)
        shape = torch.Size(coo.shape)
        sparse_tensor = torch.sparse.FloatTensor(indices, values, shape)
    
    elif torch.is_tensor(matrix):
        row, col = matrix.nonzero().t()
        indices = torch.stack([row, col])
        values = matrix[row, col]
        shape = torch.Size(matrix.shape)
        sparse_tensor = torch.sparse.FloatTensor(indices, values, shape)
    
    else:
        raise ValueError(f"matrix must be scipy.sparse or torch.Tensor (got {type(matrix)} instead).")
    
    return sparse_tensor.coalesce() 

#------------------------------------------------------------------------------------------------
#---------------------------------- Convolutional Layers ----------------------------------------
class ImpGraphConvolution(nn.Module):
    """Improved Graph Convolution layer with two weight matrices for overlapping community detection.

    Args:
        in_features: Size of each input sample (number of node features).
        out_features: Size of each output sample (dimension of the output embeddings).
    
    Key Differences from Standard GCN:
        - In standard GCN, a single weight matrix is used to transform both the node's own features and its neighbors' features.
        - In ImpGraphConvolution, two separate weight matrices are used: 
            - `weight_own`: For transforming the node's own features.
            - `weight_nbr`: For transforming the aggregated features from the node's neighbors.
        - This separation provides more flexibility and is designed for tasks like overlapping community detection, where nodes might need different transformations for their own features and their neighbors' features.
        - Standard GCN aggregates features from both the node and its neighbors using the same transformation, while here, two separate transformations are performed before combining the results.
    """
    
    def __init__(self, in_features, out_features):
        super().__init__()
        # Initialize the input and output feature dimensions
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight matrix for the node's own feature transformation (self-features)
        self.weight_own = nn.Parameter(torch.empty(in_features, out_features))
        
        # Weight matrix for the neighboring node's feature transformation (neighbor-features)
        self.weight_nbr = nn.Parameter(torch.empty(in_features, out_features))
        
        # Bias term added to the output embeddings
        self.bias = nn.Parameter(torch.empty(out_features))
        
        # Initialize the weights and bias using the reset_parameters method
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize the parameters (weights and bias) using Xavier uniform initialization for better convergence."""
        # Xavier initialization for the weight matrix of self-features
        nn.init.xavier_uniform_(self.weight_own, gain=2.0)
        
        # Xavier initialization for the weight matrix of neighbor-features
        nn.init.xavier_uniform_(self.weight_nbr, gain=2.0)
        
        # Initialize the bias to zeros
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        """
        Perform the forward pass of the graph convolution.
        
        Args:
            x: Input feature matrix (shape: [num_nodes, in_features]).
            adj: Adjacency matrix (shape: [num_nodes, num_nodes]), which defines the graph structure.
        
        Returns:
            Updated node embeddings (shape: [num_nodes, out_features]).
        """
        # Transform the neighbor-features using the weight_nbr matrix and aggregate using the adjacency matrix
        neighbor_agg = adj @ (x @ self.weight_nbr)
        
        # Transform the self-features using the weight_own matrix
        self_agg = x @ self.weight_own
        
        # Combine the neighbor aggregation, self-transformation, and bias
        return neighbor_agg + self_agg + self.bias
    

class ImprovedGCN(nn.Module):
    """An improved GCN architecture for overlapping community detection.

    This architecture improves upon the standard GCN by:
    - Using two separate weight matrices for self-node features and neighbor aggregation.
    - Applying Tanh activation instead of ReLU, which stabilizes training for overlapping community detection tasks.
    - Offering optional layer normalization for additional stability.
    
    This version provides more stable training and faster convergence for overlapping
    community detection as compared to standard GCNs.
    """
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.5, layer_norm=False):
        """
        Initialize the ImprovedGCN model.

        Args:
            input_dim: Dimension of the input feature vectors (number of node features).
            hidden_dims: List of hidden dimensions for each layer.
            output_dim: Dimension of the output (usually the number of communities).
            dropout: Dropout probability to prevent overfitting (default is 0.5).
            layer_norm: Boolean flag to apply layer normalization after each layer.
        """
        super().__init__()
        self.dropout = dropout

        # Create the list of layers based on hidden and output dimensions
        layer_dims = np.concatenate([hidden_dims, [output_dim]]).astype(np.int32)

        # Add the first ImpGraphConvolution layer (input_dim -> first hidden layer)
        self.layers = nn.ModuleList([ImpGraphConvolution(input_dim, layer_dims[0])])

        # Add subsequent ImpGraphConvolution layers (hidden layers -> output)
        for idx in range(len(layer_dims) - 1):
            self.layers.append(ImpGraphConvolution(layer_dims[idx], layer_dims[idx + 1]))

        # If layer normalization is enabled, apply it after each hidden layer
        if layer_norm:
            self.layer_norm = [
                nn.LayerNorm([dim], elementwise_affine=False) for dim in hidden_dims
            ]
        else:
            self.layer_norm = None

    @staticmethod
    def normalize_adj(adj: sp.csr_matrix):
        """Normalize the adjacency matrix and convert it to a sparse tensor.
        
        This step is important for ensuring that the neighbor aggregation accounts for
        node degree, preventing highly connected nodes from dominating.
        
        Args:
            adj: The adjacency matrix, either in scipy's CSR format or as a PyTorch tensor.

        Returns:
            A normalized adjacency matrix in sparse tensor format.
        """
        if sp.isspmatrix(adj):
            # Set the diagonal to zero to remove self-loops
            adj = adj.tolil()
            adj.setdiag(0)
            adj = adj.tocsr()

            # Calculate the degree of each node
            deg = np.ravel(adj.sum(1))

            # Compute the inverse square root of the degree matrix
            deg_sqrt_inv = 1 / np.sqrt(deg)

            # Normalize the adjacency matrix: D^(-1/2) A D^(-1/2)
            adj_norm = adj.multiply(deg_sqrt_inv[:, None]).multiply(deg_sqrt_inv[None, :])
        
        elif torch.is_tensor(adj):
            # For adjacency matrix in PyTorch tensor format
            deg = adj.sum(1)
            deg_sqrt_inv = 1 / torch.sqrt(deg)

            # Normalize the adjacency matrix
            adj_norm = adj * deg_sqrt_inv[:, None] * deg_sqrt_inv[None, :]
        
        # Convert the normalized adjacency matrix to a sparse tensor
        return to_sparse_tensor(adj_norm)

    def forward(self, x, adj):
        """
        Forward pass through the Improved GCN model.

        Args:
            x: Input feature matrix of the nodes (shape: [num_nodes, in_features]).
            adj: Adjacency matrix that defines the graph structure.

        Returns:
            Updated node embeddings (shape: [num_nodes, out_features]).
        """
        # Loop through the layers of the GCN
        for idx, gcn in enumerate(self.layers):
            
            # Apply dropout to the input features if dropout is enabled
            if self.dropout != 0:
                x = sparse_or_dense_dropout(x, p=self.dropout, training=self.training)
            
            # Pass the input through the current graph convolution layer
            x = gcn(x, adj)

            # Apply Tanh activation after each hidden layer (but not the final layer)
            if idx != len(self.layers) - 1:
                x = torch.tanh(x)

                # Optionally apply layer normalization for stability
                if self.layer_norm is not None:
                    x = self.layer_norm[idx](x)

        # Return the final output embeddings after the last layer
        return x

    def get_weights(self):
        """Return the weight matrices of the model (excluding bias terms)."""
        return [w for n, w in self.named_parameters() if 'bias' not in n]

    def get_biases(self):
        """Return the bias vectors of the model."""
        return [w for n, w in self.named_parameters() if 'bias' in n]
    
#-------------------------------------------------------------------------------------------------
class GraphConvolution(nn.Module):
    """Graph convolution layer.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.

    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        return adj @ (x @ self.weight) + self.bias


class GCN(nn.Module):
    """Graph convolution network.

    References:
        "Semi-superivsed learning with graph convolutional networks",
        Kipf and Welling, ICLR 2017
    """
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.5, batch_norm=False):
        super().__init__()
        self.dropout = dropout
        layer_dims = np.concatenate([hidden_dims, [output_dim]]).astype(np.int32)
        self.layers = nn.ModuleList([GraphConvolution(input_dim, layer_dims[0])])
        for idx in range(len(layer_dims) - 1):
            self.layers.append(GraphConvolution(layer_dims[idx], layer_dims[idx + 1]))
        if batch_norm:
            self.batch_norm = [
                nn.BatchNorm1d(dim, affine=False, track_running_stats=False) for dim in hidden_dims
            ]
        else:
            self.batch_norm = None

    @staticmethod
    def normalize_adj(adj : sp.csr_matrix):
        """Normalize adjacency matrix and convert it to a sparse tensor."""
        if sp.isspmatrix(adj):
            adj = adj.tolil()
            adj.setdiag(1)
            adj = adj.tocsr()
            deg = np.ravel(adj.sum(1))
            deg_sqrt_inv = 1 / np.sqrt(deg)
            adj_norm = adj.multiply(deg_sqrt_inv[:, None]).multiply(deg_sqrt_inv[None, :])
        elif torch.is_tensor(adj):
            deg = adj.sum(1)
            deg_sqrt_inv = 1 / torch.sqrt(deg)
            adj_norm = adj * deg_sqrt_inv[:, None] * deg_sqrt_inv[None, :]
        return to_sparse_tensor(adj_norm)

    def forward(self, x, adj):
        for idx, gcn in enumerate(self.layers):
            if self.dropout != 0:
                x = sparse_or_dense_dropout(x, p=self.dropout, training=self.training)
            x = gcn(x, adj)
            if idx != len(self.layers) - 1:
                x = F.relu(x)
                if self.batch_norm is not None:
                    x = self.batch_norm[idx](x)
        return x

    def get_weights(self):
        """Return the weight matrices of the model."""
        return [w for n, w in self.named_parameters() if 'bias' not in n]

    def get_biases(self):
        """Return the bias vectors of the model."""
        return [w for n, w in self.named_parameters() if 'bias' in n]
    
#-----------------------------------------------------------------------------------------------------------
#------------------------------------- Reconstruction Loss -> Decoder --------------------------------------

class BernoulliDecoder(nn.Module):
    def __init__(self, num_nodes, num_edges, balance_loss=False):
        """Base class for Bernoulli decoder.

        Args:
            num_nodes: Number of nodes in a graph.
            num_edges: Number of edges in a graph.
            balance_loss: Whether to balance contribution from edges and non-edges.
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.num_possible_edges = num_nodes**2 - num_nodes
        self.num_nonedges = self.num_possible_edges - self.num_edges
        self.balance_loss = balance_loss


class BerpoDecoder(BernoulliDecoder):
    def __init__(self, num_nodes, num_edges, balance_loss=False):
        """Specific implementation of the Bernoulli-Poisson decoder.

        Args:
            num_nodes: Number of nodes in the graph.
            num_edges: Number of edges in the graph.
            balance_loss: Boolean flag to balance edge and non-edge loss terms.
        """
        super().__init__(num_nodes, num_edges, balance_loss)
        # Estimate edge probability based on the ratio of actual edges to possible edges.
        edge_proba = num_edges / (num_nodes**2 - num_nodes)
        # Epsilon is a scaling factor that prevents extreme probabilities (0 or 1).
        self.eps = -np.log(1 - edge_proba)

    def forward_batch(self, emb, idx):
        """Compute probabilities for a batch of edges.

        Args:
            emb: Node embeddings, shape (num_nodes, emb_dim).
            idx: Indices of the edges, shape (batch_size, 2).

        Returns:
            edge_probs: Bernoulli distribution for the edges, shape (batch_size).
        """
        # Extract the indices of the two nodes forming each edge.
        e1, e2 = idx.t()
        # Compute the dot product between the embeddings of the two nodes.
        logits = torch.sum(emb[e1] * emb[e2], dim=1)
        # Add the epsilon scaling factor to the logits.
        logits += self.eps
        # Convert logits to probabilities using the Bernoulli-Poisson formula.
        probs = 1 - torch.exp(-logits)
        # Return a Bernoulli distribution based on the computed probabilities.
        return td.Bernoulli(probs=probs)

    def forward_full(self, emb):
        """Compute probabilities for all possible edges in the graph.

        Args:
            emb: Node embeddings, shape (num_nodes, emb_dim).

        Returns:
            edge_probs: Bernoulli distribution for all edges, shape (num_nodes, num_nodes).
        """
        # Compute the dot product for every pair of node embeddings.
        logits = emb @ emb.t()
        # Add epsilon to logits to avoid extreme probabilities.
        logits += self.eps
        # Convert logits to probabilities.
        probs = 1 - torch.exp(-logits)
        # Return a Bernoulli distribution based on the computed probabilities.
        return td.Bernoulli(probs=probs)

    def loss_batch(self, emb, ones_idx, zeros_idx):
        """Compute the Bernoulli-Poisson loss for a batch of edges and non-edges.

        Args:
            emb: Node embeddings, shape (num_nodes, emb_dim).
            ones_idx: Indices of edges (positive examples), shape (num_edges, 2).
            zeros_idx: Indices of non-edges (negative examples), shape (num_non_edges, 2).

        Returns:
            loss: The computed loss for the batch.
        """
        # Positive (edge) loss: Compute the dot products for the edges.
        e1, e2 = ones_idx[:, 0], ones_idx[:, 1]
        edge_dots = torch.sum(emb[e1] * emb[e2], dim=1)
        # Compute the log-likelihood for the edges.
        loss_edges = -torch.mean(torch.log(-torch.expm1(-self.eps - edge_dots)))

        # Negative (non-edge) loss: Compute the dot products for the non-edges.
        ne1, ne2 = zeros_idx[:, 0], zeros_idx[:, 1]
        loss_nonedges = torch.mean(torch.sum(emb[ne1] * emb[ne2], dim=1))

        # Balance the loss if necessary (to account for imbalance between edges and non-edges).
        if self.balance_loss:
            neg_scale = 1.0
        else:
            # Scale the non-edge loss by the ratio of non-edges to edges.
            neg_scale = self.num_nonedges / self.num_edges
        
        # Return the weighted sum of edge and non-edge losses.
        return (loss_edges + neg_scale * loss_nonedges) / (1 + neg_scale)

    def loss_full(self, emb, adj):
        """Compute the Bernoulli-Poisson loss for all edges and non-edges in the graph.

        Args:
            emb: Node embeddings, shape (num_nodes, emb_dim).
            adj: Adjacency matrix (sparse) indicating edges.

        Returns:
            loss: The computed loss for the entire graph.
        """
        # Positive (edge) loss: Find the indices of all edges.
        e1, e2 = adj.nonzero()
        # Compute the dot products for the edges.
        edge_dots = torch.sum(emb[e1] * emb[e2], dim=1)
        # Compute the log-likelihood for all edges.
        loss_edges = -torch.sum(torch.log(-torch.expm1(-self.eps - edge_dots)))

        # Non-edge correction: Avoid overcounting embeddings with themselves or existing edges.
        self_dots_sum = torch.sum(emb * emb)
        correction = self_dots_sum + torch.sum(edge_dots)
        # Compute the sum of embeddings for all nodes.
        sum_emb = torch.sum(emb, dim=0, keepdim=True).t()
        # Compute the non-edge loss by subtracting the correction from the total embedding sum.
        loss_nonedges = torch.sum(emb @ sum_emb) - correction

        # Balance the loss if necessary (for imbalance between edges and non-edges).
        if self.balance_loss:
            neg_scale = 1.0
        else:
            # Scale the non-edge loss by the ratio of non-edges to edges.
            neg_scale = self.num_nonedges / self.num_edges

        # Return the weighted sum of edge and non-edge losses.
        return (loss_edges / self.num_edges + neg_scale * loss_nonedges / self.num_nonedges) / (1 + neg_scale)
    
#-----------------------------------------------------------------------------------------------------------
#------------------------------------------------- Stopping ------------------------------------------------
class NoImprovementStopping():
    """Stop training when the validation metric stops improving.

    Parameters
    ----------
    validation_fn : function
        Calling this function returns the current value of the validation metric.
    mode : {'min', 'max'}
        Should the validation metric be minimized or maximized?
    patience : int
        Number of iterations without improvement before stopping.
    tolerance : float
        Minimal improvement in validation metric to not trigger patience.
    relative : bool
        Is tolerance measured in absolute units or relatively?

    Attributes
    ----------
    _best_value : float
        Best value of the validation metric seen so far.
    _num_bad_epochs : int
        Number of epochs since the last significant improvement in the validation metric.
    _time_to_save : bool
        Whether it's time to save the model weights after improvement.
    _is_better : function
        A comparison function that determines if the new validation metric is better than the previous best one.
        Signature: self._is_better(new_value, best_value).
    """
    def __init__(self, validation_fn, mode='min', patience=10, tolerance=0.0, relative=False):
        """Initialize the NoImprovementStopping class.

        Args:
            validation_fn: A function that returns the current value of the validation metric (e.g., loss, accuracy).
            mode: 'min' for minimizing the metric (e.g., loss), 'max' for maximizing the metric (e.g., accuracy).
            patience: Number of epochs to wait without improvement before stopping.
            tolerance: The minimum change in the validation metric to reset the patience counter.
            relative: If True, tolerance is measured relative to the current best value; otherwise, it’s absolute.
        """
        super().__init__()  
        self.validation_fn = validation_fn  # Store the validation function.
        self.mode = mode  # 'min' or 'max' mode for metric comparison.
        self.patience = patience  # Number of epochs to wait without improvement.
        self.tolerance = tolerance  # Minimum required improvement.
        self.relative = relative  # If True, tolerance is relative; otherwise, it's absolute.
        self.reset()  # Initialize/reset internal state.

        if mode not in ['min', 'max']:
            raise ValueError(f"Mode should be either 'min' or 'max' (got {mode} instead).")

        # Define the comparison function for metric improvement.
        if relative:
            # If tolerance is relative, calculate the improvement relative to the current best value.
            if mode == 'min':
                # Minimization mode: Improvement means new value is smaller by a relative margin.
                self._is_better = lambda new, best: new < best - (best * tolerance)
            elif mode == 'max':
                # Maximization mode: Improvement means new value is larger by a relative margin.
                self._is_better = lambda new, best: new > best + (best * tolerance)
        else:
            # If tolerance is absolute, calculate the improvement based on a fixed tolerance.
            if mode == 'min':
                # Minimization mode: Improvement means new value is smaller by an absolute margin.
                self._is_better = lambda new, best: new < best - tolerance
            elif mode == 'max':
                # Maximization mode: Improvement means new value is larger by an absolute margin.
                self._is_better = lambda new, best: new > best + tolerance

    def reset(self):
        """Reset the internal state of the stopping mechanism."""
        # Initialize the best validation metric with the current value.
        self._best_value = self.validation_fn()
        # Reset the counter for bad (non-improving) epochs.
        self._num_bad_epochs = 0
        # Set flag to indicate that the model doesn't need to be saved immediately.
        self._time_to_save = False

    def next_step(self):
        """Check the validation metric at the current iteration and update the state.

        This should be called after each epoch to evaluate the validation metric.
        """
        # Get the latest validation metric value.
        last_value = self.validation_fn()

        # If the new value is better (according to the _is_better function), update the best value.
        if self._is_better(last_value, self._best_value):
            self._time_to_save = True  # Set flag to indicate that the model should be saved.
            self._best_value = last_value  # Update the best value with the current value.
            self._num_bad_epochs = 0  # Reset the bad epochs counter.
        else:
            # If no improvement, increment the number of bad epochs.
            self._num_bad_epochs += 1

    def should_save(self):
        """Check if it's time to save the model weights.

        Returns:
            bool: True if the model should be saved, False otherwise.
        """
        if self._time_to_save:
            # Reset the save flag and return True to indicate that it's time to save.
            self._time_to_save = False
            return True
        else:
            # No need to save if there was no improvement.
            return False

    def should_stop(self):
        """Check if it's time to stop training.

        Returns:
            bool: True if the number of bad epochs has exceeded the patience limit, False otherwise.
        """
        # If the number of bad epochs exceeds the patience limit, return True to stop training.
        return self._num_bad_epochs > self.patience
#-----------------------------------------------------------------------------------------------------------
#------------------------------------------------- Sampler -------------------------------------------------
import torch.utils.data as data_utils


class EdgeSampler(data_utils.Dataset):
    """Sample edges and non-edges uniformly from a graph.

    Args:
        A: adjacency matrix.
        num_pos: number of edges per batch.
        num_neg: number of non-edges per batch.
    """
    def __init__(self, A, num_pos=1000, num_neg=1000):
        self.num_pos = num_pos
        self.num_neg = num_neg
        self.A = A
        self.edges = np.transpose(A.nonzero())
        self.num_nodes = A.shape[0]
        self.num_edges = self.edges.shape[0]
    def __getitem__(self, key):
        np.random.seed(key)
        edges_idx = np.random.randint(0, self.num_edges, size=self.num_pos, dtype=np.int64)
        next_edges = self.edges[edges_idx, :]

        # print(f"Generated {len(next_edges)} edges in the sampler.")

        # Select num_neg non-edges
        generated = False
        while not generated:
            candidate_ne = np.random.randint(0, self.num_nodes, size=(2*self.num_neg, 2), dtype=np.int64)
            cne1, cne2 = candidate_ne[:, 0], candidate_ne[:, 1]
            to_keep = (1 - self.A[cne1, cne2]).astype(np.bool_).A1 * (cne1 != cne2)
            next_nonedges = candidate_ne[to_keep][:self.num_neg]
            generated = to_keep.sum() >= self.num_neg

        # print(f"Generated {len(next_nonedges)} non-edges in the sampler.")
        
        return torch.LongTensor(next_edges), torch.LongTensor(next_nonedges)

    def __len__(self):
        return 2**32

def collate_fn(batch):
    edges, nonedges = batch[0]
    return (edges, nonedges)

def get_edge_sampler(A, num_pos=1000, num_neg=1000, num_workers=2):
    data_source = EdgeSampler(A, num_pos, num_neg)
    return data_utils.DataLoader(data_source, num_workers=num_workers, collate_fn=collate_fn)
#-----------------------------------------------------------------------------------------------------------
#---------------------------------------------Model Saver---------------------------------------------------
class ModelSaver:
    """In-memory saver for model parameters.

    Storing weights in memory is faster than saving to disk with torch.save.
    """
    def __init__(self, model):
        self.model = model

    def save(self):
        self.state_dict = deepcopy(self.model.state_dict())

    def restore(self):
        self.model.load_state_dict(self.state_dict)


def l2_reg_loss(model, scale=1e-5):
    """Get L2 loss for model weights."""
    loss = 0.0
    for w in model.get_weights():
        loss += w.pow(2.).sum()
    return loss * scale




