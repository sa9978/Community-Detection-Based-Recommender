import torch.nn.functional as F
import warnings
import numpy as np
import torch


def overlapping_nmi(X, Y):
    """Compute NMI between two overlapping community covers.

    Parameters
    ----------
    X : array-like, shape [N, m]
        Matrix with samples stored as columns.
    Y : array-like, shape [N, n]
        Matrix with samples stored as columns.

    Returns
    -------
    nmi : float
        Float in [0, 1] quantifying the agreement between the two partitions.
        Higher is better.
    """
    
    # Ensure X and Y are NumPy arrays
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    if isinstance(Y, torch.Tensor):
        Y = Y.cpu().numpy()

    if not ((X == 0) | (X == 1)).all():
        raise ValueError("X should be a binary matrix")
    if not ((Y == 0) | (Y == 1)).all():
        raise ValueError("Y should be a binary matrix")

    if X.shape[1] > X.shape[0] or Y.shape[1] > Y.shape[0]:
        warnings.warn("It seems that you forgot to transpose the F matrix")
    X = X.T
    Y = Y.T
    
    def cmp(x, y):
        """Compare two binary vectors."""
        a = (1 - x).dot(1 - y)
        d = x.dot(y)
        c = (1 - y).dot(x)
        b = (1 - x).dot(y)
        return a, b, c, d

    def h(w, n):
        """Compute contribution of a single term to the entropy."""
        if w == 0:
            return 0
        else:
            return -w * np.log2(w / n)

    def H(x, y):
        """Compute conditional entropy between two vectors."""
        a, b, c, d = cmp(x, y)
        n = len(x)
        if h(a, n) + h(d, n) >= h(b, n) + h(c, n):
            return h(a, n) + h(b, n) + h(c, n) + h(d, n) - h(b + d, n) - h(a + c, n)
        else:
            return h(c + d, n) + h(a + b, n)

    def H_uncond(X):
        """Compute unconditional entropy of a single binary matrix."""
        return sum(h(x.sum(), len(x)) + h(len(x) - x.sum(), len(x)) for x in X)

    def H_cond(X, Y):
        """Compute conditional entropy between two binary matrices."""
        m, n = X.shape[0], Y.shape[0]
        scores = np.zeros([m, n])
        for i in range(m):
            for j in range(n):
                scores[i, j] = H(X[i], Y[j])
        return scores.min(axis=1).sum()

    if X.shape[1] != Y.shape[1]:
        raise ValueError("Dimensions of X and Y don't match. (Samples must be stored as COLUMNS)")
    H_X = H_uncond(X)
    H_Y = H_uncond(Y)
    I_XY = 0.5 * (H_X + H_Y - H_cond(X, Y) - H_cond(Y, X))
    return I_XY / max(H_X, H_Y)



def get_nmi(gnn, x_norm, adj_norm, Z_gt, thresh=0.5):
    """Compute Overlapping NMI of the communities predicted by the GNN."""
    gnn.eval()  # Set GNN in evaluation mode
    Z = F.relu(gnn(x_norm, adj_norm))  # Get the output of the GNN
    Z_pred = Z.cpu().detach().numpy() > thresh  # Threshold the output to get predictions
    Z_gt = Z_gt.cpu().numpy()  # Ensure Z_gt is a NumPy array
    nmi = overlapping_nmi(Z_pred, Z_gt)  # Compute overlapping NMI
    return nmi