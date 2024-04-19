import numpy as np

def woodbury(B_tilde, alpha, gradient):
    ft = (1/alpha) * gradient
    inv = np.linalg.inv((B_tilde@B_tilde.T) + alpha * np.eye(B_tilde.shape[0]))
    st = B_tilde.T @ inv @ B_tilde @ gradient/alpha
    return ft - st
