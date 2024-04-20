import numpy as np
import time
from algorithms.util import *
from solvers.conjugate import *
from solvers.woodbury import *

def row_norm_squares_sampling(A, b, x, rl):
    '''
    Block Norm Squares Sampling The first option is to construct a sampling distribution based on the
    magnitude of Ai. That is, define
    https://arxiv.org/pdf/1607.00559.pdf
    '''
    sqrt_B = np.sqrt(rl.hessian(A, b, x))[:, np.newaxis]
    # fixed add on dimension for low-dimensional cases
    sqrt_B = reshape_row_dim(sqrt_B)
    B = sqrt_B * A
    B_norm_2 = np.linalg.norm(B, 'fro') ** 2
    Bi_norm_2 = np.linalg.norm(B, axis=1) ** 2 # row-wise
    
    return Bi_norm_2/B_norm_2 # probability of selection for each i,...,n


def nesterov_sub_sampled_newton_rnss(A, b, x0, rl, lambd, alpha, beta, st, ss):
#     np.random.seed(42)
    n, d = A.shape
    x_prev = x0
    x = x0.copy()
    x_arr, t = [], []
    x_arr.append(x0.copy())
    t.append(0)
    start = time.time()
    j = 2
    while(True):
        if (time.time() - start >= st):
            break
        
        prob = row_norm_squares_sampling(A,b,x,rl)
        q = np.minimum(ss * prob, 1)
        i = np.random.rand(n) < q # sub-sampled indices
        Ai = A[i,:]
        bi = b[i] 
        qi = q[i]
        Bi = rl.hessian(Ai, bi, x) # refer to Ai(w) in paper
        
        frac_B_tilde = (np.sqrt(Bi)/np.sqrt(qi))[:,np.newaxis]
        # fixed add on dimension for low-dimensional cases
        frac_B_tilde = reshape_row_dim(frac_B_tilde)
        B_tilde = frac_B_tilde * Ai
        H_tilde = B_tilde.T @ B_tilde + lambd * np.eye(d)
    
#         if beta == -1: beta = (j-2)/(j+1)
        if beta == -1: beta = ((j-2)/(j+1))

        y = x + (beta * (x - x_prev))
        g = rl.gradient(A,b,y) + lambd * y # g(w)
        x_prev = x
        
#         p = np.linalg.solve(H_tilde, -g) # might use conjugate gd??
#         p = woodbury(B_tilde, lambd, g)
        p = conjugate_gd(H_tilde, g, x, 0.1*np.linalg.norm(g))
        x = y - (alpha * p)
        x_arr.append(x.copy())
        t.append(time.time() - start)
        
        j += 1
        
    end = time.time()
    print(f'Nesterov Sub-sampled Newton rnss computation time: {end-start}')
    return x_arr, t