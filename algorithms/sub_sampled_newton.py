import numpy as np
import time

def row_norm_squares_sampling(A, b, x, rl):
    '''
    Block Norm Squares Sampling The first option is to construct a sampling distribution based on the
    magnitude of Ai. That is, define
    https://arxiv.org/pdf/1607.00559.pdf
    '''
    B = np.sqrt(rl.hessian(A, b, x))[:, np.newaxis] * A
    B_norm_2 = np.linalg.norm(B, 'fro') ** 2
    Bi_norm_2 = np.linalg.norm(B, axis=1) ** 2 # row-wise
    
    return Bi_norm_2/B_norm_2 # probability of selection for each i,...,n

def sub_sampled_newton_rnss(A, b, x0, rl, lambd, st, ss):
#     np.random.seed(42)
    n, d = A.shape
    x = x0
    x_arr, t = [], []
    x_arr.append(x0.copy())
    t.append(0)
    start = time.time()
    while(True):
        if(time.time() - start >= st):
            break
        prob = row_norm_squares_sampling(A,b,x,rl)
        q = np.minimum(ss * prob, 1)
        i = np.random.rand(n) < q # sub-sampled indices
        Ai = A[i,:] # features X
        bi = b[i] # y
        qi = q[i]
        Bi = rl.hessian(Ai, bi, x) # refer to Ai(w) in paper
        B_tilde = (np.sqrt(Bi)/np.sqrt(qi))[:,np.newaxis] * Ai
        H_tilde = B_tilde.T @ B_tilde + lambd * np.eye(d)
        
        g = rl.gradient(A,b,x) + lambd * x # g(w)
        p = np.linalg.solve(H_tilde, -g) # might use conjugate gd??
        x += p
        x_arr.append(x.copy())
        t.append(time.time() - start)
        
    end = time.time()
    print(f'Sub-sampled Newton rnss computation time: {end-start}')
    return x_arr, t