import numpy as np

def conjugate_gd(A, b, x, tol):
    rk = A @ x - b
    pk = -rk
    xk = x.copy()
    k = 0
    while(np.linalg.norm(rk) > tol):
        alpha_k = (rk.T @ rk) / (pk.T @ A @ pk)# 4
        
        xk = xk + alpha_k * pk # 5
        rk_1 = rk + alpha_k * A @ pk # 5
        
        beta_k_1 = (rk_1.T @ rk_1) / (rk.T @ rk)
        pk = -rk_1 + beta_k_1 * pk
        
        rk = rk_1        
        k = k + 1
        
    return xk
     