import numpy as np
import time

def newton(A, b, x0, rl, lambd, st):
    _, d = A.shape
    x_stars = x0
    x_stars_arr, t = [], []
    x_stars_arr.append(x0.copy())
    t.append(0)
    start = time.time()
    while(True):
        if(time.time() - start >= st):
            break
        B = rl.hessian(A, b, x_stars) 
        H = (A.T * B) @ (A) + lambd * np.eye(d)
        g = rl.gradient(A,b,x_stars) + lambd * x_stars
        p = np.linalg.solve(H, -g)
        x_stars += p
        x_stars_arr.append(x_stars.copy())
        t.append(time.time() - start)
    
    end = time.time()
    print(f'Newton computation time: {end-start}')
    return x_stars_arr, t 