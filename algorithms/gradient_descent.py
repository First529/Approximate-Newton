import time

def gd(A, b, x0, rl, lambd, iters):
    x = x0
    x_arr, t = [], []
    x_arr.append(x0.copy())
    t.append(0)
    alpha = 0.5
    start = time.time()
    for j in range(iters):
        g = rl.gradient(A,b,x) + lambd * x
        x = x - alpha * g
        x_arr.append(x.copy())
        t.append(time.time() - start)
    
    end = time.time()
    print(f'GD computation time: {end-start}')
    return x_arr, t