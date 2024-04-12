import time 

def accelerated_gd(A, b, x0, rl, lambd, iters):
    x_prev = x0
    x = x0.copy()
    x_arr, t = [], []
    x_arr.append(x0.copy())
    t.append(0)
    alpha = 0.5
    start = time.time()
    for j in range(2, iters+2):
#         beta = (j-2)/(j+1)
        beta = 0.72
        y = x + (beta * (x - x_prev))
        g = rl.gradient(A,b,y) + lambd * y
        x_prev = x
        x = y - alpha * g
        x_arr.append(x.copy())
        t.append(time.time() - start)
    
    end = time.time()
    print(f'AGD computation time: {end-start}')
    return x_arr, t