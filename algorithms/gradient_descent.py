import time

def gd(A, b, x0, rl, lambd, st, alpha):
    x = x0
    x_arr, t = [], []
    x_arr.append(x0.copy())
    t.append(0)
    start = time.time()
    while(True):
        if(time.time() - start >= st):
            break
  
        g = rl.gradient(A,b,x) + lambd * x
        x = x - alpha * g
        x_arr.append(x.copy())
        t.append(time.time() - start)
    
    end = time.time()
    print(f'GD computation time: {end-start}')
    return x_arr, t