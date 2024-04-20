import numpy as np

def compute_w_gaps(w, t, w_stars):
    w_gaps = []
    for i in range(len(t)):
        w_gaps_tmp = []
        for k in range(len(t[i])):
            w_gaps_tmp.append(np.linalg.norm(w[i][k] - w_stars))
        w_gaps.append(w_gaps_tmp)
    return w_gaps

def compute_f_gaps(A, b, w, t, w_stars, rl):
    f_gaps = []
    f_stars = rl.loss(A, b, w_stars)
    for i in range(len(t)):
        f_gaps_tmp = []
        for k in range(len(t[i])):
            f_gaps_tmp.append((rl.loss(A, b, w[i][k]) - f_stars))           
        f_gaps.append(f_gaps_tmp) 
    return f_gaps
