import numpy as np

def reshape_row_dim(data):
    target_shape = (data.shape[0], data.shape[-1])
    return data.reshape(target_shape)
