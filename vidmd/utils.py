import numpy as np
import pickle

def read_pickle_dump(filepath):
    "reads from a pickle dump"
    print(f"Reading data from filepath: {filepath}")
    data = []
    with open(filepath, "rb") as f:
        while True:
            try:
                data.append(pickle.load(f))
            except EOFError:
                break
    # return np.array(np.squeeze(data, axis = 0), dtype=np.uint8) # data has extra dimention so squeezed
    return np.array(np.squeeze(data, axis=0))

def param_grid(x_range, t_range, *args, **kwargs):
    x1, x2 = x_range
    t1, t2 = t_range
    n, m = kwargs["n"], kwargs["m"]
    X = np.linspace(x1, x2, num=n)    # say some param x
    T = np.linspace(t1, t2, num=m)    # say some param t
    X_grid, T_grid =  np.meshgrid(X, T)
    return X_grid, T_grid

def setup_toy_grid(n = 2000, m = 400):
    x_range = (-10, 10)
    t_range = (0, 4*np.pi)
    n, m = n, m
    X, t = param_grid(x_range = x_range, t_range = t_range, n = n, m = m)
    return X, t