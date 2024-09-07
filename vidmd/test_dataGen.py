from dataGen import DataGenerator
import numpy as np
from plotters import plot_3D

def generator_toy_fn(x_range, t_range, *args, **kwargs):
    x1, x2 = x_range
    t1, t2 = t_range
    n, m = kwargs["n"], kwargs["m"]
    X = np.linspace(x1, x2, num=n)    # say some param x
    T = np.linspace(t1, t2, num=m)    # say some param t
    X_grid, T_grid =  np.meshgrid(X, T)
    f1 = 0.5*(1/ np.cosh(X_grid+1))*(1.0*np.exp(1j*2.3*T_grid))   # time independent
    f2 = ((1/ np.cosh(X_grid))*(np.tanh(X_grid)))*(2*np.exp(1j*2.8*T_grid))
    return f1 + f2

def param_grid(x_range, t_range, *args, **kwargs):
    x1, x2 = x_range
    t1, t2 = t_range
    n, m = kwargs["n"], kwargs["m"]
    X = np.linspace(x1, x2, num=n)    # say some param x
    T = np.linspace(t1, t2, num=m)    # say some param t
    X_grid, T_grid =  np.meshgrid(X, T)
    return X_grid, T_grid

generator = DataGenerator()
print(generator)
x_range = (-10, 10)
t_range = (0, 4*np.pi)
f_range = (-2, 2)
n, m = 2000, 400
X, t = param_grid(x_range = x_range, t_range = t_range, n = n, m = m)
generator.generate_toy_data(generator_fn= generator_toy_fn, x_range= x_range, t_range= t_range, n=n, m=m)
print(generator.data.shape)
plot_3D(X, t, generator.data.real, x_range=list(x_range), y_range=list(t_range), z_range=list(f_range))
DUMP_FILE = "generated.pkl"
generator.save_data(filename=DUMP_FILE, folder=None)

