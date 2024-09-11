import numpy as np
from pathlib import Path
from utils import param_grid, read_pickle_dump
from dataGen import DataGenerator
from dataReader import DataReader 
from dmdBase import DmdBase
from plotters import plot_3D
from dmdVisualiser import DmdVisualiser

def generator_toy_fn(x_range, t_range, *args, **kwargs):
    x1, x2 = x_range
    t1, t2 = t_range
    n, m = kwargs["n"], kwargs["m"]
    X = np.linspace(x1, x2, num=n)    # say some param x
    T = np.linspace(t1, t2, num=m)    # say some param t
    X_grid, T_grid =  np.meshgrid(X, T)
    f1 = 0.5*(np.cos(X_grid)) #*(1.0 + 0*T_grid)   # time independent
    f2 = ((1/ np.cosh(X_grid))*(np.tanh(X_grid)))*(2*np.exp(1j*2.8*T_grid))
    print(f"f1 shape: {f1.shape}, f2 shape: {f2.shape}, f1 + f2 shape: {(f1 + f2).shape}")
    return f1 + f2

def generator_twin_mode_toy(x_range, t_range, *args, **kwargs):
    x1, x2 = x_range
    t1, t2 = t_range
    n, m = kwargs["n"], kwargs["m"]
    X = np.linspace(x1, x2, num=n)    # say some param x
    T = np.linspace(t1, t2, num=m)    # say some param t
    X_grid, T_grid =  np.meshgrid(X, T)
    f1 = (1/ np.cosh(X_grid + 3)) *(np.exp(1j*2.3*T_grid))   # time independent
    f2 = ((1/ np.cosh(X_grid))*(np.tanh(X_grid)))*(2*np.exp(1j*2.8*T_grid))
    print(f"f1 shape: {f1.shape}, f2 shape: {f2.shape}, f1 + f2 shape: {(f1 + f2).shape}")
    return f1 + f2

# Sys Args
RUN = 2
DATA_FOLDER =  Path(__file__).parent.parent/ "data"
RESULTS_FOLDER= Path(__file__).parent.parent/ "results"
DUMP_FILE = f"generated_data_{RUN}.pkl"

# LOW_RANK Calc
R = 20

# Parameters
x_range = (-15, 15)
t_min, t_max = 0, 8*np.pi
t_range = (t_min,t_max)
f_range = (-2, 2)
n, m = 200, 80
X, t = param_grid(x_range = x_range, t_range = t_range, n = n, m = m)
dt = t_max / m

"""# Parameters
x_range = (-10, 10)
t_min, t_max = 0, 4*np.pi
t_range = (t_min,t_max)
f_range = (-2, 2)
n, m = 400, 200
X, t = param_grid(x_range = x_range, t_range = t_range, n = n, m = m)
dt = t_max / m"""


# generatring data
generator = DataGenerator()
generator.generate_toy_data(generator_fn= generator_toy_fn, x_range= x_range, t_range= t_range, n=n, m=m)
print(generator.data.shape)
plot_3D(X, t, generator.data.real, x_range=list(x_range), y_range=list(t_range), z_range=list(f_range))
generator.save_data(filename=DUMP_FILE, folder=None)   # writing data

# Reading generated data
toy_data = DataReader(filename=DUMP_FILE, folder=None, reader=read_pickle_dump)
readData = toy_data.read()  # reading
print(readData.shape)
plot_3D(X, t, readData.real, list(x_range), list(t_range), list(f_range))

# dmd setup
dmd = DmdBase(dataObject= toy_data)
dmd.update_state(data_read = False, columns_align_temporal= False, decomposed= False, modes_calculated= False)
dmd.prepare_data(dmd_type_temporal=True)
# dmd.transpose_data()  # data read transposed, stacking timeseries along columns
print(f"shape of data in DMD object: {dmd.data.shape}")

# running dmd algo
dmd.decompose()
dmd.calc_low_rank_eigvecs_modes(r= R)
dmd.spectra(dt)
dmd.modes.dump(RESULTS_FOLDER/ f"modes_gen_{RUN}.pkl")
dmd.eigvals.dump(RESULTS_FOLDER/ f"eigvals_gen_{RUN}.pkl")
dmd.omega.dump(RESULTS_FOLDER/ f"omega_gen_{RUN}.pkl")
# print(f"printing omaga: {dmd.omega}")
# print(f"absolute values of omega: {np.abs(dmd.omega)}")
# print(f"omage values: \n {np.where(np.abs(dmd.omega) < 1)}")

# visualizing results
viz = DmdVisualiser(dmd)
viz.visualise_singular_vals()
viz.visualise_eigs()
viz.visualise_tSpace_eigs(xlim=[-16, 2])
viz.visualise_mode([0, 1, 3])
