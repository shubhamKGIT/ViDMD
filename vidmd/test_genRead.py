from dataGen import DataGenerator
from plotters import plot_3D
import numpy as np
from pathlib import Path

from dataReader import DataReader
from dmdBase import DmdBase
from utils import read_pickle_dump
from dmdVisualiser import Visualiser

def param_grid(x_range, t_range, *args, **kwargs):
    x1, x2 = x_range
    t1, t2 = t_range
    n, m = kwargs["n"], kwargs["m"]
    X = np.linspace(x1, x2, num=n)    # say some param x
    T = np.linspace(t1, t2, num=m)    # say some param t
    X_grid, T_grid =  np.meshgrid(X, T)
    return X_grid, T_grid

x_range = (-10, 10)
t_range = (0, 4*np.pi)
f_range = (-2, 2)
n, m = 2000, 400
X, t = param_grid(x_range = x_range, t_range = t_range, n = n, m = m)

datagen = DataGenerator()
print(datagen)
DUMP_FILE = "generated.pkl"
dataFolder =  Path(__file__).parent.parent/ "data"
filepath = dataFolder / DUMP_FILE
data = datagen.read_pickle_dump(filepath=filepath, overwrite_data_attrib=True)
print(data.shape)

# plot_3D(X, t, data.real, list(x_range), list(t_range), list(f_range))

dataObj = DataReader(filename=DUMP_FILE, folder=None, reader=read_pickle_dump)
readData = dataObj.read()
print(readData.shape)

plot_3D(X, t, readData.real, list(x_range), list(t_range), list(f_range))

dmd = DmdBase(dataObject= dataObj)
dmd.prepare_data()
print(f"shape of data in DMD object: {dmd.data.shape}")
# dmd.transpose_data()
print(f"shape of transposed data in DMD object for calculating spatial modes: {dmd.data.shape}")
dmd.decompose(r = 20)
# dmd.modes.dump(dataFolder/ "modes.pkl")

viz = Visualiser(dmd)
viz.visualise_eigs()