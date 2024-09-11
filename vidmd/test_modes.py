from utils import read_pickle_dump, setup_toy_grid
from pathlib import Path
from plotters import plot_3D
import numpy as np
import matplotlib.pyplot as plt

RUN = 2
x_range = (-10, 10)
t_range = (0, 4*np.pi)
f_range = (-2, 2)
X, t = setup_toy_grid(2000, 400)
X_line = np.linspace(*x_range, 2000)
T_line = np.linspace(*t_range, 400)
modes_file = f"modes_gen_{RUN}.pkl"
dataFolder = Path(__file__).parent.parent / "results"
modes_filepath = dataFolder / modes_file

modes = read_pickle_dump(modes_filepath)
print(modes.shape)

MODE = 0
plt.plot(T_line, modes[:, MODE])
plt.title(f"mode: {MODE + 1}")
plt.show()