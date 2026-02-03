
from utils import read_pickle_dump, read_pickle_dumped_array
from pathlib import Path
from plotters import multi_surf_plotter, multi_plotter, show_subimage
from dmdVisualiser import plot_omega, plot_singularVals, plot_energyInModes, annotate_plot, show_subimage_annotated
from dmdVisualiser import plot_frequency
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

if __name__=="__main__":
    resultsFolder = Path(__file__).parent.parent / "results"
    IDENTIFIER = "pyro_02"
    dmdInstance_file = resultsFolder/ f"{IDENTIFIER}_dmdObj.pkl"
    modes_file = resultsFolder/ f"{IDENTIFIER}_modes.pkl"
    omega_file = resultsFolder/ f"{IDENTIFIER}_omega.pkl"
    singularVals_file = resultsFolder/ f"{IDENTIFIER}_singularVals.pkl"
    # MODE_NUMBERS = [1, 3, 6, 8, 12]
    MODE_NUMBERS = [10, 11, 14, 16, 18]
    indexes = [mode -1 for mode in MODE_NUMBERS]
    dmdObj = read_pickle_dump(filepath=dmdInstance_file)
    modes = dmdObj.modes
    omega = dmdObj.omega
    singularVals = dmdObj.E
    # modes = read_pickle_dumped_array(filepath=modes_file)
    # omega = read_pickle_dumped_array(filepath=omega_file)
    # singularVals = read_pickle_dumped_array(filepath=singularVals_file)
    # annotated = list(np.arange(5)) + [10, 15]
    annotated = indexes
    plot_omega(omega, xlim=[-1000, 100], ylim=[-600, 600], annotations=annotated)
    plot_frequency(omega, dmdObj.b)
    print(f"Data was calculated with reduced rank, r of {len(omega)}")
    plot_singularVals(singularVals[:len(omega)])
    plot_energyInModes(singularVals[:len(omega)])
    multi_plotter(show_subimage_annotated, 
                  indexes, 
                  data=modes.real, 
                  stacked_as_flattened_cols=True,
                  frame_shape=(896, 256),
                  cmap=cm.hot,
                  subplot_title_str="Mode",
                  omega = omega[MODE_NUMBERS])
    # print(f"modes read from {modes_file}, has shape: {modes.shape}")
    multi_surf_plotter(data = modes.real, 
                       subplot_iterator= indexes,
                       stacked_as_flattened_cols=True,
                       frame_shape=(896, 256),
                       cmap = cm.hot,
                       subplot_title_str="DMD Mode")