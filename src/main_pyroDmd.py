
from dataReader import DataReader, VideoReader
from dmdBase import DmdBase
from dmdVisualiser import DmdVideoVisualiser
from utils import read_pickle_dumped_array
from videoData import get_2D_mesh
import pandas as pd
import numpy as np
from pathlib import Path
from plotters import plot_3D, mode_surface_plot, multi_surf_plotter, multi_plotter, show_imageplot, show_subimage
import matplotlib.pyplot as plt
from matplotlib import cm
import pickle

def random_video(*args):
    return np.random.randint(0, 255, size=(1001, 896, 256), dtype=np.uint16)

if __name__ == "__main__":
    dataFolder = Path(__file__).parent.parent / "data"
    resultsFolder = Path(__file__).parent.parent / "results"
    RESULT_FILE_IDT = "pyro_02"
    # data handler
    vid_holder = VideoReader(filename="pyroVideo02.pkl", 
                             folder= dataFolder, 
                             reader=read_pickle_dumped_array
                             )
    vid_data = vid_holder.read()
    print(f"video data shape: {vid_data.shape}")

    # dmd calcuator
    dmd_holder = DmdBase(vid_holder)
    dmd_holder.update_state(data_read=False, 
                            columns_align_temporal=True, 
                            decomposed=False, 
                            modes_calculated=False
                            )
    dmd_holder.prepare_data(dmd_type_temporal=True)
    dmd_holder.decompose()
    dmd_holder.calc_low_rank_eigvecs_modes(r=20)
    dmd_holder.spectra(dt = 0.001)
    dmd_holder.eigvals.dump(resultsFolder/ f"{RESULT_FILE_IDT}_eigvals.pkl")
    dmd_holder.omega.dump(resultsFolder/ f"{RESULT_FILE_IDT}_omega.pkl")
    dmd_holder.modes.dump(resultsFolder/ f"{RESULT_FILE_IDT}_modes.pkl")
    dmd_holder.E.dump(resultsFolder/ f"{RESULT_FILE_IDT}_singularVals.pkl")
    dmd_holder.coeffs()
    print(f"shape of coeffs: {dmd_holder.b.shape}")
    X_recons = dmd_holder.recons(T_space=np.linspace(0.001, 1.000, 1000))
    print(f"reconstructed shpe: {X_recons.shape}")
    dmd_dumpFile = resultsFolder/ f"{RESULT_FILE_IDT}_dmdObj.pkl"
    with open(dmd_dumpFile, 'wb') as f:
        pickle.dump(dmd_holder, f)
        f.close()
    print(f"dmd object dumped in {dmd_dumpFile} !")
    multi_plotter(sub_plotter_fn = show_subimage, 
                  data=X_recons.real, 
                  subplot_iterator=[200, 500, 800], 
                  stacked_as_flattened_cols=True, 
                  frame_shape=(896, 256),
                  cmap=cm.hot,
                  subplot_title_str="Recons snapshot")
    # show_imageplot(X_recons[:, 500].real.reshape(896, 256), cmap="hot", title="Reconstructred snapshot" )
    # visualisation
    mode_numbers=[0, 2, 4, 6, 8]
    vis_holder = DmdVideoVisualiser(dmdObject= dmd_holder)
    vis_holder.visualise_tSpace_eigs(xlim=[-500, 100])
    vis_holder.visualize_deflattened_modes(mode_numbers)
    # mode surface plot, verification

    for mode in mode_numbers:
        mode_surface_plot(dmd_holder=dmd_holder, mode_number=mode)
        plt.show()
