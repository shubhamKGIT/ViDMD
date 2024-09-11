
from dataReader import DataReader, VideoReader
from dmdBase import DmdBase
from dmdVisualiser import DmdVideoVisualiser
from utils import read_pickle_dump
import pandas as pd
import numpy as np
from pathlib import Path

def random_video(*args):
    return np.random.randint(0, 255, size=(1001, 896, 256), dtype=np.uint16)

if __name__ == "__main__":
    dataFolder = Path(__file__).parent.parent / "data"
    # data handler
    vid_holder = VideoReader(filename="pyroVideo.pkl", folder= dataFolder, reader=read_pickle_dump)
    vid_data = vid_holder.read()
    print(f"video data shape: {vid_data.shape}")

    # dmd calcuator
    dmd_holder = DmdBase(vid_holder)
    dmd_holder.update_state(data_read=False, columns_align_temporal=True, decomposed=False, modes_calculated=False)
    dmd_holder.prepare_data(dmd_type_temporal=True)
    dmd_holder.decompose()
    dmd_holder.calc_low_rank_eigvecs_modes(r=20)
    dmd_holder.spectra(dt = 0.001)

    # visualisation
    vis_holder = DmdVideoVisualiser(dmdObject= dmd_holder)
    vis_holder.visualise_tSpace_eigs(xlim=[-500, 100])
    vis_holder.visualize_deflattened_modes(mode_numbers=[0, 2, 4, 6, 8])