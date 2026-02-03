from dataReader import DataReader, VideoReader
from dmdBase import DmdBase
from dmdVisualiser import DmdVideoVisualiser
import unittest
import pandas as pd
import numpy as np

def reader_fn(*args):
    return pd.read_csv(*args)

def vid_reader(*args):
    # return np.random.random(size=(1000, 256, 512))
    return np.random.randint(0, 255, size=(1001, 896, 256), dtype=np.uint8)

"""
data_holder = DataReader(filename="test.csv", folder=None, reader=reader_fn)
data = data_holder.read()
print(data)"""

vid_holder = VideoReader(filename="test.csv", folder= None, reader=vid_reader)
vid_data = vid_holder.read()
print(f"video data shape: {vid_data.shape}")

dmd_holder = DmdBase(vid_holder)
dmd_holder.update_state(data_read=False, columns_align_temporal=True, decomposed=False, modes_calculated=False)
dmd_holder.prepare_data(dmd_type_temporal=True)
dmd_holder.decompose()
dmd_holder.calc_low_rank_eigvecs_modes(r= 7)
dmd_holder.spectra(dt = 1)

vis_holder = DmdVideoVisualiser(dmdObject= dmd_holder)
vis_holder.visualise_tSpace_eigs(xlim=[-10, 2])
vis_holder.visualize_deflattened_modes(mode_numbers=[0, 1, 2, 3, 4])