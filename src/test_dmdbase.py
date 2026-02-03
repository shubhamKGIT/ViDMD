from dmdBase import DmdBase
from dataReader import DataReader
import numpy as np
import pandas as pd

dataObj = DataReader(filename="test.csv", folder=None, reader=None)
# print(dataObj.__dict__.keys())
print(dataObj)
dmd = DmdBase(dataObject=dataObj)

def my_dummy_reader(*args):
    n_cols = 100
    dataframe =  pd.DataFrame(np.random.random(size=(256*256, n_cols)), columns=[str(i) for i in range(n_cols)])
    data = dataframe.to_numpy(dtype = np.uint8)    # decides the datatype
    return data

dummy_Obj = DataReader(filename="something.some", folder=None, reader=my_dummy_reader)
data = dummy_Obj.read()
print(f"shape of data: {data.shape}")
dummy_dmd = DmdBase(dataObject=dummy_Obj)
print(dummy_dmd)
dummy_dmd.update_state(data_read= False, columns_align_temporal= True, decomposed=False, modes_calculated= False)
print(dummy_dmd)
dummy_dmd.prepare_data(dmd_type_temporal=True)
dummy_dmd.decompose()
dummy_dmd.calc_low_rank_eigvecs_modes(r = 30)
print(f"shape of modes: {dummy_dmd.modes.shape}")
