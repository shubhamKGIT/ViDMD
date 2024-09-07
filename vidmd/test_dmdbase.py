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
print(dummy_Obj)
dummy_dmd = DmdBase(dataObject=dummy_Obj)
dummy_dmd.prepare_data()
dummy_dmd.decompose(r=30)
print(f"shape of modess: {dummy_dmd.modes.shape}")
