from dataReader import DataReader
import unittest
import pandas as pd

def reader_fn(*args):
    return pd.read_csv(*args)

data_holder = DataReader(filename="test.csv", folder=None, reader=reader_fn)
data = data_holder.read()
print(data)