from dataReader import DataReader
from dmdBase import DmdBase
from dmdVisualiser import Visualiser
from typing import Optional

import numpy as np
import pandas as pd

def reader(*args) -> Optional[np.ndarray]:
    return np.random.random(size=(100, 20))

dataObj = DataReader(filename="test.csv", folder=None, reader=reader)
print(dataObj.read())
dmdObj = DmdBase(dataObject=dataObj)
dmdObj.prepare_data()
print(dmdObj.data)
dmdObj.decompose(r=10)
vis = Visualiser(dmdObjHandle=dmdObj)
vis.visualise_eigs()