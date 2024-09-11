from dataReader import DataReader
from dmdBase import DmdBase
from dmdVisualiser import DmdVisualiser
from typing import Optional

import numpy as np
import pandas as pd

def reader(*args) -> Optional[np.ndarray]:
    return np.random.random(size=(100, 20))

dataObj = DataReader(filename="test.csv", folder=None, reader=reader)
print(dataObj.read())
dmdObj = DmdBase(dataObject=dataObj)
dmdObj.prepare_data(dmd_type_temporal=True)
print(dmdObj.data)
dmdObj.decompose()
dmdObj.calc_low_rank_eigvecs_modes(r = 10)
dmdObj.spectra(dt= 0.1)
vis = DmdVisualiser(dmdObjHandle=dmdObj)
vis.visualise_eigs()
vis.visualise_tSpace_eigs(xlim= [-100, 10])