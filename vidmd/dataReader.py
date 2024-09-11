
from typing import Optional, Callable
from pathlib import Path
import pandas as pd
import numpy as np

__all__ = "DataReader"

def read_video(filepath, *args):
    "default reader if not reader is passed, assumes csv file format with comma delimiter"
    return pd.read_csv(filepath, delimiter=",", *args)

class DataReader:
    """Stores data path, reads data with reader function passed

    ATTRIBUTES
    ----------
        filepath: Path
            data file path
        dataReader: Callable
            reader function which reads the data

    METHODS
    -------
        read(reader): reads the data using custom reader function passed
    
    """
    def __init__(self, filename: str, folder: Optional[Path], reader: Optional[Callable]):
        try:
            if folder is None:
                folder = Path(__file__).parent.parent / "data"
            self.filepath = folder/ filename
        except:
            raise Exception("Cound not find the data file")
        try:
            if reader is None:
                reader = read_video
            self.dataReader = reader
        except:
            raise Exception("Data reader function not defined properly")
    
    def __repr__(self):
        return f" {self.__class__.__name__} has attributes: {self.__dict__.keys()}"
        
    def read(self):
        print(f"Reading data from file: {self.filepath} with reader function: {self.dataReader.__name__}")
        return self.dataReader(self.filepath)

class VideoReader(DataReader):
    "Object to read video frames"
    def __init__(self, filename, folder, reader):
        super().__init__(filename, folder, reader)
    
    def read(self):
        print(f"Reading video data from: {self.filepath}, using custom reader function: {self.dataReader.__name__}")
        raw_data = self.dataReader(self.filepath)
        print(f"raw data shape: {raw_data.shape}, type: {type(raw_data)}")
        self.data_shape = (raw_data.shape[1], raw_data.shape[2])
        print(f"Data Read! \n")
        return self.flatten_and_roll(raw_data)
    
    def flatten_and_roll(self, data):
        "assumed input data in the forms of (time, n_pixel, n_pixel)"
        return np.rollaxis(data.reshape(data.shape[0], data.shape[1]*data.shape[2]), axis=1)