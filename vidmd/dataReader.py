
from typing import Type, TypeAlias, TypeVar, Optional, Callable
from types import FunctionType
from pathlib import Path
import numpy as np
import pandas as pd

__all__ = "DataReader"

def read_video(filepath: Path):
    "default reader if not reader is passed, assumes csv file format with comma delimiter"
    return pd.read_csv(filepath, delimiter=",")

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