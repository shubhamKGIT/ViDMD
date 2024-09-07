import numpy as np
from pathlib import Path
from typing import Optional
import pickle


class DataGenerator:
    """Object to generator data, save it in a file
    
        ATTRIBUTES
        ----------
            dataSource: dict
                arguments used in generator
            metadata: dict
                meta data about the generator
        
        METHODS
        -------
        generate_toy_data(generator_fn)
            generates toy case
        
        save_data(dum_filepath)
            saves the generated data in a custom file
    """
    def __init__(self, dataSource: Optional[dict] = None, metadata: Optional[dict] = None):
        self.dataSource = dataSource
        self.metadata = metadata
    
    def __repr__(self):
        return f"attributes: {self.__dict__.keys()}"

    def generate_toy_data(self, generator_fn, *args, **kwargs):
        "generator from custom function passed"
        self.data: np.ndarray = generator_fn(*args, **kwargs)

    def save_data(self, filename: str, folder: Optional[Path] = None):
        "dumps data to a file, can be read later, prefer pickle format"
        if folder is None:
            folder = Path(__file__).parent.parent / "data"
        dump_filepath = folder / filename
        print(f"saving data in file: {dump_filepath}")
        self.data.dump(file= dump_filepath)
        print(f"Data dumped!")
    
    def read_pickle_dump(self, filepath, overwrite_data_attrib: bool = False):
        "reads from a pickle dump"
        print(f"Reading data from filepath: {filepath}")
        data = []
        with open(filepath, "rb") as f:
            while True:
                try:
                    data.append(pickle.load(f))
                except EOFError:
                    break
        if overwrite_data_attrib:
            self.data = data
        # return np.array(np.squeeze(data, axis = 0), dtype=np.uint8) # data has extra dimention so squeezed
        return np.array(np.squeeze(data, axis=0))
