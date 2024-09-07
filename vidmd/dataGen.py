import numpy as np
from pathlib import Path
from typing import Optional

def generator(x_range, t_range, *args, **kwargs):
    x1, x2 = x_range
    t1, t2 = t_range
    X = np.linspace(x1, x2, num=50)    # say some param x
    T = np.linspace(t1, t2, num=4)    # say some param t
    param_grid =  np.meshgrid(X, T)
    f1 = np.sech(X)

class DataGenerator:
    """Object to generator data, save it in a file
    
        ATTRIBUTES
        ----------
            args: dict
                arguments used in generator
            meta_info: dict
                meta data about the generator
            data: np.ndarray
                generated data with custom generator function
        
        METHODS
        -------
        generate_toy_data(generator_fn)
            generates toy case
        
        save_data(dum_filepath)
            saves the generated data in a custom file
    """
    def __init__(self, kwargs: Optional[dict] = None, metadata: Optional[dict] = None):
        self.kwargs = kwargs
        self.metadata = metadata

    def generate_toy_data(self, generator_fn):
        self.data: np.ndarray = generator_fn(self.kwargs)

    def save_data(self, filename: str , folder: Optional[Path]):
        folder = Path(__file__).parent.parent / "data"
        dump_filepath = folder / filename
        print(f"saving data in file: {dump_filepath}")
        self.data.dump(file= dump_filepath)
        print(f"Data dumped !")
