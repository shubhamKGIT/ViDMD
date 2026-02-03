import numpy as np
from pathlib import Path
from typing import Optional
import pickle

class DataGenerator:
    """Object to generate data, save it in a file
    
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


def generate_synthetic_dmd_video(
    frame_shape: tuple[int, int],
    timesteps: int,
    lambdas: np.ndarray,
    amplitudes: Optional[np.ndarray] = None,
    seed: int = 0,
    real_output: bool = False,
):
    """Generate a low-rank synthetic "video" dataset with known DMD modes.

    Returns a dict with:
        data: (n_pixels, timesteps) matrix with columns as time snapshots
        modes: (n_pixels, r) true spatial modes
        lambdas: (r,) discrete-time eigenvalues
        amplitudes: (r,) modal amplitudes
        frame_shape: (H, W) for deflattening
    """
    rng = np.random.default_rng(seed)
    h, w = frame_shape
    n_pixels = h * w
    lambdas = np.asarray(lambdas, dtype=complex)
    r = lambdas.shape[0]
    if amplitudes is None:
        amplitudes = rng.uniform(0.5, 2.0, size=r)
    else:
        amplitudes = np.asarray(amplitudes, dtype=complex)

    # Random spatial modes, orthonormalized for stability
    phi_raw = rng.standard_normal(size=(n_pixels, r))
    phi, _ = np.linalg.qr(phi_raw)

    t = np.arange(timesteps, dtype=float)
    dynamics = np.vstack([amplitudes[i] * (lambdas[i] ** t) for i in range(r)])
    data = phi @ dynamics
    if real_output:
        data = data.real

    return {
        "data": data,
        "modes": phi,
        "lambdas": lambdas,
        "amplitudes": amplitudes,
        "frame_shape": frame_shape,
    }
