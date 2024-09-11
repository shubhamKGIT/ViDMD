
import numpy as np
from typing import Type, TypeAlias, TypeVar, Optional, Union
from pathlib import Path
import pandas as pd
from dataReader import DataReader, VideoReader
from utils import timeit

__all__ = "DmdBase"

def data_reader_fn(*args):
    "basic reader function"
    return pd.read_csv(*args)

class DmdBase:
    """Sets up the DMD data, reads and rewrites data, decomposes, finds relation matrix

        PARAMS
        ------
            dataObj: dataObject
                dataHolder with reader function as dataReader attribute
            state: dict
                metadata about object
    
        ATTRIBUTES
        ----------
            dataObj: DataReader
                data object which can be used to get data filepath and data reader function, called when data needed
            state: dict
                data_read, columns_align_temporal, decomposed
            data: ndaray | None
                main dataset
            U, E, V_t: Decomposed Matrices, np.ndarray
                U, E, V_t = svd(data)
            time_shifted: Time Shifted Matrix, np.ndarray
                Time shifted copy of data for building the relations matrix,  U(n+1) = A*U(n)
            eigvals, eigvecs = eig(A_tilde) where A_tilde is low rank relationship matrix between data[]: , :-1] and time shifted or data[:, 1:]
                A_tilde = X' @ X.T
            modes = low rank modes
            
        METHODS
        -------
            prepare_data(array_dtype)
                fetches by calling dataObj.read() and persists the data in the DMD object 

            decompose(data)
                decomposes the data

            _time_shift(data)
                generates time shifted copy of data matrix
    """
    def __init__(self, dataObject):
        self.dataObj: DataReader | VideoReader = dataObject
        self.state = {"data_read": False, 
                      "columns_align_temporal": False, 
                      "decomposed": False,
                      "modes_calculated": False
                      }
        self.data = None
    
    def __repr__(self):
        return f" {self.__class__.__name__} has attributes: {self.__dict__.keys()}, \n Object state: {self.state}"
    
    def update_state(self, data_read: bool, columns_align_temporal: bool, decomposed: bool, modes_calculated: bool):
        self.state["data_read"] = data_read
        self.state["columns_align_temporal"] = columns_align_temporal
        self.state["decomposed"] = decomposed
        self.state["modes_calculated"] = modes_calculated

    @timeit
    def read_data(self):
        print(f"calling data read from {self.__class__.__name__} with attributes {self.dataObj.__dict__.keys()}")
        self.data: np.ndarray = self.dataObj.read()   # dataObject is assumed to have read method which returns numpy array
        print(f"Data read into the DMD object, please check if columns_align_temporal and update state")
        self.state["data_read"] = True
    
    def transpose_data(self):
        "can run transpose of data before decompose if spatial modes needed"
        self.data = self.data.T
    
    def _time_shift(self, data):
        "shifts the columns of data to right"
        return data[:, 1:]
    
    def prepare_data(self, dmd_type_temporal: bool = True):
        "read data if not read yet, makes time shifted copy"
        if not self.state["data_read"]:
            self.read_data()
        if dmd_type_temporal and not self.state["columns_align_temporal"]:
            print(f"Temporal DMD needed, columns do not align temporal, transposing the data")
            self.transpose_data()
        elif not dmd_type_temporal and self.state["columns_align_temporal"]:
            print(f"Spatial DMD needed, data also aligned temporal, transpose needed")
            self.transpose_data()
        else: 
            print(f"column alighnment fits the type of dmd")
        print(f"shape of array X: {self.data[:, :-1].shape}")    # (1 to n-1 columns)
        self.time_shifted = self._time_shift(self.data)    # (2 to n columns) or X'
        print(f"shape of time shifted array X': {self.time_shifted.shape}")
    
    @timeit
    def fast_decompose(self):
        "decomposes using method of snapshots"
        X = self.data[:, :-1] 
        M = X.T @ X
        print(f"shape of M matrix: {M.shape}")
        V, S, V_t = np.linalg.svd(M)
        E_inv = np.array([np.reciprocal(elem) if elem else 0. for elem in np.sqrt(S)])
        self.U = X @ np.diag(E_inv) @ V
        self.E = np.sqrt(S)
        self.V_t = V_t
        print(f"shape of decomposed matrices, U: {self.U.shape}, E: {self.E.shape}, V_t = {self.V_t.shape}")

    @timeit
    def decompose(self):
        """decomposes data matrix, builds relationship matrix, find the eigenvectors of reduced order A
        """
        if not self.state["decomposed"]:
            X = self.data[:, :-1]   # (1 to n-1 columns)
            # Decomposition of X
            print(f"Starting decomposition of X")
            self.U, self.E, self.V_t = np.linalg.svd(X, full_matrices=False)
            print(f"Decomposition finined! \n\n")
            print(f"shape of decomposed matrices, U: {self.U.shape}, E: {self.E.shape}, V_t = {self.V_t.shape}")
            print(f"raw value of E, first 10: {self.E[:10]}")
            self.state["decomposed"] = True
        else:
            print(f"Data already decomposed in U, E, V_t, use attributes of these names or update object state to rerun.")
        
    @timeit
    def calc_low_rank_eigvecs_modes(self, r: int):
        "calculates low rank (r) eigenvectors of matrix A"
        self.reduced_rank = r
        E_diag = np.diag(self.E)
        E_inv = np.array([np.reciprocal(elem) if elem else 0. for elem in self.E])
        print(f"decomposed matrix shapes: \n U:  {self.U.shape}, E_diag: {E_diag.shape}, V_t: {self.V_t.shape}")
        # Low rank truncation
        print(f"Starting eigen-calculation of reduced order relation matrix, A_tilde ...")
        U_r = self.U[:, :r]
        Einv_r = np.diag(E_inv)[:r, :r]
        Vt_r = self.V_t[:r, :]
        print(f"Truncated U, S, V_t matrix shapes: \n U_r:  {U_r.shape}, Einv_r: {Einv_r.shape}, Vt_r: {Vt_r.shape}")
        # Full rank
        # A_full = self.U.T @ self.time_shifted @ self.V_t.T @ np.diag(E_inv)
        # Low rank relationship matrix A_tilde
        A_tilde = (U_r.T @ self.time_shifted) @ (Vt_r.T @ Einv_r) # relation matrix
        print(f"shape of low-rank relationship matrix, A_tilde : {A_tilde.shape}")
        # Decomposition of A_tilde
        print(f"Calculaing eigenvecs of A_tilde ...")
        self.eigvals, self.eigvecs = np.linalg.eig(A_tilde)
        print(f"Eigenvector calculation finished!")
        print(f"first 5 eigenvalues of A_tilde (lambdas): {self.eigvals[:5]} \n shape of eigenvecs of A_tilde: {self.eigvecs.shape}")
        print(f"Calculating Modes ... ")
        self.modes = self.time_shifted @ (Vt_r.T @ Einv_r) @ self.eigvecs
        print(f"Modes calculation finished!")
        self.state["modes_calculated"] = True
        print(f"Finished all dmd calculations ! \n\n")
    
    @timeit
    def spectra(self, dt):
        print(f"Calculating dmd spectra...")
        self.omega = np.log(self.eigvals)/ dt
        print(f"Calculation finished!")
        print(f"first 5 continous time eigenvalues of A_tilde (omegas): {self.omega[:5]} \n")
    
    @timeit
    def coeffs(self):
        print(f"Getting coefffs b...")
        x1 = self.data[:, 0]
        self.b = np.linalg.pinv(self.modes) @ x1    # inital coefficients
        print(f"Calculation finished!")

    def recons(self, T_space):
        m1 = self.data.shape[1] - 1   # 
        self.dynamics = np.zeros(shape=(self.reduced_rank, m1))
        print(f"dynamics vector(e^omega*t).*b where b is coeffs has shape of: {self.dynamics.shape} ")
        for i, t in enumerate(T_space[:-1]):   # T Space has all t's, we need to skip last one
            self.dynamics[:, i] = np.diag(self.b) @ np.exp(self.omega * t)
        X_dmd = self.modes @ self.dynamics
        return X_dmd