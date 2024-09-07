
import numpy as np
from typing import Type, TypeAlias, TypeVar, Optional, Union
from pathlib import Path
import pandas as pd
# Frame = TypeVar("Frame", np.ndarray[np.uint8], list[np.uint8, np.uint8])
# Video = TypeVar("Video", list[Frame], np.array[Frame], Frame, list[Frame])
from dataReader import DataReader
    
def data_reader_fn(*args):
    return pd.read_csv(*args)

class DmdBase:
    """Sets up the DMD data, reads and rewrites data, decomposes, finds relation matrix
    
        ATTRIBUTES
        ----------
            dataObj: DataReader
                data object which can be used to get data filepath and data reader function, called when data needed
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
        self.dataObj: DataReader = dataObject
    
    def __repr__(self):
        return f" {self.__class__.__name__} has attributes: {self.__dict__.keys()}"
    
    def prepare_data(self):
        print(f"calling data read from {self.__class__.__name__} with attributes {self.dataObj.__dict__.keys()}")
        self.data: np.ndarray = self.dataObj.read()   # dataObject is assumed to have read method which returns numpy array

    def decompose(self, r = 5):
        "r is low rank proejction of relationship matrix A"
        X = self.data[:, :-1]   # (1 to n-1 columns)
        self.time_shifted = self._time_shift(self.data)    # (2 to n columns) or X'
        print(f"shape of time shifted array X': {self.time_shifted.shape}")
        # Decomposition of X
        print(f"Starting decomposition.")
        self.U, self.E, self.V_t = np.linalg.svd(X)
        print(f"Decomposition finined !")
        print(f"raw value of E: {self.E}")
        E_diag = np.diag(self.E)
        E_inv = np.array([np.reciprocal(elem) if elem else 0. for elem in self.E])
        print(f"decomposed matrix details: \n U:  {self.U.shape}, E_diag: {E_diag.shape}, V_t: {self.V_t.shape}")
        # Low rank truncation
        U_r = self.U[:, :r]
        Einv_r = np.diag(E_inv)[:r, :r]
        Vt_r = self.V_t[:r, :]
        print(f"Truncated matrix details: \n U_r:  {U_r.shape}, Einv_r: {Einv_r.shape}, Vt_r: {Vt_r.shape}")
        # Full rank
        A_full = self.U.T @ self.time_shifted @ self.V_t.T @ np.diag(E_inv)
        # Low rank relationship matrix A_tilde
        A_tilde = (U_r.T @ self.time_shifted) @ (Vt_r.T @ Einv_r) # relation matrix
        print(f"shape of low-rank relationship matrix A_tilde : {A_tilde.shape}")
        # Decomposition of A_tilde
        print(f"Calculation of eigenvecs of A_tilde ...")
        self.eigvals, self.eigvecs = np.linalg.eig(A_tilde)
        print(f"eigenvalues: {self.eigvals} \n shape of eigenvecs: {self.eigvecs.shape}")
        print(f"Calculating Modes ... ")
        self.modes = self.time_shifted @ (Vt_r.T @ Einv_r) @ self.eigvecs
        print(f"Finished all calculations !")
    
    def _time_shift(self, data):
        "shifts the timeseries data to right"
        return data[:, 1:]
        