import numpy as np
import matplotlib.pyplot as plt
from dmdBase import DmdBase

class Visualiser:
    def __init__(self, dmdObjHandle):
        self.dmdObj: DmdBase  = dmdObjHandle

    def visualise_eigs(self):
        eigs = self.dmdObj.eigvals
        fig = plt.figure(figsize=(8, 8))
        plt.scatter(eigs.imag, eigs.real)
        plt.xlabel("$lambda$_{re}", fontsize = 18)
        plt.ylabel("$lambda$_{imag}", fontsize = 18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True)
        plt.title("Eigenvalues of $A^{tilde}$", fontsize = 20)
        plt.show()