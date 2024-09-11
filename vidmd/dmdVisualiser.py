import numpy as np
import matplotlib.pyplot as plt
from dmdBase import DmdBase


class DmdVisualiser:
    def __init__(self, dmdObjHandle):
        self.dmdObj: DmdBase  = dmdObjHandle

    def visualise_singular_vals(self):
        r = self.dmdObj.reduced_rank
        singular_vals = (self.dmdObj.E / np.sum(self.dmdObj.E))[:r]  # taking only first r
        fig = plt.figure(figsize=(8, 8))
        plt.scatter(np.arange(1, len(singular_vals) + 1), singular_vals)
        plt.xlabel("$k$", fontsize = 18)
        plt.xlim(0, len(singular_vals)+1)
        plt.ylabel("$\sigma_k$", fontsize = 18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True)
        plt.title("Singular Values", fontsize = 20)
        plt.show()

    def visualise_eigs(self):
        eigs = self.dmdObj.eigvals
        fig = plt.figure(figsize=(8, 8))
        plt.scatter(eigs.real, eigs.imag)
        plt.xlabel("$\lambda_r$", fontsize = 18)
        plt.ylabel("$\lambda_i$", fontsize = 18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True)
        plt.title("Eigenvalues of $\~A$, discrete space", fontsize = 20)
        plt.show()
    
    def visualise_tSpace_eigs(self, xlim):
        omega = self.dmdObj.omega
        fig = plt.figure(figsize=(8, 8))
        plt.scatter( x= omega.real, y = omega.imag)
        plt.xlabel("$\omega_r$", fontsize = 18)
        plt.xlim(xlim)
        plt.ylabel("$\omega_i$", fontsize = 18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True)
        plt.title("Eigenvalues of $\~A$, continuous time space", fontsize = 20)
        plt.show()

    def visualise_mode(self, mode_numbers):
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize = (8, 6))
        for mode_number in mode_numbers:
            ax.plot(self.dmdObj.modes[:, mode_number].real, linewidth = 4)
        ax.set_title(f"DMD modes", fontsize = 20)
        ax.set_xlabel("X", fontsize = 24)
        ax.set_ylabel("$\Phi$", fontsize = 24)
        ax.legend([mode_number+1 for mode_number in mode_numbers])
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.show()
    
class DmdVideoVisualiser(DmdVisualiser):
    def __init__(self, dmdObject):
        super().__init__(dmdObject)

    def visualize_deflattened_modes(self, mode_numbers):
        if len(mode_numbers) % 2 == 0:
            n_cols = 2
            n_rows = int(len(mode_numbers) / n_cols)
        elif len(mode_numbers) % 3 == 0:
            n_cols = 3
            n_rows = int(len(mode_numbers) / n_cols)
        else:
            n_cols = 5
            n_rows = (len(mode_numbers) / n_cols).__ceil__()
        fig, ax = plt.subplots(ncols=n_cols, nrows=n_rows)
        ax = ax.flatten()
        image_shape = self.dmdObj.dataObj.data_shape
        for i, mode_number in enumerate(mode_numbers):
            cax = ax[i].imshow(self.dmdObj.modes[:, mode_number].real.reshape(image_shape), cmap="gray")
            ax[i].set(title = f"Mode {mode_number + 1}")
        if len(mode_numbers) % 5 == 0:
            ax[-1].axis("off")
        plt.tight_layout()
        plt.show()
    
