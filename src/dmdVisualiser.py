import numpy as np
import matplotlib.pyplot as plt
from dmdBase import DmdBase
from adjustText import adjust_text

def show_subimage_annotated(ax, image, cmap, title, **kwargs):
    ax.imshow(image, cmap=cmap)
    omega = kwargs["omega"]
    index = kwargs["index"]
    omega_i = omega[index]
    extra_text = f"\n$\omega$ = {omega_i:.2f}"
    ax.set(title = title + extra_text)

def annotate_plot(points_to_annotate, x, y, labels, ax: plt.Axes):
    # Annotate only selected points
    for i in points_to_annotate:
        ax.annotate(labels[i], (x[i], y[i]), textcoords="offset points", xytext=(0, 10), ha='right', va="top", fontsize=20)


def plot_omega(omega, xlim, ylim, annotations: list[int] | None = [0, 2, 4, 6]):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot()
    ax.scatter( x= omega.real, y = omega.imag)
    ax.set_xlabel("$\omega_r$", fontsize = 24)
    ax.set_xlim(xlim)
    ax.set_ylabel("$\omega_i$", fontsize = 24)
    ax.set_ylim(ylim)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    if annotations is not None:
        labels=[an + 1 for an in annotations]
        texts = [plt.text(omega.real[i], omega.imag[i], label, fontsize=20) for i, label in zip(annotations, labels)]
        adjust_text(texts, autoalign='xy', expand_text= (1.05, 1,2), arrowprops=dict(arrowstyle='->', color='red', lw=2))
        # annotate_plot(annotations, omega.real[annotations], omega.imag[annotations], labels=[an + 1 for an in annotations], ax=ax)
    ax.grid(True)
    # ax.set_title("Eigenvalues of $\~A$, continuous time space", fontsize = 20)
    plt.show()

def plot_frequency(omega, amplitude):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot()
    frequency = np.abs(omega.imag) / (2*np.pi)
    order_indices = np.argsort(-frequency)
    amp_norm = np.log(np.abs(amplitude))
    ax.stem(frequency[order_indices], amp_norm[order_indices], markerfmt='o')
    ax.set_xlabel("$Frequecy, 1/sec $", fontsize = 18)
    ax.set_ylabel("Amplitide, b", fontsize = 18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax.set_title("Frequency of oscillations", fontsize = 20)
    plt.show()


def plot_singularVals(singular_vals):
    fig = plt.figure(figsize=(8, 8))
    singular_vals = singular_vals
    plt.scatter(np.arange(1, len(singular_vals) + 1), np.log(singular_vals))
    plt.xlabel("$k$", fontsize = 18)
    plt.xlim(0, len(singular_vals)+1)
    plt.ylabel("log($\sigma_k$)", fontsize = 18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.title("Singular Values", fontsize = 20)
    plt.show()

def plot_energyInModes(singular_vals):
    fig = plt.figure(figsize=(8, 8))
    singular_vals = singular_vals/ np.sum(singular_vals)
    cum_energy = np.cumsum(singular_vals)
    plt.plot(np.arange(1, len(singular_vals) + 1), cum_energy, marker="*")
    plt.xlabel("$k$", fontsize = 18)
    plt.xlim(0, len(singular_vals)+1)
    plt.ylabel("$\Sigma \sigma_k$", fontsize = 18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.title("Cummmulative energy in modes", fontsize = 20)
    plt.show()

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
    
