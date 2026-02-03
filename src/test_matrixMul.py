import numpy as np

E = np.array(np.arange(0, 9), dtype=float)
V = np.random.random(size=(9, 9))
A = np.random.random(size=(1024, 9))
E_inv = np.array([np.reciprocal(elem) if elem else 0. for elem in E])
E_diag_inv = np.diag(E_inv)
print(E_inv)
print(E_diag_inv)
print((A @ V @ E_inv).shape)

eigvals, eigvecs = np.linalg.eig(A @ A.T)
print(f"eigen values {np.sqrt(eigvals)}")
print(f"eigenc vectors shape: {eigvecs.shape}")
