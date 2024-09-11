import numpy as np

a = np.random.random(size=(10, 256, 256))

print(np.rollaxis(a.reshape(10, a.shape[1]*a.shape[2]), axis=1).shape)

print(np.arange(1, 10+1))