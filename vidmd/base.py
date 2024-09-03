from pydmd import DMD, BOPDMD
from pydmd.plotter import plot_summary
from pydmd.preprocessing import zero_mean_preprocessing

import numpy as np
import matplotlib.pylab as plt
N_SNAPS = 10
SNAP_SHAPE = 1024

dmd = DMD(svd_rank=4)
centered_dmd = zero_mean_preprocessing(DMD(svd_rank=3))
X = np.random.random(size=(SNAP_SHAPE, N_SNAPS))
# Y = np.tanh(X)
t = np.arange(N_SNAPS)

# dmd.fit(X)
# centered_dmd.fit(X)
# plot_summary(centered_dmd)

# plt.scatter(t, np.mean(X, axis=0))
# plt.show()

bpodmd = BOPDMD(
    svd_rank=4,
    num_trials=30,
    trial_size=0.4,
    eig_constraints = {"imag", "conjugate_pairs"},
    varpro_opts_dict = {"tol": 0.2, "verbose": True}
)
bpodmd.fit(X, t)

plot_summary(
    bpodmd,
    figsize=(12, 8),
    index_modes = (0, 2, 4),
    snapshots_shape = SNAP_SHAPE,
    order="F",
    mode_cmap="siesmic",
    dynamics_color="k",
    flip_continuous_axes = True,
    max_sval_plot = 3
)
