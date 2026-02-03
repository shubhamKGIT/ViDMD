import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from utils import get_2D_mesh
from matplotlib import cm
import numpy as np
from functools import wraps
from typing import Optional

def show_imageplot(image, cmap, title):
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.show()

def show_subimage(ax, image, cmap, title, index = None, **kwargs):
    ax.imshow(image, cmap=cmap, **kwargs)
    ax.set(title = title)

def plot_3D(X, Y, Z, 
            x_range, y_range, z_range,
            ax = None,
            aspect = None,
            surf_cmap = cm.Blues,
            edgecolor = 'royalblue', 
            title = "surface_plot",
            view_init = (40, -30, 0),
            **kwargs
            ):
    if ax is None:
        ax = plt.figure().add_subplot(projection='3d')
    if X is None or Y is None or Z is None:
        print(f"plotting sample as input dataset missing !!. Please check data passed.")
        X, Y, Z = axes3d.get_test_data(0.05)

    # Plot the 3D surface
    ax.plot_surface(X, Y, Z, 
                    cmap= surf_cmap, 
                    edgecolor= edgecolor, 
                    lw=0.5, rstride=8, cstride=8,
                    alpha=0.8
                    )

    # Plot projections of the contours for each dimension.  By choosing offsets
    # that match the appropriate axes limits, the projected contours will sit on
    # the 'walls' of the graph.
    ax.contour(X, Y, Z, zdir='z', offset=-100, cmap='coolwarm')
    ax.contour(X, Y, Z, zdir='x', offset=-40, cmap='coolwarm')
    ax.contour(X, Y, Z, zdir='y', offset=40, cmap='coolwarm')
    ax.set(xlim=x_range, 
           ylim = y_range, 
           zlim= z_range,
           xlabel='X', 
           ylabel='Y', 
           zlabel='Z',
           xticklabels = [],
           yticklabels = [],
           title= title
           )
    # Set zoom and angle view
    ax.view_init(*view_init)
    ax.set_box_aspect(aspect, zoom=0.9)

def mode_surface_plot(dmd_holder, mode_number = 2):
    "plotting video based modes as 2D surfaces"
    # surface vide of mode
    X_mesh, Y_mesh = get_2D_mesh(dmd_holder.modes[:, 0].real.reshape(896, 256))
    Y_MIN, Y_MAX = 0, 1024 - 128
    X_MIN, X_MAX = 512 - 64, 1024 - 256 - 64
    mode_data = dmd_holder.modes[:, mode_number].real
    Z_MIN, Z_MAX = mode_data.min(), mode_data.max()
    plot_3D(X_mesh, Y_mesh,  mode_data.reshape(896, 256), 
                x_range=[0, X_MAX - X_MIN], y_range=[0, Y_MAX - Y_MIN], z_range=[Z_MIN, Z_MAX],
                aspect=(1., 3., 1.),
                view_init=(-20, -40, 0)
                )

def subplots_row_col(counter_list):
        if len(counter_list) % 2 == 0:
            n_cols = 2
            n_rows = int(len(counter_list) / n_cols)
        elif len(counter_list) % 3 == 0:
            n_cols = 3
            n_rows = int(len(counter_list) / n_cols)
        else:
            n_cols = 5
            n_rows = (len(counter_list) / n_cols).__ceil__()

        return n_rows, n_cols
    

def multi_plotter(sub_plotter_fn,
                    subplot_iterator: list, 
                    data: np.ndarray, 
                    stacked_as_flattened_cols: bool = False, 
                    frame_shape: Optional[tuple[int, int]] = None, 
                    cmap= "gray",
                    subplot_title_str: str = "Mode",
                    **kwargs):
    n_rows, n_cols = subplots_row_col(subplot_iterator)
    fig, ax = plt.subplots(ncols=n_cols, nrows=n_rows)
    ax = ax.flatten()   
    for i, iter in enumerate(subplot_iterator):
        if stacked_as_flattened_cols:
            sub_data = data[:, iter].reshape(frame_shape)  # flat images in time series as [flat_image, mode]
        else: 
            sub_data = data[iter, :, :]  # video data as [frame_num, width * height]
        # plotter(data, cmap=cmap, title = f"{subplot_title_str} {iter + 1}")
        subplot_title = f"{subplot_title_str} {iter + 1}"
        sub_plotter_fn(ax[i], sub_data, cmap=cmap, title=subplot_title, index = i, **kwargs)
    if n_cols == 5 and len(subplot_iterator) % 5 != 0:
        ax[-1].axis("off")
    plt.tight_layout()
    plt.show()
        
def multi_surf_plotter(subplot_iterator,
                       data, 
                       stacked_as_flattened_cols: bool = False, 
                       frame_shape: Optional[tuple[int, int]] = None, 
                       cmap= cm.hot,
                       subplot_title_str="$\Delta$t = 0.001, Frame"):
    Y_MIN, Y_MAX = 0, 1024 - 128   # pixel range (0, 896 as range 0, 896
    X_MIN, X_MAX = 512 - 64, 1024 - 256 - 64   # pixel range (448, 704) as range 0, 256
    Z_MIN, Z_MAX = data.min(), data.max()
    if stacked_as_flattened_cols:
        X_mesh, Y_mesh = get_2D_mesh(None, (896, 256))
    else:
        X_mesh, Y_mesh = get_2D_mesh(data[0])  # use any frame
    print(f"mesh, X shape: {X_mesh.shape}, Y shape: {Y_mesh.shape}")
    fig = plt.figure(figsize=(12, 8), facecolor="white")
    n_rows, n_cols = subplots_row_col(subplot_iterator)
    for i, iter in enumerate(subplot_iterator):
        if stacked_as_flattened_cols:
            sub_data = data[:, iter].reshape(frame_shape)  # flat images in time series as [flat_image, mode]
        else: 
            sub_data = data[iter]  # video data as [frame_num, width * height]
        ax = fig.add_subplot(n_rows, n_cols, i+1, projection='3d')
        subplot_title = f"{subplot_title_str} {iter}"
        plot_3D(X = X_mesh, Y = Y_mesh,  Z = sub_data, 
                x_range=[0, X_MAX - X_MIN], y_range=[0, Y_MAX - Y_MIN], z_range=[Z_MIN, Z_MAX],
                aspect=(1., 3.5, 1.),
                ax = ax,
                surf_cmap=cmap,
                edgecolor="gray",
                title=subplot_title,
                view_init=(-20, -40, 0)
                )
    # plt.tight_layout()
    plt.show()
