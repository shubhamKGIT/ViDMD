import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

def plot_3D(X, Y, Z, x_range, y_range, z_range):
    ax = plt.figure().add_subplot(projection='3d')
    if X is None or Y is None or Z is None:
        print(f"plotting sample as input dataset missing !!. Please check data passed.")
        X, Y, Z = axes3d.get_test_data(0.05)

    # Plot the 3D surface
    ax.plot_surface(X, Y, Z, edgecolor='royalblue', lw=0.5, rstride=8, cstride=8,
                    alpha=0.8)

    # Plot projections of the contours for each dimension.  By choosing offsets
    # that match the appropriate axes limits, the projected contours will sit on
    # the 'walls' of the graph.
    ax.contour(X, Y, Z, zdir='z', offset=-100, cmap='coolwarm')
    ax.contour(X, Y, Z, zdir='x', offset=-40, cmap='coolwarm')
    ax.contour(X, Y, Z, zdir='y', offset=40, cmap='coolwarm')
    ax.set(xlim=x_range, ylim = y_range, zlim= z_range,
        xlabel='X', ylabel='Y', zlabel='Z')

    # Set zoom and angle view
    ax.view_init(40, -30, 0)
    ax.set_box_aspect(None, zoom=0.9)
    plt.show()

# plot_3D(None, None, None)