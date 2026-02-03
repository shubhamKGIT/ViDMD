from mraw import load_video, get_cih
from pathlib import Path
from utils import get_filename_with_ext, get_2D_mesh
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from plotters import show_imageplot, plot_3D, show_subimage, multi_plotter, subplots_row_col, multi_surf_plotter
import seaborn as sns

def trim_frames(video):
    Y_MIN, Y_MAX = 0, 1024 - 128   # pixel range (0, 896 as range 0, 896
    X_MIN, X_MAX = 512 - 64, 1024 - 256 - 64   # pixel range (448, 704) as range 0, 256
    trimmed_video = video = video[:, Y_MIN:Y_MAX, X_MIN:X_MAX]  # trimmed here
    return trimmed_video 

def normalise_video(video, a):
    video_norm = np.array(255* (video - video.min())/ (a*video.max() - video.min()),  dtype=np.uint8)
    return video_norm

def invert_video(video):
    return 255 - video

def get_pyro_video(dataFolder, subtract_mean = True):
    # file
    files = [f for f in dataFolder.iterdir()]
    cihx_file = get_filename_with_ext(files, ".cihx")
    mraw_file = get_filename_with_ext(files, ".mraw")
    print(get_cih(cihx_file))
    # reading video
    video, cihx_data = load_video(cih_file=cihx_file)
    print(f"video information, shape {video.shape}, dtype: {video.dtype}")
    print(f"max value {video.max()}")
    # transforming video data
    # plt.imshow((video[999, 0: 1024 - 256, 512 - 64: 1024 - 256 - 64]/ video.max())*255, cmap="hot")
    Y_MIN, Y_MAX = 0, 1024 - 128   # pixel range (0, 896 as range 0, 896
    X_MIN, X_MAX = 512 - 64, 1024 - 256 - 64   # pixel range (448, 704) as range 0, 256
    # video = video[:, Y_MIN:Y_MAX, X_MIN:X_MAX]  # trimmed here
    video = trim_frames(video)
    plt.hist(video[200: 800, :, :].flatten())
    plt.show()
    print(f"trimmed video shape: {video.shape}")
    if subtract_mean:
        video = video - video.mean(axis=0)
    print(f"Subtracting mean")
    print(f"post transform min: {video.min()}, max: {video.max()}")
    video_norm = invert_video(normalise_video(video, a=0.25))
    print(f"normalised video shape: {video_norm.shape}")
    print(f"post normalisation min: {video_norm.min()}, max: {video_norm.max()}")
    # plt.imshow(255* video[500, :, :]/ video.max(), cmap="hot")
    # plt.imshow(255* (video[500, :, :] - video.min())/ (video.max() - video.min()), cmap="hot")
    # display data
    EXP = 12
    FRAME_NUMS = [200, 500, 800]
    TITLE = f"Exp {EXP} sample snapshot @ {FRAME_NUMS[0]} $\Delta$t"
    show_imageplot(video_norm[FRAME_NUMS[0], :, :], cmap="hot", title=TITLE)
    # multi frame plot
    multi_plotter(sub_plotter_fn=show_subimage,
                  subplot_iterator=FRAME_NUMS,
                  data = video_norm,
                  stacked_as_flattened_cols=False,
                  frame_shape=None,
                  cmap=cm.hot,
                  subplot_title_str="Frame")
   
    # multiple surface plot
    X_mesh, Y_mesh = get_2D_mesh(video_norm[500, :, :])  # use any frame
    print(f"mesh, X shape: {X_mesh.shape}, Y shape: {Y_mesh.shape}")
    fig = plt.figure(figsize=(12, 4), facecolor="white")
    n_rows, n_cols = subplots_row_col(FRAME_NUMS)
    for i, frame_num in enumerate(FRAME_NUMS):
        ax = fig.add_subplot(n_rows, n_cols, i+1, projection='3d')
        subplot_title = f"snapshot @ {frame_num} $\Delta$t"
        plot_3D(X = X_mesh, Y = Y_mesh,  Z = video_norm[frame_num, :, :], 
                x_range=[0, X_MAX - X_MIN], y_range=[0, Y_MAX - Y_MIN], z_range=[0, 255],
                aspect=(1., 3.5, 1.),
                ax = ax,
                surf_cmap=cm.hot,
                edgecolor="gray",
                title=subplot_title,
                view_init=(-20, -40, 0)
                )
    plt.tight_layout()
    plt.show()

    # multi surf plot simplified
    multi_surf_plotter(FRAME_NUMS, video_norm, subplot_title_str="$\Delta$t = 0.001, Frame")
    return video_norm

if __name__=="__main__":
    data_Folder = Path(__file__).parent.parent / "data"
    VIDEO_DUMP = "pyroVideo02.pkl"
    video_data = get_pyro_video(dataFolder=data_Folder, subtract_mean=False)
    # video_data.dump(data_Folder/ VIDEO_DUMP)
    
