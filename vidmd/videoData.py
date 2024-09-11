from mraw import load_video, get_cih
from pathlib import Path
from utils import get_filename_with_ext
import matplotlib.pyplot as plt
import numpy as np


def get_pyro_video(dataFolder):
    files = [f for f in dataFolder.iterdir()]
    cihx_file = get_filename_with_ext(files, ".cihx")
    mraw_file = get_filename_with_ext(files, ".mrwa")
    print(get_cih(cihx_file))
    video, cihx_data = load_video(cih_file=cihx_file)
    print(f"video information, shape {video.shape}, dtype: {video.dtype}")
    print(f"max value {video.max()}")
    # plt.imshow((video[999, 0: 1024 - 256, 512 - 64: 1024 - 256 - 64]/ video.max())*255, cmap="hot")
    Y_MIN, Y_MAX = 0, 1024 - 128
    X_MIN, X_MAX = 512 - 64, 1024 - 256 - 64
    video = video[:, Y_MIN:Y_MAX, X_MIN:X_MAX] 
    video = video - video.mean(axis=0)
    print(f"Subtracting mean")
    print(f"post transform min: {video.min()}, max: {video.max()}")
    video_norm = 255 - np.array(255* (video - video.min())/ (video.max() - video.min()), dtype=np.uint8)
    print(f"post normalisation min: {video_norm.min()}, max: {video_norm.max()}")
    # plt.imshow(255* video[500, :, :]/ video.max(), cmap="hot")
    # plt.imshow(255* (video[500, :, :] - video.min())/ (video.max() - video.min()), cmap="hot")
    plt.imshow(video_norm[500, :, :], cmap="hot")
    plt.title(f"sample snapshot, selected pixels")
    plt.colorbar()
    plt.show()
    return video_norm

if __name__=="__main__":
    data_Folder = Path(__file__).parent.parent / "data"
    VIDEO_DUMP = "pyroVideo.pkl"
    video_data = get_pyro_video(dataFolder=data_Folder)
    video_data.dump(data_Folder/ VIDEO_DUMP)
    
