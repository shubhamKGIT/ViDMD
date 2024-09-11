from dataReader import DataReader
from pathlib import Path
from utils import read_pickle_dump
import matplotlib.pyplot as plt

if __name__ == "__main__":
    dataFolder = Path(__file__).parent.parent / "data"
    # data handler
    vid_holder = DataReader(filename="pyroVideo.pkl", folder= dataFolder, reader=read_pickle_dump)
    video_data = 255 - vid_holder.read()
    print(f"raw data shape: {video_data.shape}")
    plt.imshow(video_data[500, :, :], cmap="hot")
    plt.show()
