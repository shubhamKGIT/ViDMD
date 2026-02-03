import argparse
from pathlib import Path
import numpy as np
import pickle

from transport_window import MovingWindowTransport
from transport_radial import RadialTransport
from dmdBase import DmdBase
from utils import read_pickle_dumped_array


class _ArrayReader:
    def __init__(self, data, frame_shape):
        self._data = data
        self.data_shape = frame_shape

    def read(self):
        return self._data


def flatten_video(video):
    # video shape: (t, y, x) -> (y*x, t)
    t, h, w = video.shape
    return video.reshape(t, h * w).T, (h, w)


def main():
    parser = argparse.ArgumentParser(description="Transport DMD pipeline with moving window.")
    parser.add_argument("--video_pkl", type=str, required=True, help="Pickle video file path")
    parser.add_argument("--out_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--transport", type=str, default="linear", choices=["linear", "radial"])
    parser.add_argument("--window_size", type=int, default=64)
    parser.add_argument("--axis", type=str, default="auto", choices=["auto", "x", "y"])
    parser.add_argument("--smooth_win", type=int, default=9)
    parser.add_argument("--n_theta", type=int, default=360)
    parser.add_argument("--center", type=str, default=None, help="Radial center as 'x,y' in pixels")
    parser.add_argument("--use_gradient", action="store_true", help="Use gradient peak for radial front")
    parser.add_argument("--r", type=int, default=20)
    parser.add_argument("--dt", type=float, default=0.001)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    video = read_pickle_dumped_array(Path(args.video_pkl))

    if args.transport == "linear":
        mover = MovingWindowTransport(
            video,
            axis=args.axis,
            window_size=args.window_size,
            smooth_win=args.smooth_win,
            remove_mean=True,
            detrend=False,
            dual_mode=True,
        )
        windows = mover.extract()
        mover.save(out_dir / "front_window.npy", windows)
        mover.save_metadata(out_dir / "front_window_meta.npz")
        data, frame_shape = flatten_video(windows)
        dmd_prefix = "front_window"
    else:
        center = None
        if args.center:
            cx, cy = [float(v.strip()) for v in args.center.split(",")]
            center = (cx, cy)
        mover = RadialTransport(
            video,
            center=center,
            n_theta=args.n_theta,
            smooth_win=args.smooth_win,
            use_gradient=args.use_gradient,
            remove_mean=True,
            detrend=False,
        )
        aligned = mover.extract()
        np.save(out_dir / "radial_aligned.npy", aligned)
        data, frame_shape = flatten_video(aligned)
        dmd_prefix = "radial_aligned"

    reader = _ArrayReader(data, frame_shape)
    dmd = DmdBase(reader)
    dmd.update_state(False, True, False, False)
    dmd.prepare_data(True)
    dmd.decompose()
    dmd.calc_low_rank_eigvecs_modes(r=args.r)
    dmd.spectra(dt=args.dt)
    dmd.coeffs()

    with open(out_dir / f"{dmd_prefix}_dmdObj.pkl", "wb") as f:
        pickle.dump(dmd, f)

    dmd.eigvals.dump(out_dir / f"{dmd_prefix}_eigvals.pkl")
    dmd.omega.dump(out_dir / f"{dmd_prefix}_omega.pkl")
    dmd.modes.dump(out_dir / f"{dmd_prefix}_modes.pkl")
    dmd.E.dump(out_dir / f"{dmd_prefix}_singularVals.pkl")


if __name__ == "__main__":
    main()
