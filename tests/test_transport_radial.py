import unittest
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transport_radial import RadialTransport, cartesian_to_polar


def synthetic_radial_wave(t=80, h=96, w=96, v_px=0.6, seed=0):
    rng = np.random.default_rng(seed)
    y = np.arange(h)
    x = np.arange(w)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    cx, cy = w / 2.0, h / 2.0
    video = np.zeros((t, h, w), dtype=float)
    r0 = 12.0
    for k in range(t):
        r = np.hypot(xx - cx, yy - cy)
        center = r0 + v_px * k
        ring = 1.2 * np.exp(-((r - center) ** 2) / (2 * (2.0**2)))
        wake = 0.3 * np.exp(-((r - (center - 5)) ** 2) / (2 * (4.0**2)))
        video[k] = ring + wake + 0.002 * rng.standard_normal((h, w))
    return video


class TestTransportRadial(unittest.TestCase):
    def test_radial_front_alignment(self):
        video = synthetic_radial_wave()
        mover = RadialTransport(
            video, n_theta=360, smooth_win=1, remove_mean=False, use_gradient=False
        )
        _ = mover.extract()

        # Check that front radius is stabilized in aligned polar domain
        radii = []
        for polar in mover.aligned_polar:
            prof = polar.mean(axis=0)
            win = 9
            kernel = np.ones(win) / win
            prof = np.convolve(prof, kernel, mode="same")
            idx = int(np.argmax(prof))
            radii.append(idx)
        radii = np.array(radii, dtype=float)
        self.assertLess(np.std(radii - np.median(radii)), 4.0)


if __name__ == "__main__":
    unittest.main()
