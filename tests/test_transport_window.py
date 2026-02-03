import unittest
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transport_window import MovingWindowTransport


def synthetic_front_video(t=80, h=64, w=48, v_px=0.5, seed=0):
    rng = np.random.default_rng(seed)
    y = np.arange(h)
    x = np.arange(w)
    X, Y = np.meshgrid(x, y)
    video = np.zeros((t, h, w), dtype=float)
    y0 = h * 0.3
    for k in range(t):
        center = y0 + v_px * k
        front = np.exp(-((Y - center) ** 2) / (2 * (3.0**2)))
        wake = 0.3 * np.exp(-((Y - (center - 6)) ** 2) / (2 * (6.0**2)))
        blob = 0.15 * np.exp(-((X - (w * 0.6 + 2 * np.sin(0.1 * k))) ** 2) / (2 * (4.0**2)))
        video[k] = front + wake + blob + 0.02 * rng.standard_normal((h, w))
    return video


class TestTransportWindow(unittest.TestCase):
    def test_window_stabilizes_front_position(self):
        video = synthetic_front_video()
        mover = MovingWindowTransport(video, axis="y", window_size=32, smooth_win=7)
        windows = mover.extract()

        # Recompute front positions inside window: should be near center
        center_expected = windows.shape[1] / 2.0
        positions = []
        for frame in windows:
            prof = frame.mean(axis=1)
            pos = int(np.argmax(np.abs(np.gradient(prof))))
            positions.append(pos)
        positions = np.array(positions, dtype=float)
        self.assertLess(np.std(positions - center_expected), 1.5)


if __name__ == "__main__":
    unittest.main()
