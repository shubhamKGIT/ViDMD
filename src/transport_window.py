import numpy as np
from pathlib import Path


def _smooth_1d(x, win=9):
    if win <= 1:
        return x
    win = int(win)
    if win % 2 == 0:
        win += 1
    kernel = np.ones(win, dtype=float) / win
    return np.convolve(x, kernel, mode="same")


def _front_position_from_profile(profile, use_gradient=True):
    if use_gradient:
        g = np.gradient(profile)
        idx = int(np.argmax(np.abs(g)))
    else:
        idx = int(np.argmax(profile))
    return idx


def _profile_from_frame(frame, axis):
    # axis: 0 for y (vertical), 1 for x (horizontal)
    if axis == 0:
        return frame.mean(axis=1)
    return frame.mean(axis=0)


def _detect_axis(video, smooth_win=9):
    # video shape: (t, y, x)
    y_positions = []
    x_positions = []
    for frame in video:
        prof_y = _smooth_1d(_profile_from_frame(frame, axis=0), win=smooth_win)
        prof_x = _smooth_1d(_profile_from_frame(frame, axis=1), win=smooth_win)
        y_positions.append(_front_position_from_profile(prof_y))
        x_positions.append(_front_position_from_profile(prof_x))
    y_positions = np.array(y_positions, dtype=float)
    x_positions = np.array(x_positions, dtype=float)
    y_span = np.ptp(y_positions)
    x_span = np.ptp(x_positions)
    return ("y" if y_span >= x_span else "x"), y_positions, x_positions


def _interp_along_axis(frame, coords, axis):
    # Interpolate along axis for each line
    if axis == 0:
        y = np.arange(frame.shape[0])
        out = np.zeros((len(coords), frame.shape[1]), dtype=frame.dtype)
        for j in range(frame.shape[1]):
            out[:, j] = np.interp(coords, y, frame[:, j])
        return out
    x = np.arange(frame.shape[1])
    out = np.zeros((frame.shape[0], len(coords)), dtype=frame.dtype)
    for i in range(frame.shape[0]):
        out[i, :] = np.interp(coords, x, frame[i, :])
    return out


class MovingWindowTransport:
    """Front-tracking moving window to reduce translation artifacts."""

    def __init__(
        self,
        video,
        axis="auto",
        window_size=64,
        smooth_win=9,
        use_gradient=True,
        detrend=False,
        remove_mean=True,
        dual_mode=True,
        dual_ratio_thresh=1.5,
        min_span_px=3.0,
    ):
        self.video = video  # shape (t, y, x)
        self.axis = axis
        self.window_size = int(window_size)
        self.smooth_win = smooth_win
        self.use_gradient = use_gradient
        self.detrend = detrend
        self.remove_mean = remove_mean
        self.dual_mode = dual_mode
        self.dual_ratio_thresh = dual_ratio_thresh
        self.min_span_px = min_span_px
        self.positions = None
        self.axis_used = None
        self.dual_detected = False

    def detect_axis(self):
        axis, y_pos, x_pos = _detect_axis(self.video, smooth_win=self.smooth_win)
        self.positions = {"y": y_pos, "x": x_pos}
        y_span = np.ptp(y_pos)
        x_span = np.ptp(x_pos)
        if self.dual_mode and y_span > self.min_span_px and x_span > self.min_span_px:
            ratio = max(y_span, x_span) / (min(y_span, x_span) + 1e-12)
            if ratio < self.dual_ratio_thresh:
                self.dual_detected = True
        return axis

    def _track_positions(self, axis):
        positions = []
        for frame in self.video:
            prof = _smooth_1d(_profile_from_frame(frame, axis=axis), win=self.smooth_win)
            pos = _front_position_from_profile(prof, use_gradient=self.use_gradient)
            positions.append(pos)
        positions = np.array(positions, dtype=float)
        positions = _smooth_1d(positions, win=self.smooth_win)
        return positions

    def extract(self):
        if self.axis == "auto":
            axis = self.detect_axis()
        else:
            axis = self.axis
        self.axis_used = axis

        axis_idx = 0 if axis == "y" else 1
        positions = self._track_positions(axis_idx)
        self.positions = {axis: positions}

        half = (self.window_size - 1) / 2.0
        windows = []
        for t, frame in enumerate(self.video):
            center = positions[t]
            coords = center + (np.arange(self.window_size, dtype=float) - half)
            if axis_idx == 0:
                coords = np.clip(coords, 0, frame.shape[0] - 1)
                win = _interp_along_axis(frame, coords, axis=0)
            else:
                coords = np.clip(coords, 0, frame.shape[1] - 1)
                win = _interp_along_axis(frame, coords, axis=1)
            windows.append(win)
        windows = np.stack(windows, axis=0)

        if self.remove_mean:
            windows = windows - windows.mean(axis=0, keepdims=True)

        if self.detrend:
            # simple linear detrend per pixel over time
            t = np.arange(windows.shape[0], dtype=float)
            t = (t - t.mean()) / (t.std() + 1e-12)
            for i in range(windows.shape[1]):
                for j in range(windows.shape[2]):
                    y = windows[:, i, j]
                    coef = np.polyfit(t, y, 1)
                    windows[:, i, j] = y - (coef[0] * t + coef[1])

        return windows

    def save(self, out_path: Path, windows: np.ndarray):
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, windows)
        return out_path

    def save_metadata(self, out_path: Path):
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_path,
            axis_used=self.axis_used,
            dual_detected=self.dual_detected,
            positions=self.positions,
        )
        return out_path
