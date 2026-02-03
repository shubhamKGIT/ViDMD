import numpy as np


def _smooth_1d(x, win=9):
    if win <= 1:
        return x
    win = int(win)
    if win % 2 == 0:
        win += 1
    kernel = np.ones(win, dtype=float) / win
    return np.convolve(x, kernel, mode="same")


def _bilinear_sample(img, xs, ys):
    h, w = img.shape
    x0 = np.floor(xs).astype(int)
    y0 = np.floor(ys).astype(int)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x0 = np.clip(x0, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)

    wx = xs - x0
    wy = ys - y0

    Ia = img[y0, x0]
    Ib = img[y0, x1]
    Ic = img[y1, x0]
    Id = img[y1, x1]

    return (Ia * (1 - wx) * (1 - wy) +
            Ib * wx * (1 - wy) +
            Ic * (1 - wx) * wy +
            Id * wx * wy)


def cartesian_to_polar(img, center=None, n_r=None, n_theta=360):
    h, w = img.shape
    if center is None:
        center = (w / 2.0, h / 2.0)
    cx, cy = center
    if n_r is None:
        n_r = int(np.hypot(max(cx, w - cx), max(cy, h - cy)))
    r = np.linspace(0, n_r - 1, n_r)
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)

    rr, tt = np.meshgrid(r, theta, indexing="xy")
    xs = cx + rr * np.cos(tt)
    ys = cy + rr * np.sin(tt)
    polar = _bilinear_sample(img, xs, ys)
    return polar, r, theta


def polar_to_cartesian(polar, shape, center=None):
    h, w = shape
    if center is None:
        center = (w / 2.0, h / 2.0)
    cx, cy = center
    n_theta, n_r = polar.shape

    y = np.arange(h)
    x = np.arange(w)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    dx = xx - cx
    dy = yy - cy
    r = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx)
    theta = np.mod(theta, 2 * np.pi)

    r_idx = np.clip(r, 0, n_r - 1)
    t_idx = theta / (2 * np.pi) * n_theta
    t_idx = np.clip(t_idx, 0, n_theta - 1)

    return _bilinear_sample(polar, r_idx, t_idx)


class RadialTransport:
    """Radial front alignment for spherical traveling waves."""

    def __init__(
        self,
        video,
        center=None,
        n_theta=360,
        smooth_win=9,
        use_gradient=True,
        remove_mean=True,
        detrend=False,
    ):
        self.video = video  # (t, y, x)
        self.center = center
        self.n_theta = n_theta
        self.smooth_win = smooth_win
        self.use_gradient = use_gradient
        self.remove_mean = remove_mean
        self.detrend = detrend
        self.positions = None
        self.r_target = None

    def _front_radius_from_profile(self, prof):
        if self.use_gradient:
            g = np.gradient(prof)
            idx = int(np.argmax(np.abs(g)))
        else:
            idx = int(np.argmax(prof))
        return idx

    def extract(self):
        radii = []
        polars = []
        for frame in self.video:
            polar, r, _ = cartesian_to_polar(
                frame, center=self.center, n_theta=self.n_theta
            )
            prof = polar.mean(axis=0)
            prof = _smooth_1d(prof, win=self.smooth_win)
            radii.append(self._front_radius_from_profile(prof))
            polars.append(polar)

        radii = np.array(radii, dtype=float)
        radii = _smooth_1d(radii, win=self.smooth_win)
        self.positions = radii
        self.r_target = int(np.median(radii))

        aligned = []
        for i, polar in enumerate(polars):
            shift = self.r_target - radii[i]
            rr = np.arange(polar.shape[1], dtype=float) - shift
            rr = np.clip(rr, 0, polar.shape[1] - 1)
            # interpolate along radius for each theta
            shifted = np.zeros_like(polar)
            for t in range(polar.shape[0]):
                shifted[t, :] = np.interp(rr, np.arange(polar.shape[1]), polar[t, :])
            aligned.append(shifted)

        aligned = np.stack(aligned, axis=0)
        self.aligned_polar = aligned
        if self.remove_mean:
            aligned = aligned - aligned.mean(axis=0, keepdims=True)

        if self.detrend:
            t = np.arange(aligned.shape[0], dtype=float)
            t = (t - t.mean()) / (t.std() + 1e-12)
            for i in range(aligned.shape[1]):
                for j in range(aligned.shape[2]):
                    y = aligned[:, i, j]
                    coef = np.polyfit(t, y, 1)
                    aligned[:, i, j] = y - (coef[0] * t + coef[1])

        # back to cartesian
        out = []
        for polar in aligned:
            out.append(polar_to_cartesian(polar, self.video.shape[1:], center=self.center))
        return np.stack(out, axis=0)
