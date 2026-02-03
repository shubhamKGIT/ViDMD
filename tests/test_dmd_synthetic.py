import unittest
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dmdBase import DmdBase
from dataGen import generate_synthetic_dmd_video


class _ArrayReader:
    def __init__(self, data, frame_shape):
        self._data = data
        self.data_shape = frame_shape

    def read(self):
        return self._data


class TestDmdSynthetic(unittest.TestCase):
    def test_dmd_recovers_eigs_modes_and_reconstruction(self):
        synth = generate_synthetic_dmd_video(
            frame_shape=(6, 5),
            timesteps=50,
            lambdas=np.array([0.9, 0.8]),
            amplitudes=np.array([2.0, -1.5]),
            seed=42,
        )

        reader = _ArrayReader(synth["data"], synth["frame_shape"])
        dmd = DmdBase(reader)
        dmd.update_state(
            data_read=False,
            columns_align_temporal=True,
            decomposed=False,
            modes_calculated=False,
        )
        dmd.prepare_data(dmd_type_temporal=True)
        dmd.decompose()
        dmd.calc_low_rank_eigvecs_modes(r=2)
        dmd.spectra(dt=1.0)
        dmd.coeffs()

        # Eigenvalues should match (order independent)
        est = np.sort(dmd.eigvals.real)
        true = np.sort(synth["lambdas"])
        self.assertTrue(np.allclose(est, true, atol=1e-8))

        # Modes should match up to scale and ordering
        est_modes = dmd.modes.real.copy()
        true_modes = synth["modes"].copy()
        est_modes /= np.linalg.norm(est_modes, axis=0, keepdims=True)
        true_modes /= np.linalg.norm(true_modes, axis=0, keepdims=True)
        corr = np.abs(true_modes.T @ est_modes)
        self.assertTrue(np.all(corr.max(axis=1) > 0.95))

        # Reconstruction should match the original snapshots (up to numerical error)
        t_space = np.arange(synth["data"].shape[1], dtype=float)
        x_recons = dmd.recons(T_space=t_space)
        err = np.linalg.norm(x_recons - synth["data"][:, :-1]) / np.linalg.norm(
            synth["data"][:, :-1]
        )
        self.assertLess(err, 1e-8)

    def test_dmd_recovers_complex_oscillatory_pair(self):
        # Real-valued oscillatory system with complex conjugate eigenvalues
        mag = 0.98
        theta = 0.25 * np.pi
        A = mag * np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        true_lambdas = np.array(
            [mag * (np.cos(theta) + 1j * np.sin(theta)), mag * (np.cos(theta) - 1j * np.sin(theta))]
        )
        timesteps = 60
        rng = np.random.default_rng(7)
        x0 = rng.standard_normal(2)
        X = np.zeros((2, timesteps))
        X[:, 0] = x0
        for k in range(1, timesteps):
            X[:, k] = A @ X[:, k - 1]

        # Embed into "video" pixels via spatial modes
        frame_shape = (5, 4)
        n_pixels = frame_shape[0] * frame_shape[1]
        phi_raw = rng.standard_normal((n_pixels, 2))
        phi, _ = np.linalg.qr(phi_raw)
        data = phi @ X

        reader = _ArrayReader(data, frame_shape)
        dmd = DmdBase(reader)
        dmd.update_state(
            data_read=False,
            columns_align_temporal=True,
            decomposed=False,
            modes_calculated=False,
        )
        dmd.prepare_data(dmd_type_temporal=True)
        dmd.decompose()
        dmd.calc_low_rank_eigvecs_modes(r=2)

        # Compare eigenvalues as an unordered set (complex conj pair)
        est = dmd.eigvals
        true = true_lambdas
        # Match by minimal distance
        d0 = min(np.abs(est[0] - true[0]), np.abs(est[0] - true[1]))
        d1 = min(np.abs(est[1] - true[0]), np.abs(est[1] - true[1]))
        self.assertLess(d0, 1e-6)
        self.assertLess(d1, 1e-6)


if __name__ == "__main__":
    unittest.main()
