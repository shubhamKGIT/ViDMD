# Commands

This file collects the key commands used in this project so runs can be reproduced quickly.

## 1. Linear (moving-window) transport DMD

Purpose: Track a traveling front along x/y, crop a 64px moving window, and run DMD on the stabilized window.

Command:
```bash
python src/main_transport_dmd.py \
  --video_pkl data/pyroVideo02.pkl \
  --transport linear \
  --window_size 64 \
  --axis auto \
  --smooth_win 9 \
  --r 20 \
  --dt 0.001 \
  --out_dir results
```

Expected outputs:
- `results/front_window.npy` (windowed video)
- `results/front_window_meta.npz` (axis + positions + dual-mode flag)
- `results/front_window_dmdObj.pkl` (DMD object)
- `results/front_window_{eigvals,omega,modes,singularVals}.pkl`

## 2. Radial transport DMD

Purpose: Align a spherical front radially in a Schlieren-like video and run DMD in the stabilized frame.

Command:
```bash
python src/main_transport_dmd.py \
  --video_pkl data/pyroVideo02.pkl \
  --transport radial \
  --n_theta 360 \
  --smooth_win 9 \
  --use_gradient \
  --r 20 \
  --dt 0.001 \
  --out_dir results
```

Optional center override (pixel coordinates):
```bash
--center 512,512
```

Expected outputs:
- `results/radial_aligned.npy` (radially-aligned video)
- `results/radial_aligned_dmdObj.pkl` (DMD object)
- `results/radial_aligned_{eigvals,omega,modes,singularVals}.pkl`

## 3. Mode analysis and reconstruction summary

Purpose: Cluster conjugate eigenvalue pairs, categorize modes by physics labels, rank by amplitude/decay, and make a reconstruction comparison.

Command:
```bash
python src/analysis_helper.py \
  --dmd_obj results/front_window_dmdObj.pkl \
  --fps 1000 \
  --pixel_size_m 17e-6 \
  --front_speed_cm_s 1 3 \
  --axis y \
  --exclude_translation \
  --frames 200,500,800
```

Expected outputs:
- `results/dmd_omega_summary.png` (category plot in omega plane)
- `results/dmd_mode_rankings.png` (amplitude and decay rankings)
- `results/dmd_reconstruction_compare.png` (side-by-side frame comparison)
- `results/dmd_traveling_front_summary.txt`

## 4. Validation tests (synthetic data)

Purpose: Verify that DMD recovers known eigenvalues, modes, and reconstruction (real + complex cases).

Command:
```bash
python -m unittest tests/test_dmd_synthetic.py
```

Expected outcome:
- All tests pass (OK)

## 5. Moving-window transport test (synthetic)

Purpose: Verify that the moving window stabilizes the front position.

Command:
```bash
python -m unittest tests/test_transport_window.py
```

Expected outcome:
- Test passes (OK)

## 6. Radial transport test (synthetic)

Purpose: Verify radial alignment for a spherical traveling wave.

Command:
```bash
python -m unittest tests/test_transport_radial.py
```

Expected outcome:
- Test passes (OK)
