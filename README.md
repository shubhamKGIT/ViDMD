# ViDMD
Koopman-operator-based decomposition of experimental video data, built from scratch for combustion time-series imaging.

## Overview
ViDMD provides a Dynamic Mode Decomposition (DMD) pipeline tailored for video-based flame dynamics. It includes:
- Core DMD implementation (SVD, low-rank eigendecomposition, modes, spectra).
- Video readers and preprocessing helpers.
- Transport DMD utilities for traveling fronts (linear moving-window and radial spherical fronts).
- Analysis and visualization helpers for interpreting eigenvalues and modes.

## Key Features
### Core DMD
- `DmdBase` with snapshot SVD, reduced-order eigensystem, and reconstruction.
- Continuous-time spectra (`omega`) and modal coefficients (`b`).

### Transport DMD (Traveling Fronts)
- **Linear moving-window transport**: tracks a front along x/y, aligns a 64px window, and reduces translation artifacts.
- **Radial transport**: aligns spherical fronts using polar coordinates (Schlieren-like views).
- Dual-mode detection and metadata export for mixed-direction motion.

### Analysis Helpers
- Categorize modes by physics (background, stable, growing, decaying, oscillatory).
- Cluster complex-conjugate eigenvalue pairs.
- Optional translation-like mode detection and exclusion.

## Repository Structure
### `src/`
- `dmdBase.py`: DMD math and reconstruction.
- `dataReader.py`: Data and video readers.
- `dmdVisualiser.py`: Plotting utilities for spectra and modes.
- `transport_window.py`: Linear moving-window transport.
- `transport_radial.py`: Radial front alignment.
- `analysis_helper.py`: Mode categorization and summaries.
- `main_transport_dmd.py`: End-to-end transport DMD pipeline.

### `tests/`
- Synthetic DMD validation (real + complex eigenpairs).
- Moving-window transport stabilization test.
- Radial transport alignment test.

## What Was Added Recently
- Moving-window transport with axis auto-detection and dual-mode flagging.
- Radial transport for spherical traveling fronts.
- Translation-like mode detection and exclusion option.
- Transport DMD pipeline CLI (`main_transport_dmd.py`).
- Synthetic verification tests for linear and radial transport.
- Reproducible command list in `COMMANDS.md`.

## Notes
If you need a more established DMD implementation, consider PyDMD. This repo focuses on extensibility for combustion video workflows, with transport alignment and domain-specific interpretation layers.
