import argparse
from pathlib import Path
import numpy as np
import pickle
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_dmd_object(dmd_path: Path):
    with open(dmd_path, "rb") as f:
        return pickle.load(f)


def classify_modes(omega, b, eps_omega=1e-3, eps_real=1e-3, eps_imag=1e-6):
    amp = np.abs(b)
    omega_mag = np.abs(omega)
    categories = {
        "background": omega_mag < eps_omega,
        "oscillatory": np.abs(omega.imag) > eps_imag,
        "growing": omega.real > eps_real,
        "decaying": omega.real < -eps_real,
        "stable": np.abs(omega.real) <= eps_real,
    }
    return amp, categories


def find_conjugate_pairs(omega, tol=1e-6):
    pairs = []
    used = set()
    for i, oi in enumerate(omega):
        if i in used:
            continue
        conj = np.conj(oi)
        j = None
        for k, ok in enumerate(omega):
            if k == i or k in used:
                continue
            if np.abs(ok - conj) < tol:
                j = k
                break
        if j is not None:
            pairs.append((i, j))
            used.add(i)
            used.add(j)
    return pairs


def detect_translation_like_modes(modes, mean_frame, axis, omega, eps_imag=1e-4, corr_thresh=0.6):
    h, w = mean_frame.shape
    if axis == "y":
        template = np.gradient(mean_frame, axis=0)
    else:
        template = np.gradient(mean_frame, axis=1)
    tmpl = template.reshape(-1)
    tmpl = tmpl / (np.linalg.norm(tmpl) + 1e-12)
    flags = np.zeros(modes.shape[1], dtype=bool)
    for i in range(modes.shape[1]):
        if np.abs(omega.imag[i]) > eps_imag:
            continue
        m = modes[:, i].real
        m = m / (np.linalg.norm(m) + 1e-12)
        corr = np.abs(np.dot(m, tmpl))
        if corr >= corr_thresh:
            flags[i] = True
    return flags


def plot_omega_summary(out_path, omega, b, categories, pairs):
    amp = np.abs(b)
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {
        "background": "#6c757d",
        "oscillatory": "#1f77b4",
        "growing": "#d62728",
        "decaying": "#2ca02c",
        "stable": "#9467bd",
    }

    # Base scatter
    ax.scatter(omega.real, omega.imag, s=20 + 80 * (amp / (amp.max() + 1e-12)), c="#cccccc")

    # Overlay categories
    for name, mask in categories.items():
        ax.scatter(
            omega.real[mask],
            omega.imag[mask],
            s=30 + 90 * (amp[mask] / (amp.max() + 1e-12)),
            c=colors[name],
            label=name,
            alpha=0.8,
        )

    # Annotate conjugate pairs
    for idx, (i, j) in enumerate(pairs):
        ax.annotate(f"P{idx+1}", (omega.real[i], omega.imag[i]), fontsize=9, color="black")
        ax.annotate(f"P{idx+1}", (omega.real[j], omega.imag[j]), fontsize=9, color="black")

    ax.set_xlabel("Re(omega)")
    ax.set_ylabel("Im(omega)")
    ax.set_title("DMD Eigenvalues (omega) with Categories")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_rankings(out_path, omega, b, categories, top_n=10):
    amp = np.abs(b)
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))

    # Amplitude ranking by category
    ax = axes[0]
    for name, mask in categories.items():
        idx = np.where(mask)[0]
        if idx.size == 0:
            continue
        order = idx[np.argsort(-amp[idx])][:top_n]
        ax.plot(amp[order], label=name, marker="o")
    ax.set_title("Top Amplitudes by Category")
    ax.set_xlabel("Rank")
    ax.set_ylabel("|b|")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Decay ranking by category (most negative Re first)
    ax = axes[1]
    for name, mask in categories.items():
        idx = np.where(mask)[0]
        if idx.size == 0:
            continue
        order = idx[np.argsort(omega.real[idx])][:top_n]
        ax.plot(omega.real[order], label=name, marker="o")
    ax.set_title("Top Decay (Most Negative Re) by Category")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Re(omega)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def reconstruct_from_modes(modes, omega, b, t_space, mode_idx):
    modes_sel = modes[:, mode_idx]
    omega_sel = omega[mode_idx]
    b_sel = b[mode_idx]
    dynamics = np.zeros((len(mode_idx), len(t_space) - 1), dtype=complex)
    for i, t in enumerate(t_space[:-1]):
        dynamics[:, i] = np.diag(b_sel) @ np.exp(omega_sel * t)
    x_rec = modes_sel @ dynamics
    return x_rec.real


def plot_reconstruction_compare(out_path, dmd_obj, t_space, frame_nums, mode_idx):
    data = dmd_obj.data
    h, w = dmd_obj.dataObj.data_shape
    x_rec = reconstruct_from_modes(dmd_obj.modes, dmd_obj.omega, dmd_obj.b, t_space, mode_idx)

    n = len(frame_nums)
    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(6, 3 * n))
    if n == 1:
        axes = np.array([axes])
    for row, frame in enumerate(frame_nums):
        axes[row, 0].imshow(data[:, frame].real.reshape(h, w), cmap="hot")
        axes[row, 0].set_title(f"Original frame {frame}")
        axes[row, 0].axis("off")
        axes[row, 1].imshow(x_rec[:, frame].real.reshape(h, w), cmap="hot")
        axes[row, 1].set_title(f"Reconstruction frame {frame}")
        axes[row, 1].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_summary_text(out_path, fps, pixel_size_m, front_speed_cm_s):
    dt = 1.0 / fps
    speed_m_s = np.array(front_speed_cm_s) / 100.0
    px_per_s = speed_m_s / pixel_size_m
    px_per_frame = px_per_s * dt
    with open(out_path, "w") as f:
        f.write("DMD Traveling Front Summary\n")
        f.write("===========================\n\n")
        f.write(f"Frame rate: {fps:.1f} fps (dt={dt:.6f} s)\n")
        f.write(f"Pixel size: {pixel_size_m:.2e} m/pixel\n")
        f.write(
            f"Front speed range: {front_speed_cm_s[0]:.2f}-{front_speed_cm_s[1]:.2f} cm/s\n"
        )
        f.write(
            f"Expected front motion: {px_per_frame[0]:.3f}-{px_per_frame[1]:.3f} px/frame\n"
        )


def main():
    parser = argparse.ArgumentParser(description="Analyze DMD modes for traveling fronts.")
    parser.add_argument("--dmd_obj", type=str, required=True, help="Path to saved DMD object .pkl")
    parser.add_argument("--out_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--fps", type=float, default=1000.0, help="Frame rate (fps)")
    parser.add_argument("--pixel_size_m", type=float, default=17e-6, help="Pixel size (m)")
    parser.add_argument("--front_speed_cm_s", type=float, nargs=2, default=[1.0, 3.0])
    parser.add_argument("--frames", type=str, default="200,500,800", help="Comma-separated frame numbers")
    parser.add_argument("--top_n", type=int, default=10, help="Top N to rank per category")
    parser.add_argument("--axis", type=str, default="y", choices=["x", "y"])
    parser.add_argument("--exclude_translation", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dmd_obj = load_dmd_object(Path(args.dmd_obj))

    omega = dmd_obj.omega
    b = dmd_obj.b
    amp, categories = classify_modes(omega, b)
    pairs = find_conjugate_pairs(omega)

    plot_omega_summary(out_dir / "dmd_omega_summary.png", omega, b, categories, pairs)
    plot_rankings(out_dir / "dmd_mode_rankings.png", omega, b, categories, top_n=args.top_n)
    save_summary_text(
        out_dir / "dmd_traveling_front_summary.txt",
        args.fps,
        args.pixel_size_m,
        args.front_speed_cm_s,
    )

    # Pick top oscillatory modes by amplitude for reconstruction comparison
    osc_idx = np.where(categories["oscillatory"])[0]
    exclude = np.zeros_like(osc_idx, dtype=bool)
    if args.exclude_translation:
        mean_frame = dmd_obj.data.mean(axis=1).reshape(dmd_obj.dataObj.data_shape)
        trans_flags = detect_translation_like_modes(
            dmd_obj.modes, mean_frame, args.axis, omega
        )
        osc_idx = np.array([i for i in osc_idx if not trans_flags[i]])
    if osc_idx.size > 0:
        osc_order = osc_idx[np.argsort(-amp[osc_idx])]
        mode_idx = osc_order[: min(4, len(osc_order))]
    else:
        mode_idx = np.argsort(-amp)[:4]

    frames = [int(x.strip()) for x in args.frames.split(",") if x.strip()]
    t_space = np.arange(dmd_obj.data.shape[1], dtype=float) / args.fps
    plot_reconstruction_compare(
        out_dir / "dmd_reconstruction_compare.png", dmd_obj, t_space, frames, mode_idx
    )


if __name__ == "__main__":
    main()
