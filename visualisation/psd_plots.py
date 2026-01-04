import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_group_average_psd(
    base_psd_dir: Path,
    groups: list,
    selected_channels: list,
    fs_range=(1, 30),
    colors=None
):
    """
    Plot group-averaged PSD with variability shading.
    """
    if colors is None:
        colors = {"mdd": "red", "normal": "blue"}

    # Load reference frequency axis
    ref = next(base_psd_dir.rglob("chan_*_psd.csv"))
    freqs = pd.read_csv(ref)["frequency"].values
    mask = (freqs >= fs_range[0]) & (freqs <= fs_range[1])
    freqs = freqs[mask]

    plt.figure(figsize=(8, 6))

    for g in groups:
        subject_curves = []

        for subj_dir in (base_psd_dir / g).iterdir():
            chan_curves = []

            for ch in selected_channels:
                fn = subj_dir / f"chan_{ch}_psd.csv"
                if not fn.exists():
                    continue

                psd = pd.read_csv(fn)["power_density"].values
                chan_curves.append(psd[mask])

            if chan_curves:
                subject_curves.append(
                    np.mean(np.stack(chan_curves), axis=0)
                )

        arr = np.stack(subject_curves)
        mean_ = arr.mean(axis=0)
        lo, hi = np.percentile(arr, [25, 75], axis=0)

        plt.plot(freqs, mean_, label=g.capitalize(), color=colors[g])
        plt.fill_between(freqs, lo, hi, color=colors[g], alpha=0.3)

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Density (µV²/Hz)")
    plt.title("Group-Averaged PSD (1–30 Hz)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
