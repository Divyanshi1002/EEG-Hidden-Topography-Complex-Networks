import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import welch
import matplotlib.pyplot as plt


def compute_psd(
    signal: np.ndarray,
    fs: int = 250,
    nperseg: int = 1024,
    noverlap: int = 512
):
    """
    Compute Power Spectral Density (PSD) using Welch's method.
    """
    freqs, psd = welch(
        signal,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="density"
    )
    psd_db = 10 * np.log10(psd + 1e-12)
    return freqs, psd, psd_db


def run_psd_analysis(
    input_root: Path,
    output_root: Path,
    fs: int = 250
):
    """
    Compute PSD for all subjects and channels.
    """
    output_root.mkdir(parents=True, exist_ok=True)

    for subject_dir in sorted(input_root.iterdir()):
        if not subject_dir.is_dir():
            continue

        subject_out = output_root / subject_dir.name
        subject_out.mkdir(exist_ok=True)

        psd_for_plot = []

        for channel_file in sorted(subject_dir.glob("channel_*.csv")):
            channel_id = channel_file.stem.split("_")[1]
            signal = pd.read_csv(channel_file, header=None).iloc[:, 0].values

            freqs, psd, psd_db = compute_psd(signal, fs)

            pd.DataFrame({
                "frequency": freqs,
                "psd": psd,
                "psd_db": psd_db
            }).to_csv(
                subject_out / f"channel_{channel_id}_psd.csv",
                index=False
            )

            psd_for_plot.append((channel_id, freqs, psd_db))

        plot_psd_summary(
            psd_for_plot,
            subject_out / f"{subject_dir.name}_psd.png"
        )


