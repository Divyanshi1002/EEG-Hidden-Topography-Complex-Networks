import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt

from networks.visibility_graph import compute_visibility_graph
from networks.network_metrics import compute_network_metrics
from networks.hub_classification import (
    within_module_degree_zscore,
    classify_node_roles
)


# ---------------- BANDPASS FILTER ----------------
def bandpass_filter(
    signal: np.ndarray,
    fs: int,
    lowcut: float,
    highcut: float,
    order: int = 4
) -> np.ndarray:
    """
    Apply Butterworth bandpass filter.
    """
    nyq = fs / 2
    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


# ---------------- FREQUENCY BANDS ----------------
FREQUENCY_BANDS = {
    "theta": (4.0, 7.5),
    "alpha": (8.0, 12.0),
    "beta": (13.0, 30.0)
}


# ---------------- CORE ANALYSIS ----------------
def analyze_frequency_band(
    epoch_signal: np.ndarray,
    fs: int,
    band: tuple
):
    """
    Compute hub roles for one epoch and one frequency band.
    """
    filtered_signal = bandpass_filter(
        epoch_signal, fs, band[0], band[1]
    )

    adj = compute_visibility_graph(filtered_signal)
    metrics = compute_network_metrics(adj)

    z = within_module_degree_zscore(
        adj, metrics["communities"]
    )

    roles = classify_node_roles(
        metrics["participation"], z
    )

    return roles


# ---------------- DATASET-LEVEL PIPELINE ----------------
def run_band_specific_network_analysis(
    input_root: Path,
    output_csv: Path,
    significant_channels: list,
    fs: int = 250
):
    """
    Frequency-specific hub analysis for significant channels only.

    Parameters
    ----------
    input_root : Path
        Epoch-level EEG directory:
        subject_X/channel_Y/epoch_Z.csv

    significant_channels : list
        List of channel folder names (e.g. ["channel_31", "channel_124"])

    Returns
    -------
    CSV with average R5, R6, R7 hubs per band and channel.
    """

    results = []

    for subject_dir in sorted(input_root.iterdir()):
        if not subject_dir.is_dir():
            continue

        for channel in significant_channels:
            channel_dir = subject_dir / channel
            if not channel_dir.exists():
                continue

            for epoch_file in channel_dir.glob("epoch_*.csv"):
                signal = pd.read_csv(
                    epoch_file, header=None
                ).iloc[:, 0].values

                for band_name, band_range in FREQUENCY_BANDS.items():
                    roles = analyze_frequency_band(
                        signal, fs, band_range
                    )

                    results.append({
                        "subject": subject_dir.name,
                        "channel": channel,
                        "epoch": epoch_file.stem,
                        "band": band_name,
                        "R5": roles.count("R5"),
                        "R6": roles.count("R6"),
                        "R7": roles.count("R7")
                    })

    df = pd.DataFrame(results)

    # Average across epochs and subjects (as in paper)
    summary = (
        df.groupby(["channel", "band"])
        [["R5", "R6", "R7"]]
        .mean()
        .reset_index()
    )

    summary.to_csv(output_csv, index=False)