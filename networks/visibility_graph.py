import numpy as np
import pandas as pd
import multiprocessing as mp
from pathlib import Path
import shutil


# ---------- Core NVG ----------
def compute_visibility_graph(time_series: np.ndarray) -> np.ndarray:
    """
    Optimized Natural Visibility Graph (NVG) using max-slope criterion.
    """
    N = len(time_series)
    adj = np.zeros((N, N), dtype=int)

    for i in range(N - 1):
        max_slope = float("-inf")
        for j in range(i + 1, N):
            slope_ij = (time_series[j] - time_series[i]) / (j - i)

            if j > i + 1:
                prev_slope = (time_series[j - 1] - time_series[i]) / (j - 1 - i)
                max_slope = max(max_slope, prev_slope)

            if max_slope <= slope_ij:
                adj[i, j] = adj[j, i] = 1

    return adj


# ---------- Epoch-level ----------
def load_epoch(file_path: Path) -> np.ndarray:
    """
    Load a single EEG epoch (1D signal).
    """
    return pd.read_csv(file_path, header=None, skiprows=1).iloc[:, 0].values


def process_epoch(args):
    """
    Atomic processing unit:
    One epoch → Visibility Graph → Adjacency matrix
    """
    epoch_file, output_file = args
    signal = load_epoch(epoch_file)
    adj = compute_visibility_graph(signal)
    np.savetxt(output_file, adj, delimiter=",", fmt="%d")


# ---------- Channel-level ----------
def process_channel(channel_input_dir: Path, channel_output_dir: Path):
    """
    Process all epochs of a single channel.
    Epochs are parallelized.
    """
    channel_output_dir.mkdir(parents=True, exist_ok=True)

    epoch_files = sorted(
        channel_input_dir.glob("*.csv"),
        key=lambda x: int(x.stem.split("_")[-1])
    )

    tasks = [
        (epoch_file, channel_output_dir / f"vg_{epoch_file.name}")
        for epoch_file in epoch_files
    ]

    with mp.Pool(processes=max(1, mp.cpu_count() - 2)) as pool:
        pool.map(process_epoch, tasks)


# ---------- Subject-level ----------
def process_subject(subject_input_dir: Path, subject_output_dir: Path):
    """
    Process all channels of a subject.
    """
    subject_output_dir.mkdir(exist_ok=True)

    channel_dirs = sorted(
        subject_input_dir.iterdir(),
        key=lambda x: int(x.name.split("_")[-1])
    )

    for channel_dir in channel_dirs:
        if channel_dir.is_dir():
            process_channel(
                channel_dir,
                subject_output_dir / channel_dir.name
            )


# ---------- Pipeline ----------
def run_visibility_graph_pipeline(input_root: Path, output_root: Path):
    """
    Full NVG pipeline with hierarchy:
    Epochs → Channels → Subjects

    (Epochs are the fundamental computational unit)
    """
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    subject_dirs = sorted(input_root.iterdir())

    for subject_dir in subject_dirs:
        if subject_dir.is_dir():
            process_subject(
                subject_dir,
                output_root / subject_dir.name
            )
