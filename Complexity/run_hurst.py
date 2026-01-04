import numpy as np
import pandas as pd
from pathlib import Path
from Complexity.hurst_rs_analysis import hurst_rs_multiscale


def compute_hurst_for_dataset(input_root: Path, output_csv: Path):
    """
    Compute Hurst exponent for all subjects and channels.
    """
    results = []

    for subject_dir in sorted(input_root.iterdir()):
        if not subject_dir.is_dir():
            continue

        subject_id = subject_dir.name

        for channel_dir in sorted(subject_dir.iterdir()):
            if not channel_dir.is_dir():
                continue

            channel_id = channel_dir.name

            for epoch_file in channel_dir.glob("epoch_*.csv"):
                signal = pd.read_csv(epoch_file, header=None).iloc[:, 0].values
                H = hurst_rs_multiscale(signal)

                results.append({
                    "subject": subject_id,
                    "channel": channel_id,
                    "epoch": epoch_file.stem,
                    "hurst": H
                })

    pd.DataFrame(results).to_csv(output_csv, index=False)
