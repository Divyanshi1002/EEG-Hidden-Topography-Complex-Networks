import pandas as pd
from pathlib import Path


def split_into_epochs(
    input_root: Path,
    output_root: Path,
    fs: int = 250,
    epoch_duration: int = 10,
    total_samples: int = 75000
):
    """
    Split continuous EEG recordings into fixed-length epochs.

    Directory structure expected:
    input_root/
        subject_X/
            channel_Y.csv

    Output structure:
    output_root/
        subject_X/
            channel_Y/
                epoch_1.csv
                epoch_2.csv
                ...

    Parameters
    ----------
    fs : int
        Sampling frequency (Hz)
    epoch_duration : int
        Epoch length in seconds
    total_samples : int
        Number of samples to read from each channel
    """

    samples_per_epoch = fs * epoch_duration
    num_epochs = total_samples // samples_per_epoch

    for subject_dir in sorted(input_root.iterdir()):
        if not subject_dir.is_dir():
            continue

        subject_output_dir = output_root / subject_dir.name
        subject_output_dir.mkdir(parents=True, exist_ok=True)

        for channel_file in sorted(subject_dir.glob("channel_*.csv")):
            channel_id = channel_file.stem
            channel_output_dir = subject_output_dir / channel_id
            channel_output_dir.mkdir(exist_ok=True)

            eeg_data = pd.read_csv(
                channel_file,
                usecols=[0],
                nrows=total_samples,
                header=None
            )

            for epoch_idx in range(num_epochs):
                start = epoch_idx * samples_per_epoch
                end = (epoch_idx + 1) * samples_per_epoch

                epoch_data = eeg_data.iloc[start:end]
                epoch_file = channel_output_dir / f"epoch_{epoch_idx + 1}.csv"

                epoch_data.to_csv(
                    epoch_file,
                    index=False,
                    header=[channel_id]
                )
