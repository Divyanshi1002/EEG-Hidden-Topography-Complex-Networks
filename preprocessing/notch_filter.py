import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import iirnotch, filtfilt


def notch_filter(
    signal: np.ndarray,
    fs: int,
    notch_freq: float = 50.0,
    quality_factor: float = 30.0
) -> np.ndarray:
    """
    Apply IIR notch filter to remove powerline noise.

    Parameters
    ----------
    signal : np.ndarray
        1D EEG signal
    fs : int
        Sampling frequency (Hz)
    notch_freq : float
        Powerline frequency (Hz)
    quality_factor : float
        Controls notch bandwidth

    Returns
    -------
    np.ndarray
        Notch filtered signal
    """
    nyquist = fs / 2
    w0 = notch_freq / nyquist
    b, a = iirnotch(w0, quality_factor)
    return filtfilt(b, a, signal)


def apply_notch_to_dataset(
    input_root: Path,
    output_root: Path,
    fs: int = 250
):
    """
    Apply 50 Hz notch filter to all subjects and channels.
    """
    output_root.mkdir(parents=True, exist_ok=True)

    for subject_dir in sorted(input_root.iterdir()):
        if not subject_dir.is_dir():
            continue

        subject_out = output_root / subject_dir.name
        subject_out.mkdir(exist_ok=True)

        for channel_file in sorted(subject_dir.glob("channel_*.csv")):
            signal = pd.read_csv(channel_file, header=None).iloc[:, 0].values
            filtered = notch_filter(signal, fs)

            out_file = subject_out / channel_file.name
            pd.DataFrame(filtered).to_csv(out_file, index=False, header=False)
