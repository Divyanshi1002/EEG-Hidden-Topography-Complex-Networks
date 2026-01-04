import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import firwin, filtfilt


def bandpass_filter(
    signal: np.ndarray,
    fs: int,
    lowcut: float = 1.0,
    highcut: float = 30.0,
    numtaps: int = 401
) -> np.ndarray:
    """
    Apply FIR bandpass filter to EEG signal.

    Parameters
    ----------
    signal : np.ndarray
        1D EEG signal
    fs : int
        Sampling frequency (Hz)
    lowcut : float
        Lower cutoff frequency (Hz)
    highcut : float
        Upper cutoff frequency (Hz)
    numtaps : int
        Filter order (should be odd)

    Returns
    -------
    np.ndarray
        Bandpass filtered signal
    """
    nyquist = fs / 2
    taps = firwin(
        numtaps,
        [lowcut / nyquist, highcut / nyquist],
        pass_zero=False
    )
    return filtfilt(taps, [1.0], signal)


def apply_bandpass_to_dataset(
    input_root: Path,
    output_root: Path,
    fs: int = 250
):
    """
    Apply bandpass filter to all subjects and channels.
    """
    output_root.mkdir(parents=True, exist_ok=True)

    for subject_dir in sorted(input_root.iterdir()):
        if not subject_dir.is_dir():
            continue

        subject_out = output_root / subject_dir.name
        subject_out.mkdir(exist_ok=True)

        for channel_file in sorted(subject_dir.glob("channel_*.csv")):
            signal = pd.read_csv(channel_file, header=None).iloc[:, 0].values
            filtered = bandpass_filter(signal, fs)

            out_file = subject_out / channel_file.name
            pd.DataFrame(filtered).to_csv(out_file, index=False, header=False)
