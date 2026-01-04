import numpy as np
from scipy.stats import linregress


def hurst_rs_multiscale(time_series: np.ndarray) -> float:
    """
    Compute Hurst exponent using multi-scale Rescaled Range (R/S) analysis.

    This implementation follows standard R/S methodology
    used in EEG time-series analysis.

    Parameters
    ----------
    time_series : np.ndarray
        1D EEG signal

    Returns
    -------
    float
        Hurst exponent
    """
    N = len(time_series)
    window_sizes = np.floor(
        np.logspace(np.log10(10), np.log10(N // 2), num=10)
    ).astype(int)

    RS = []

    for w in window_sizes:
        if w < 10:
            continue

        num_segments = N // w
        rs_vals = []

        for i in range(num_segments):
            segment = time_series[i * w:(i + 1) * w]
            mean_seg = np.mean(segment)
            dev = segment - mean_seg
            cum_dev = np.cumsum(dev)

            R = np.max(cum_dev) - np.min(cum_dev)
            S = np.std(segment, ddof=1)

            if S > 0:
                rs_vals.append(R / S)

        if len(rs_vals) > 0:
            RS.append(np.mean(rs_vals))

    log_w = np.log(window_sizes[:len(RS)])
    log_RS = np.log(RS)

    slope, _, _, _, _ = linregress(log_w, log_RS)
    return slope
