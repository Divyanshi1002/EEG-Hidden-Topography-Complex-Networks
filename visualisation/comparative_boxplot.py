import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns


def load_and_average_metric(
    base_path: Path,
    groups,
    freqs,
    channels,
    metric_file
):
    """
    Load metric CSVs and average across epochs and channels per subject.
    """
    records = []

    for freq in freqs:
        for group in groups:
            subject_values = []

            for channel in channels:
                file_path = (
                    base_path
                    / group
                    / freq
                    / f"channel_{channel}"
                    / metric_file
                )

                if not file_path.exists():
                    continue

                # shape: (subjects x epochs)
                df = pd.read_csv(file_path, header=None)

                # epoch-averaged per subject
                subj_avg = df.mean(axis=1)
                subject_values.append(subj_avg)

            if subject_values:
                # average across channels
                df_all = pd.concat(subject_values, axis=1)
                final_vals = df_all.mean(axis=1)

                for v in final_vals:
                    records.append({
                        "Group": group.upper(),
                        "Frequency": freq.upper(),
                        "Value": v,
                        "Metric": metric_file.replace(".csv", "")
                    })

    return pd.DataFrame(records)


def plot_boxplots(
    df,
    output_path: Path,
    title: str
):
    """
    Plot boxplots for all metrics.
    """
    sns.set(style="whitegrid")

    metrics = df["Metric"].unique()
    fig, axs = plt.subplots(2, 5, figsize=(22, 10))
    axs = axs.flatten()

    for i, metric in enumerate(metrics):
        df_m = df[df["Metric"] == metric]

        sns.boxplot(
            data=df_m,
            x="Frequency",
            y="Value",
            hue="Group",
            ax=axs[i],
            showmeans=True,
            meanprops={
                "marker": "o",
                "markerfacecolor": "red",
                "markeredgecolor": "black",
                "markersize": 5,
            },
        )

        axs[i].set_title(metric, fontsize=11)
        axs[i].set_xlabel("Frequency Band")
        axs[i].set_ylabel("Averaged Value")

    plt.suptitle(title, fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=300)
    plt.show()


if __name__ == "__main__":

    BASE_INPUT = Path("results/VG_hubs_nonhub")
    OUTPUT_DIR = Path("visualization_outputs")
    OUTPUT_DIR.mkdir(exist_ok=True)

    GROUPS = ["mdd", "normal"]
    FREQS = ["all", "beta", "alpha", "theta"]
    CHANNELS = [20, 31, 33, 67, 70, 89, 124]

    METRICS = [
        "hub_nonhub_ratio.csv",
        "hub_percent.csv",
        "nonhub_percent.csv",
        "R1.csv", "R2.csv", "R3.csv",
        "R4.csv", "R5.csv", "R6.csv", "R7.csv"
    ]

    all_data = []

    for metric in METRICS:
        df_metric = load_and_average_metric(
            BASE_INPUT, GROUPS, FREQS, CHANNELS, metric
        )
        all_data.append(df_metric)

    df_all = pd.concat(all_data, ignore_index=True)

    plot_boxplots(
        df_all,
        OUTPUT_DIR / "connectivity_boxplots.png",
        "Connectivity Comparison: MDD vs Control (Epoch & Channel Averaged)"
    )
