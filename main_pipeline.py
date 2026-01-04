import pandas as pd
import numpy as np
from pathlib import Path

from networks.visibility_graph import compute_visibility_graph
from networks.network_metrics import compute_network_metrics
from networks.hub_classification import (
    within_module_degree_zscore,
    classify_node_roles
)

# ---------------- CONFIG ----------------
DATA_ROOT = Path("data")   # data/mdd/, data/normal/
OUTPUT_FILE = Path("results/network_metrics_results.csv")
OUTPUT_FILE.parent.mkdir(exist_ok=True)

GROUPS = ["mdd", "normal"]

SIGNIFICANT_CHANNELS = [
    "channel_31", "channel_124", "channel_33",
    "channel_20", "channel_67", "channel_70", "channel_89"
]


# ---------------- MAIN PIPELINE ----------------
def run_network_pipeline():
    """
    Run epoch-level EEG network analysis.

    Each EEG epoch is independently processed to construct a visibility
    graph, compute network metrics, and classify hub roles.
    """
    results = []

    for group in GROUPS:
        group_dir = DATA_ROOT / group
        if not group_dir.exists():
            continue

        for subject_dir in sorted(group_dir.iterdir()):
            if not subject_dir.is_dir():
                continue

            subject_id = subject_dir.name

            for channel in SIGNIFICANT_CHANNELS:
                channel_dir = subject_dir / channel
                if not channel_dir.exists():
                    continue

                for epoch_file in sorted(channel_dir.glob("epoch_*.csv")):
                    # 1️ Load single epoch
                    signal = pd.read_csv(
                        epoch_file, header=None
                    ).iloc[:, 0].values

                    if len(signal) < 10:
                        continue

                    # 2️ Visibility Graph
                    adj_matrix = compute_visibility_graph(signal)

                    # 3️ Network metrics
                    metrics = compute_network_metrics(adj_matrix)

                    # 4️ Within-module z-score
                    z = within_module_degree_zscore(
                        adj_matrix,
                        metrics["communities"]
                    )

                    # 5️ Hub classification
                    roles = classify_node_roles(
                        metrics["participation"],
                        z
                    )

                    # 6️ Store epoch-level results
                    results.append({
                        "group": group.upper(),
                        "subject": subject_id,
                        "channel": channel,
                        "epoch": epoch_file.stem,

                        "avg_degree": np.mean(metrics["degree"]),
                        "avg_clustering": metrics["avg_clustering"],
                        "modularity": metrics["modularity"],
                        "avg_participation": np.mean(metrics["participation"]),
                        "avg_eigenvector": np.mean(metrics["eigenvector_centrality"]),

                        "R5_count": roles.count("R5"),
                        "R6_count": roles.count("R6"),
                        "R7_count": roles.count("R7")
                    })

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)


if __name__ == "__main__":
    run_network_pipeline()
