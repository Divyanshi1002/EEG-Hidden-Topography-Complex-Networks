import numpy as np
from collections import defaultdict


def within_module_degree_zscore(
    adj_matrix: np.ndarray,
    communities: dict
) -> np.ndarray:
    """
    Compute within-module degree z-score z_i
    """
    N = adj_matrix.shape[0]
    z = np.zeros(N)

    # Group nodes by module
    module_nodes = defaultdict(list)
    for node, mod in communities.items():
        module_nodes[mod].append(node)

    for mod, nodes in module_nodes.items():
        k_s = []

        for i in nodes:
            ki_s = sum(
                adj_matrix[i, j]
                for j in nodes
            )
            k_s.append(ki_s)

        mean_k = np.mean(k_s)
        std_k = np.std(k_s)

        for idx, i in enumerate(nodes):
            if std_k > 0:
                z[i] = (k_s[idx] - mean_k) / std_k
            else:
                z[i] = 0

    return z


def classify_node_roles(
    participation: np.ndarray,
    z: np.ndarray
) -> list:
    """
    Classify nodes into R1–R7 roles (Guimerà & Amaral).
    """
    roles = []

    for Pi, zi in zip(participation, z):
        if zi < 2.5:  # Non-hubs
            if Pi < 0.05:
                roles.append("R1")  # Ultra-peripheral non-hub
            elif Pi < 0.62:
                roles.append("R2")  # Peripheral non-hub
            elif Pi < 0.8:
                roles.append("R3")  # Connector non-hub
            else:
                roles.append("R4")  # Kinless non-hub
        else:  # Hubs
            if Pi < 0.3:
                roles.append("R5")  # Provincial hub
            elif Pi < 0.75:
                roles.append("R6")  # Connector hub
            else:
                roles.append("R7")  # Kinless hub

    return roles
