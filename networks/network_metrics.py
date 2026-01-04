import numpy as np
import networkx as nx
from community import community_louvain


def build_graph(adj_matrix: np.ndarray) -> nx.Graph:
    """
    Build an undirected NetworkX graph from adjacency matrix.
    """
    return nx.from_numpy_array(adj_matrix)


# ---------------- DEGREE ----------------
def degree(adj_matrix: np.ndarray) -> np.ndarray:
    """
    Degree k_i = sum_j A_ij
    """
    return np.sum(adj_matrix, axis=1)


# ---------------- CLUSTERING COEFFICIENT ----------------
def clustering_coefficient(G: nx.Graph) -> np.ndarray:
    """
    Node-wise clustering coefficient C_i
    """
    return np.array(list(nx.clustering(G).values()))


def average_clustering(G: nx.Graph) -> float:
    """
    Dynamic clustering coefficient C(t)
    """
    return nx.average_clustering(G)


# ---------------- COMMUNITY & MODULARITY ----------------
def compute_communities(G: nx.Graph) -> dict:
    """
    Louvain community detection.
    Returns node -> community mapping.
    """
    return community_louvain.best_partition(G)


def modularity(G: nx.Graph, communities: dict) -> float:
    """
    Modularity Q
    """
    return community_louvain.modularity(communities, G)


# ---------------- PARTICIPATION COEFFICIENT ----------------
def participation_coefficient(
    G: nx.Graph,
    communities: dict
) -> np.ndarray:
    """
    Participation coefficient P_i
    """
    degrees = dict(G.degree())
    modules = set(communities.values())
    P = np.zeros(len(G.nodes()))

    for i in G.nodes():
        ki = degrees[i]
        if ki == 0:
            P[i] = 0
            continue

        sum_frac = 0
        for m in modules:
            kis = sum(
                1 for j in G.neighbors(i)
                if communities[j] == m
            )
            sum_frac += (kis / ki) ** 2

        P[i] = 1 - sum_frac

    return P


# ---------------- EIGENVECTOR CENTRALITY ----------------
def eigenvector_centrality(G: nx.Graph) -> np.ndarray:
    """
    Eigenvector centrality v_i
    """
    ec = nx.eigenvector_centrality_numpy(G)
    return np.array(list(ec.values()))


# ---------------- MASTER FUNCTION ----------------
def compute_network_metrics(adj_matrix: np.ndarray) -> dict:
    """
    Compute all network metrics for one visibility graph.
    """
    G = build_graph(adj_matrix)
    communities = compute_communities(G)

    return {
        "degree": degree(adj_matrix),
        "clustering": clustering_coefficient(G),
        "avg_clustering": average_clustering(G),
        "modularity": modularity(G, communities),
        "participation": participation_coefficient(G, communities),
        "eigenvector_centrality": eigenvector_centrality(G),
        "communities": communities
    }
