import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


def permutation_test(
    data1,
    data2,
    num_permutations=5000,
    measure=np.mean,
    alternative="two-sided"
):
    """
    Non-parametric permutation test (mean difference).

    Returns
    -------
    p_value : float
    observed_diff : float
    """
    observed_diff = measure(data1) - measure(data2)
    combined = np.concatenate([data1, data2])
    n1 = len(data1)

    extreme_count = 0

    for _ in range(num_permutations):
        np.random.shuffle(combined)
        perm_diff = measure(combined[:n1]) - measure(combined[n1:])

        if alternative == "two-sided":
            extreme_count += abs(perm_diff) >= abs(observed_diff)
        elif alternative == "greater":
            extreme_count += perm_diff >= observed_diff
        elif alternative == "less":
            extreme_count += perm_diff <= observed_diff

    p_value = extreme_count / num_permutations
    return p_value, observed_diff

# Align channels
common_cols = data1.columns.intersection(data2.columns)

data1 = data1[common_cols]
data2 = data2[common_cols]

results = []
p_values = []

for col in data1.columns:
    p_val, mean_diff = permutation_test(
        data1[col].values,
        data2[col].values,
        num_permutations=5000
    )
    p_values.append(p_val)
    results.append([col, mean_diff, p_val])

# FDR correction (Benjamini–Hochberg)
reject, pvals_fdr, _, _ = multipletests(
    p_values,
    alpha=0.05,
    method="fdr_bh"
)

results_df = pd.DataFrame(
    results,
    columns=["Channel", "mean_difference", "p_value"]
)
results_df["p_value_fdr"] = pvals_fdr
results_df["significant"] = reject

results_df.to_csv(
    "permutation_test_fdr_5000.csv",
    index=False
)

print("Permutation test with FDR correction completed.")