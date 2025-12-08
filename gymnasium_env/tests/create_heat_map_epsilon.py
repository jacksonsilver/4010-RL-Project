import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# ---------------------------------------------------
# Data provided by user
# ---------------------------------------------------

rows = [
    "init=0.1,end=0.001","init=0.1,end=0.05","init=0.1,end=0.1",
    "init=0.3,end=0.001","init=0.3,end=0.05","init=0.3,end=0.1",
    "init=0.5,end=0.001","init=0.5,end=0.05","init=0.5,end=0.1"
]

cols = ["decay=1M", "decay=4M", "decay=8M"]

percent_tiles = np.array([
    [49.5455,50.9091,46.8182],
    [45.0000,44.0909,40.4545],
    [40.4545,40.4545,39.5455],
    [43.1818,41.3636,39.5455],
    [42.2727,40.4545,40.4545],
    [41.3636,40.4545,38.6364],
    [41.3636,38.6364,38.6364],
    [38.6364,39.5455,39.5455],
    [40.4545,38.6364,38.6364]
])

success_rate = np.array([
    [92.2910,73.2130,91.1550],
    [83.8230,87.1000,80.8800],
    [74.7410,80.9790,80.0300],
    [93.6040,92.0560,88.4420],
    [83.1480,86.6910,85.2490],
    [78.7260,77.3550,78.4210],
    [93.7180,88.9520,84.9790],
    [87.4930,84.0310,82.0410],
    [79.1520,77.4960,77.1640]
])

# ---------------------------------------------------
# Construct DataFrames
# ---------------------------------------------------

df_visited = pd.DataFrame(percent_tiles, index=rows, columns=cols)
df_success = pd.DataFrame(success_rate, index=rows, columns=cols)

# ---------------------------------------------------
# Combined Heatmaps (same PNG)
# ---------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(22, 10))

# Heatmap 1 — Percent Tiles Visited
sns.heatmap(
    df_visited,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    cbar=True,
    ax=axes[0]
)
axes[0].set_title("% Tiles Visited (Epsilon Decay Combinations)", fontsize=16)
axes[0].set_xlabel("Decay Rate")
axes[0].set_ylabel("Epsilon Settings")

# Heatmap 2 — Success Rate
sns.heatmap(
    df_success,
    annot=True,
    fmt=".2f",
    cmap="Greens",
    cbar=True,
    ax=axes[1]
)
axes[1].set_title("Avg Success Rate (Epsilon Decay Combinations)", fontsize=16)
axes[1].set_xlabel("Decay Rate")
axes[1].set_ylabel("")

plt.tight_layout()

# Save combined image
plt.savefig("epsilon_decay_combined.png", dpi=300)
plt.close()