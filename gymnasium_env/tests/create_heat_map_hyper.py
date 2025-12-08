import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Data: alpha, episodes, epsilon, percent_correct, avg_reward, successful_episodes
data = [
    [0.1, 2000, 0.1, 38.6363, 11.079, 1.5],
    [0.1, 2000, 0.3, 43.1818, 44.2245, 52.9],
    [0.1, 2000, 0.5, 38.6363, 37.7195, 41.7],
    [0.01, 2000, 0.1, 50.0, 23.4405, 5.45],
    [0.01, 2000, 0.3, 56.8181, 50.417, 57.5],
    [0.01, 2000, 0.5, 70.4545, 44.7275, 42.9],
    [0.001, 2000, 0.1, 25.0, 12.9775, 5.9],
    [0.001, 2000, 0.3, 84.0909, 59.646, 59.1],
    [0.001, 2000, 0.5, 25.0, 21.771, 15.15],
    [0.1, 10000, 0.1, 43.1818, 62.3422, 87.45],
    [0.1, 10000, 0.3, 38.6363, 57.9159, 82.26],
    [0.1, 10000, 0.5, 38.6363, 55.8622, 78.16],
    [0.01, 10000, 0.1, 59.0909, 25.995, 4.4],
    [0.01, 10000, 0.3, 43.1818, 19.3447, 4.48],
    [0.01, 10000, 0.5, 56.8181, 39.7941, 30.15],
    [0.001, 10000, 0.1, 52.2727, 22.2012, 0.42],
    [0.001, 10000, 0.3, 56.8181, 22.9961, 0.65],
    [0.001, 10000, 0.5, 56.8181, 59.635, 73.66],
    [0.1, 20000, 0.1, 43.1818, 58.5197, 77.165],
    [0.1, 20000, 0.3, 38.6363, 60.14615, 87.395],
    [0.1, 20000, 0.5, 38.6363, 58.8996, 84.49],
    [0.01, 20000, 0.1, 38.6363, 59.50975, 86.875],
    [0.01, 20000, 0.3, 43.1818, 61.78755, 83.41],
    [0.01, 20000, 0.5, 43.1818, 60.7729, 81.705],
    [0.001, 20000, 0.1, 31.8181, 14.0126, 2.53],
    [0.001, 20000, 0.3, 61.3636, 24.8608, 0.51],
    [0.001, 20000, 0.5, 34.0909, 18.09915, 7.9]
]

data = np.array(data)
alphas = np.unique(data[:,0])
epsilons = np.unique(data[:,2])
episodes_list = np.unique(data[:,1])

# ---------------------------------------------------
# Heatmaps for percent_correct and successful_episodes for each episode count
# ---------------------------------------------------

fig, axes = plt.subplots(len(episodes_list), 2, figsize=(14, 14))
fig.suptitle("Q-Learning Performance Analysis", fontsize=18, fontweight='bold')

for i, ep_count in enumerate(episodes_list):
    subset = data[data[:,1]==ep_count]
    
    percent_matrix = np.zeros((len(alphas), len(epsilons)))
    success_matrix = np.zeros((len(alphas), len(epsilons)))
    
    for ai, alpha in enumerate(alphas):
        for ei, epsilon in enumerate(epsilons):
            row = subset[(subset[:,0]==alpha) & (subset[:,2]==epsilon)]
            if len(row) > 0:
                percent_matrix[ai, ei] = row[0,3]
                success_matrix[ai, ei] = row[0,5]
    
    # Create DataFrames for seaborn
    row_labels = [f"α={a}" for a in alphas]
    col_labels = [f"ε={e}" for e in epsilons]
    
    df_percent = pd.DataFrame(percent_matrix, index=row_labels, columns=col_labels)
    df_success = pd.DataFrame(success_matrix, index=row_labels, columns=col_labels)
    
    # Heatmap 1 — Percent Correct with values inside squares
    sns.heatmap(
        df_percent,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        cbar=True,
        ax=axes[i,0],
        vmin=0,
        vmax=100
    )
    axes[i,0].set_xlabel("Epsilon")
    axes[i,0].set_ylabel("Alpha")
    axes[i,0].set_title(f"% Tiles Visited - Episodes: {int(ep_count)}", fontsize=12)
    
    # Heatmap 2 — Successful Episodes with values inside squares
    sns.heatmap(
        df_success,
        annot=True,
        fmt=".2f",
        cmap="plasma",
        cbar=True,
        ax=axes[i,1],
        vmin=0,
        vmax=100
    )
    axes[i,1].set_xlabel("Epsilon")
    axes[i,1].set_ylabel("")
    axes[i,1].set_title(f"Success Rate (%) - Episodes: {int(ep_count)}", fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.97])

# Save the figure
plt.savefig("hyper_param_test.png", dpi=300)
plt.show()
