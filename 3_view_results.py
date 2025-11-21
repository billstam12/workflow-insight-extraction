import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("data/workflows/workflows_clustered.csv")

# pick metrics that convey performance & fairness
metric_cols = [
    "AUC-ROC",
    "Accuracy",
    "Equal Opportunity",
    "Equalized Odds",
    "Statistical Parity",
    "Disparate Impact",
    "Well Calibration",
]

X = df[metric_cols].values
X_std = StandardScaler().fit_transform(X)

pca = PCA(n_components=2, random_state=0)
X_2d = pca.fit_transform(X_std)

df["PC1"] = X_2d[:, 0]
df["PC2"] = X_2d[:, 1]


plt.figure(figsize=(5, 4))

# Define different markers for each cluster
markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h', '+']
clusters = sorted(df['cluster'].unique())

# Plot each cluster with a different marker
for cluster in clusters:
    cluster_data = df[df['cluster'] == cluster]
    plt.scatter(
        cluster_data['PC1'],
        cluster_data['PC2'],
        marker=markers[cluster % len(markers)],
        s=100,
        alpha=0.8,
        label=f'Cluster {cluster}',
        edgecolors='black',
        linewidth=0.5
    )

ax = plt.gca()
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.)
plt.tight_layout()
plt.show()