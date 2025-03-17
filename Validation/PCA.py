import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Configurable parameters
file_path = 'Validation/EHAM_LIMC.csv'
features = ['latitude', 'longitude', 'altitude', 'timedelta']
target_n_trajectories = 6591
n_clusters = 2

# Step 1: Load and filter trajectories
df = pd.read_csv(file_path, usecols=['flight_id'] + features)
unique_flight_ids = df['flight_id'].unique()[:target_n_trajectories]
df_trajectories = df[df['flight_id'].isin(unique_flight_ids)]
print(f"Processed {df_trajectories.shape[0]} points from {df_trajectories['flight_id'].nunique()} trajectories")

# Step 2: Summarize trajectories
df_trajectories = df_trajectories.sort_values(['flight_id', 'timedelta'])
summary_df = (df_trajectories.groupby('flight_id')[features]
              .agg(['mean', 'std', 'min', 'max'])
              .fillna(0))
feature_columns = summary_df.columns.map('_'.join)
summary_df.columns = feature_columns
trajectory_summary = summary_df.values
print(f"Using {trajectory_summary.shape[0]} summarized points for PCA")

# Step 3: Normalize and apply PCA
scaler = StandardScaler()
trajectory_summary = scaler.fit_transform(trajectory_summary)
pca = PCA(n_components=2)
embedding_array = pca.fit_transform(trajectory_summary)
explained_variance_ratio = pca.explained_variance_ratio_
explained_variance_ratio = explained_variance_ratio ** 2
explained_variance_ratio /= explained_variance_ratio.sum()
print(f"Explained variance ratio: {explained_variance_ratio[0]:.4f} (PC1), {explained_variance_ratio[1]:.4f} (PC2)")

# Step 4: Clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(embedding_array)

# Step 5: Visualize
plt.figure(figsize=(6, 5))
scatter = plt.scatter(embedding_array[:, 0], embedding_array[:, 1], c=clusters, s=10, cmap='tab10', alpha=0.7)
plt.colorbar(scatter, label=f'KMeans Clusters (n={n_clusters})')
plt.title(f"PCA (PC1: {explained_variance_ratio[0]:.2%}, PC2: {explained_variance_ratio[1]:.2%})")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.tight_layout()
plt.show()