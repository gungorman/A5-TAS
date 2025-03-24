import pandas as pd
import numpy as np
from openTSNE import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import trustworthiness
from sklearn.cluster import KMeans

# Configurable parameters
file_paths = {
    'EHAM_LIMC': 'Validation/EHAM_LIMC.csv',
    'ESSA_LFPG': 'Validation/ESSA_LFPG.csv',
    'LOWW_EGLL': 'Validation/LOWW_EGLL.csv'
}
features = ['latitude', 'longitude', 'altitude', 'timedelta']
target_n_trajectories_per_pair = 100
perplexity = 30  # Match plot's perplexity
n_clusters = 3

# Step 1: Collect unique trajectories from all files
all_trajectories = []
for pair, file_path in file_paths.items():
    df = pd.read_csv(file_path, usecols=['flight_id'] + features)
    unique_flight_ids = df['flight_id'].unique()[:target_n_trajectories_per_pair]
    df = df[df['flight_id'].isin(unique_flight_ids)].assign(airport_pair=pair)
    all_trajectories.append(df)
df_trajectories = pd.concat(all_trajectories).drop_duplicates(subset='flight_id')
print(f"Processed {df_trajectories.shape[0]} points from {df_trajectories['flight_id'].nunique()} trajectories across {len(file_paths)} airport pairs")

# Step 2: Summarize trajectories
df_trajectories = df_trajectories.sort_values(['flight_id', 'timedelta'])
summary_df = (df_trajectories.groupby('flight_id')[features]
              .agg(['mean', 'std', 'min', 'max'])
              .fillna(0))
feature_columns = summary_df.columns.map('_'.join)
summary_df.columns = feature_columns
trajectory_summary = summary_df.values
# Map flight_ids to airport pairs for coloring
flight_to_pair = df_trajectories[['flight_id', 'airport_pair']].drop_duplicates().set_index('flight_id')['airport_pair']
print(f"Using {trajectory_summary.shape[0]} summarized points for t-SNE")

# Step 3: Normalize and apply t-SNE
scaler = StandardScaler()
trajectory_summary = scaler.fit_transform(trajectory_summary)
tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate='auto', n_iter=1000, n_jobs=-1, verbose=1)
embedding_array = np.array(tsne.fit(trajectory_summary))

# Step 4: Clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(embedding_array)

# Step 5: Visualize with explicit colors for airport pairs
plt.figure(figsize=(12, 5))

# Left plot: Color by airport pair with distinct colors
plt.subplot(1, 2, 1)
colors = {'EHAM_LIMC': 'blue', 'ESSA_LFPG': 'green', 'LOWW_EGLL': 'red'}
for pair in file_paths.keys():
    mask = flight_to_pair.reindex(summary_df.index) == pair
    if mask.sum() > 0:  # Ensure there are points to plot
        plt.scatter(embedding_array[mask, 0], embedding_array[mask, 1], 
                    c=colors[pair], s=10, alpha=0.7, label=pair)
plt.legend()
plt.title(f"t-SNE (Perplexity = {perplexity})")
plt.xlabel("t-SNE Dim 1")
plt.ylabel("t-SNE Dim 2")

# Right plot: KMeans clusters
plt.subplot(1, 2, 2)
scatter = plt.scatter(embedding_array[:, 0], embedding_array[:, 1], 
                      c=clusters, s=10, cmap='tab10', alpha=0.7)
plt.colorbar(scatter, label=f'KMeans Clusters (n={n_clusters})')
plt.title(f"t-SNE (Perplexity = {perplexity})")
plt.xlabel("t-SNE Dim 1")
plt.ylabel("t-SNE Dim 2")

plt.tight_layout()
plt.show()

# Step 6: Validate
trust_score = trustworthiness(trajectory_summary, embedding_array, n_neighbors=5)
print(f"Trustworthiness score: {trust_score:.4f}")