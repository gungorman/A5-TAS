import pandas as pd
import numpy as np
from openTSNE import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import trustworthiness
from sklearn.cluster import KMeans

# File path
file_path = 'Validation/EHAM_LIMC.csv'  # Replace with your actual file path
features = ['latitude', 'longitude', 'altitude', 'timedelta']
target_n_trajectories = 1500  # Target number of trajectories
perplexity = 10  # Single perplexity value, adjustable by you

# Step 1: Collect all points for the first 650 trajectories
chunk_size = 1_000_000
unique_flight_ids = set()
trajectory_points = []

for chunk in pd.read_csv(file_path, chunksize=chunk_size):
    chunk_flight_ids = chunk['flight_id'].unique()
    for flight_id in chunk_flight_ids:
        if len(unique_flight_ids) < target_n_trajectories:
            unique_flight_ids.add(flight_id)
    filtered_chunk = chunk[chunk['flight_id'].isin(unique_flight_ids)]
    if not filtered_chunk.empty:
        trajectory_points.append(filtered_chunk[['flight_id'] + features])
    if len(unique_flight_ids) >= target_n_trajectories:
        break

# Combine into a single DataFrame
df_trajectories = pd.concat(trajectory_points)
print(f"Processed {df_trajectories.shape[0]} points from {len(unique_flight_ids)} trajectories")

# Step 2: Summarize each trajectory into a single point
df_trajectories = df_trajectories.sort_values(['flight_id', 'timedelta'])

# Compute statistical summaries for each trajectory
summaries = []
flight_ids_ordered = []  # Collect flight_ids in order

for flight_id, group in df_trajectories.groupby('flight_id'):
    flight_ids_ordered.append(flight_id)  # Add flight_id to ordered list
    summary = {}
    for feature in features:
        summary[f"{feature}_mean"] = group[feature].mean()
        summary[f"{feature}_std"] = group[feature].std()
        summary[f"{feature}_min"] = group[feature].min()
        summary[f"{feature}_max"] = group[feature].max()
    summaries.append(summary)

# Convert summaries to a DataFrame
summary_df = pd.DataFrame(summaries)
summary_df['flight_id'] = flight_ids_ordered  # Assign ordered flight_ids

# Features for t-SNE
feature_columns = [col for col in summary_df.columns if col != 'flight_id']
trajectory_summary = summary_df[feature_columns].fillna(0).values
flight_ids_summary = summary_df['flight_id'].astype('category').cat.codes
print(f"Using {trajectory_summary.shape[0]} summarized points for t-SNE")

# Step 3: Normalize
scaler = StandardScaler()
trajectory_summary = scaler.fit_transform(trajectory_summary)

# Step 4: Apply t-SNE
tsne = TSNE(
    n_components=2,
    perplexity=perplexity,
    learning_rate=200,
    n_iter=1000,
    n_jobs=-1,
    verbose=1
)
embedding = tsne.fit(trajectory_summary)
embedding_array = np.array(embedding)

# Step 5: Visualize
plt.figure(figsize=(12, 5))

# Plot 1: Colored by Flight ID
plt.subplot(1, 2, 1)
scatter = plt.scatter(embedding_array[:, 0], embedding_array[:, 1], 
                      c=flight_ids_summary, s=10, cmap='tab20', alpha=0.7)
plt.colorbar(scatter, label='Flight ID (Coded)')
plt.title(f"t-SNE of {trajectory_summary.shape[0]} Trajectories (Perplexity = {perplexity})")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")

# Plot 2: Colored by KMeans Clusters
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_clusters = kmeans.fit_predict(embedding_array)

plt.subplot(1, 2, 2)
scatter = plt.scatter(embedding_array[:, 0], embedding_array[:, 1], 
                      c=kmeans_clusters, s=10, cmap='tab10', alpha=0.7)
plt.colorbar(scatter, label='KMeans Cluster')
plt.title(f"KMeans Clusters (Perplexity = {perplexity})")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")

plt.tight_layout()
plt.show()

# Step 6: Validation
trust_score = trustworthiness(trajectory_summary, embedding_array, n_neighbors=5)
print(f"Trustworthiness score: {trust_score:.4f}")