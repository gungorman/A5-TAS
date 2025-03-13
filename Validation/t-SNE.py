import pandas as pd
import numpy as np
from openTSNE import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import trustworthiness

# File path
file_path = 'Validation/EHAM_LIMC.csv'  # Replace with your actual file path

# Features for t-SNE
features = ['latitude', 'longitude', 'altitude']

# Process the CSV in chunks and collect the first 650 trajectories
chunk_size = 1_000_000  # Adjust based on your RAM
target_n_trajectories = 650
unique_flight_ids = set()
trajectory_summaries = []

for chunk in pd.read_csv(file_path, chunksize=chunk_size):
    # Get unique flight_ids in this chunk
    chunk_flight_ids = chunk['flight_id'].unique()
    
    # Add new flight_ids until we reach 650
    for flight_id in chunk_flight_ids:
        if flight_id not in unique_flight_ids and len(unique_flight_ids) < target_n_trajectories:
            unique_flight_ids.add(flight_id)
    
    # Stop if we've collected 650 flight_ids
    if len(unique_flight_ids) >= target_n_trajectories:
        # Filter chunk to only include the first 650 flight_ids
        filtered_chunk = chunk[chunk['flight_id'].isin(unique_flight_ids)]
        summary = filtered_chunk.groupby('flight_id')[features].mean()
        trajectory_summaries.append(summary)
        break  # Exit after this chunk
    
    # Summarize this chunk for flight_ids we’re tracking
    filtered_chunk = chunk[chunk['flight_id'].isin(unique_flight_ids)]
    if not filtered_chunk.empty:
        summary = filtered_chunk.groupby('flight_id')[features].mean()
        trajectory_summaries.append(summary)

# Combine summaries and ensure exactly 650 trajectories
trajectory_data = pd.concat(trajectory_summaries).groupby('flight_id').mean().values
if trajectory_data.shape[0] > target_n_trajectories:
    trajectory_data = trajectory_data[:target_n_trajectories]  # Trim to exactly 650

print(f"Processed {trajectory_data.shape[0]} trajectories with {trajectory_data.shape[1]} features")

# Normalize the data
scaler = StandardScaler()
trajectory_data = scaler.fit_transform(trajectory_data)

# Apply t-SNE
tsne = TSNE(
    n_components=2,
    perplexity=10,        # Suitable for 650 points; adjust if needed (e.g., 5-50)
    learning_rate="auto",
    n_jobs=-1,
    verbose=1
)
embedding = tsne.fit(trajectory_data)
embedding_array = np.array(embedding)

# Color by flight index (0 to 649)
colors = np.arange(650)

# Plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(embedding_array[:, 0], embedding_array[:, 1], c=colors, s=50, cmap='tab20')
plt.colorbar(scatter, label='Trajectory Index')
plt.title("t-SNE Embedding of First 650 Flight Trajectories")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.show()

# Compute trustworthiness
trust_score = trustworthiness(trajectory_data, embedding_array, n_neighbors=5)
print(f"Trustworthiness score: {trust_score:.4f}")