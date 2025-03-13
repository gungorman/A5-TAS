import pandas as pd
import numpy as np
from openTSNE import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import trustworthiness

# Step 1: Load your CSV data
file_path = 'validation/Testing_Data.csv'  # Replace with your actual file path
df = pd.read_csv(file_path)

# Step 2: Prepare the data
# Select spatial features for t-SNE
features = ['latitude', 'longitude', 'altitude',]
trajectory_data = df[features].values  # Shape: (n_points, 3)

# Normalize the data (important for t-SNE)
scaler = StandardScaler()
trajectory_data = scaler.fit_transform(trajectory_data)

# Print dataset info (similar to the simple usage example)
print(f"Data set contains {trajectory_data.shape[0]} samples with {trajectory_data.shape[1]} features")

# Step 3: Apply t-SNE (inspired by openTSNE simple usage)
tsne = TSNE(
    n_components=2,       # Reduce to 2D for visualization
    perplexity=30,        # Default value, adjust based on your data size
    learning_rate="auto", # Automatic learning rate
    n_jobs=-1,            # Use all CPU cores
    verbose=1             # Show progress
)
embedding = tsne.fit(trajectory_data)

# Convert embedding to NumPy array
embedding_array = np.array(embedding)

# Step 4: Visualize the embedding
# Color points by cluster (or flight_id if preferred)
colors = df['cluster'].astype(int)  # Use cluster column for coloring
plt.figure(figsize=(10, 8))
scatter = plt.scatter(embedding_array[:, 0], embedding_array[:, 1], c=colors, s=5, cmap='tab20')
plt.colorbar(scatter, label='Cluster')
plt.title("t-SNE Embedding of Flight Trajectory Points")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.show()

# Step 5: Validation - Compute trustworthiness score
trust_score = trustworthiness(trajectory_data, embedding_array, n_neighbors=5)
print(f"Trustworthiness score: {trust_score:.4f}")  # Closer to 1 is better