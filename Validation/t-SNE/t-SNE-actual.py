import pandas as pd
import numpy as np
from openTSNE import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Define parameters
file_path = 'Validation/EHAM_LIMC.csv'  # Using just one file
features = ['latitude', 'longitude', 'altitude', 'timedelta']
target_n_trajectories = 2000  # Reduced number of trajectories to process
perplexity = 10

# Step 1: Load and preprocess data
df = pd.read_csv(file_path, usecols=['flight_id'] + features)
df = df[df['flight_id'].isin(df['flight_id'].unique()[:target_n_trajectories])].copy()
print(f"Processed {df.shape[0]} points from {df['flight_id'].nunique()} trajectories")

# Step 2: Summarize trajectories
df = df.sort_values(['flight_id', 'timedelta'])
summary_df = df.groupby('flight_id')[features].agg(['mean', 'std', 'min', 'max']).fillna(0)
summary_df.columns = ['_'.join(col) for col in summary_df.columns]
print(f"Using {summary_df.shape[0]} summarized points for t-SNE")

# Step 3: Normalize the data
scaler = StandardScaler()
trajectory_summary = scaler.fit_transform(summary_df)

# Step 4: Compute t-SNE embedding
tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate='auto', n_iter=1000, n_jobs=-1, verbose=1)
embedding = tsne.fit(trajectory_summary)

# Step 5: Plot the results
plt.figure(figsize=(10, 8))
plt.scatter(embedding[:, 0], embedding[:, 1], s=10, alpha=0.7)
plt.title("t-SNE of Flight Trajectories (EHAM_LIMC)")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.show()