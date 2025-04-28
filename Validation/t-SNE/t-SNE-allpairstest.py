import pandas as pd
import numpy as np
from openTSNE import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Define parameters
file_paths = {
    'EHAM_LIMC': 'Validation/EHAM_LIMC.csv',
    'ESSA_LFPG': 'Validation/ESSA_LFPG.csv',
    'LOWW_EGLL': 'Validation/LOWW_EGLL.csv'
}
features = ['latitude', 'longitude', 'altitude', 'timedelta']
target_n_trajectories_per_pair = 2000
perplexity = 10

# Step 1: Load and preprocess data
all_trajectories = []
for pair, file_path in file_paths.items():
    df = pd.read_csv(file_path, usecols=['flight_id'] + features)
    unique_flight_ids = df['flight_id'].unique()[:target_n_trajectories_per_pair]
    df = df[df['flight_id'].isin(unique_flight_ids)].copy()
    # Make flight_id unique across pairs
    df['flight_id'] = df['flight_id'].astype(str) + '_' + pair
    df['airport_pair'] = pair
    all_trajectories.append(df)
df_trajectories = pd.concat(all_trajectories, ignore_index=True)
df_trajectories = df_trajectories.drop_duplicates(subset=['flight_id', 'timedelta'])
print(f"Processed {df_trajectories.shape[0]} points from {df_trajectories['flight_id'].nunique()} trajectories across {len(file_paths)} airport pairs")

# Step 2: Summarize trajectories
df_trajectories = df_trajectories.sort_values(['flight_id', 'timedelta'])
summary_df = (df_trajectories.groupby('flight_id')[features]
              .agg(['mean', 'std', 'min', 'max'])
              .fillna(0))
feature_columns = summary_df.columns.map('_'.join)
summary_df.columns = feature_columns
flight_to_pair = df_trajectories.groupby('flight_id')['airport_pair'].first()
flight_to_pair = flight_to_pair.reindex(summary_df.index)
print(f"Using {summary_df.shape[0]} summarized points for t-SNE")
print("Airport pair distribution:\n", flight_to_pair.value_counts())

# Step 3: Normalize the data
scaler = StandardScaler()
trajectory_summary = scaler.fit_transform(summary_df)

# Step 4: Compute t-SNE embedding
tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate='auto', n_iter=1000, n_jobs=-1, verbose=1)
embedding_array = np.array(tsne.fit(trajectory_summary))

# Step 5: Create a single plot colored by airport pair
plt.figure(figsize=(10, 8))
colors = {'EHAM_LIMC': 'blue', 'ESSA_LFPG': 'green', 'LOWW_EGLL': 'red'}
for pair in file_paths.keys():
    mask = (flight_to_pair == pair)
    if mask.sum() > 0:
        plt.scatter(embedding_array[mask, 0], embedding_array[mask, 1],
                    c=colors[pair], s=10, alpha=0.7, label=pair)
plt.legend()
plt.title("t-SNE of Flight Trajectories by Airport Pair")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.show()