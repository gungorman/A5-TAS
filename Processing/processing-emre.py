import numpy as np

# Load the .npz file
data_file = np.load(r"Processing\air_subsampled_train_perc_10.npz")

# Access the array (assuming the key is 'data' based on previous steps)
data = data_file["data"]

# Check the shape to confirm
print("Data shape:", data.shape)
print(data)