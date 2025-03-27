import pandas as pd
import numpy as np

def number_of_flights(files):

# Assuming 'files' is the path to your CSV file
    data = pd.read_csv(files)

# Get the unique values in the 'flight_id' column
    unique_flight_ids = data['flight_id'].unique()

# Get the number of unique flight IDs
    num_unique_flight_ids = len(unique_flight_ids)

    return unique_flight_ids

def group_by_flight(files):
# Assuming 'files' is the path to your CSV file
    data = pd.read_csv(files)

# Group the data by 'flight_id'
    grouped_data = data.groupby('flight_id')

# Iterate over each group
    for flight_id, group in grouped_data:
        print(f"Flight ID: {flight_id}")
        print(group)
    # You can perform further operations on each group here

def numpy_array(files):
    # Assuming 'files' is the path to your CSV file
    data = pd.read_csv(files)

    # Group the data by 'flight_id'
    grouped_data = data.groupby('flight_id')

    # Sort the grouped data by the numeric part of the flight_id
    grouped_data = sorted(grouped_data, key=lambda x: int(x[0].split('_')[-1]))

    # Determine the number of unique flight IDs
    num_flight_ids = len(grouped_data)

    # Determine the number of timesteps (rows) for each flight
    # Assuming the number of timesteps is constant and the same for all flights
    timesteps = len(grouped_data[0][1])  # Number of rows in the first group

    # Number of features (latitude, longitude, altitude, timedelta, flight_id)
    num_features = 5

    # Initialize an empty NumPy array with the desired shape
    output_array = np.zeros((num_flight_ids, timesteps, num_features))

    # Fill the array with data
    for i, (flight_id, group) in enumerate(grouped_data):
        # Extract the numeric part of the flight_id (e.g., TRAJ_0 --> 0)
        flight_id_numeric = int(flight_id.split('_')[-1])  # Split on '_' and take the last part

        # Extract the relevant columns
        group_data = group[['latitude', 'longitude', 'altitude', 'timedelta']].to_numpy()

        # Add flight_id as an additional feature (repeated for each timestep)
        flight_id_column = np.full((len(group), 1), flight_id_numeric)
        group_data_with_id = np.hstack((group_data, flight_id_column))

        # Fill the output array
        output_array[i, :len(group), :] = group_data_with_id

        #print(f"Flight ID: {flight_id}, Numeric Flight ID: {flight_id_numeric}")
        #print(f"Index: {i}")

    # Output the resulting NumPy array
    return output_array


def numpy_array_sampled(files, n):
    """
    Convert flight trajectory data from CSV to a NumPy array, sampling every nth timestep.
    
    Args:
        files (str): Path to the CSV file
        n (int): Sampling interval for timesteps (take every nth point)
    
    Returns:
        np.ndarray: Array of shape (num_flights, sampled_timesteps, num_features)
    """
    # Read the CSV file
    data = pd.read_csv(files)

    # Group the data by 'flight_id'
    grouped_data = data.groupby('flight_id')

    # Sort the grouped data by the numeric part of the flight_id
    grouped_data = sorted(grouped_data, key=lambda x: int(x[0].split('_')[-1]))

    # Determine the number of unique flight IDs
    num_flight_ids = len(grouped_data)

    # Determine the original number of timesteps (rows) for each flight
    original_timesteps = len(grouped_data[0][1])
    
    # Calculate the number of timesteps after sampling
    sampled_timesteps = (original_timesteps + n - 1) // n  # Ceiling division

    # Number of features (latitude, longitude, altitude, timedelta, flight_id)
    num_features = 5

    # Initialize an empty NumPy array with the desired shape
    output_array = np.zeros((num_flight_ids, sampled_timesteps, num_features))

    # Fill the array with sampled data
    for i, (flight_id, group) in enumerate(grouped_data):
        # Extract the numeric part of the flight_id
        flight_id_numeric = int(flight_id.split('_')[-1])

        # Extract the relevant columns and sample every nth row
        group_data = group[['latitude', 'longitude', 'altitude', 'timedelta']].iloc[::n].to_numpy()

        # Add flight_id as an additional feature (repeated for each sampled timestep)
        flight_id_column = np.full((len(group_data), 1), flight_id_numeric)
        group_data_with_id = np.hstack((group_data, flight_id_column))

        # Fill the output array
        output_array[i, :len(group_data), :] = group_data_with_id

    return output_array

def array_split(output_array):
    """
    Split the output_array into train, test, and validation arrays, excluding the flight_id column.

    Parameters:
        output_array (np.ndarray): The input array to split, with shape (num_flight_ids, max_timesteps, num_features).

    Returns:
        train_array (np.ndarray): Training data (80% of flights) with shape (num_train_flights, max_timesteps, num_features - 1).
        test_array (np.ndarray): Testing data (10% of flights) with shape (num_test_flights, max_timesteps, num_features - 1).
        val_array (np.ndarray): Validation data (10% of flights) with shape (num_val_flights, max_timesteps, num_features - 1).
    """
    # Step 1: Randomly shuffle the indices of flight_ids
    indices = np.arange(output_array.shape[0])  # Create an array of indices [0, 1, 2, ..., num_flight_ids - 1]
    np.random.shuffle(indices)  # Shuffle the indices

    # Step 2: Calculate the sizes for train, test, and validation sets
    num_flight_ids = output_array.shape[0]
    train_size = int(0.8 * num_flight_ids)
    test_size = int(0.1 * num_flight_ids)
    val_size = num_flight_ids - train_size - test_size  # Remaining for validation

    # Step 3: Split the shuffled indices into train, test, and validation sets
    train_indices = indices[:train_size]
    test_indices = indices[train_size:train_size + test_size]
    val_indices = indices[train_size + test_size:]

    # Step 4: Use the indices to slice the output_array, excluding the flight_id column (last column)
    train_array = output_array[train_indices, :, :-1]  # Exclude the last column (flight_id)
    test_array = output_array[test_indices, :, :-1]    # Exclude the last column (flight_id)
    val_array = output_array[val_indices, :, :-1]      # Exclude the last column (flight_id)

    # Output the shapes of the resulting arrays
    print(f"Train array shape: {train_array.shape}")
    print(f"Test array shape: {test_array.shape}")
    print(f"Validation array shape: {val_array.shape}")

    return train_array, test_array, val_array

def array_split_seed(output_array, random_seed=1):
    """
    Split the output_array into train, test, and validation arrays, excluding the flight_id column.

    Parameters:
        output_array (np.ndarray): The input array to split, with shape (num_flight_ids, max_timesteps, num_features).
        random_seed (int, optional): Seed for the random number generator to ensure reproducibility. Defaults to None.

    Returns:
        train_array (np.ndarray): Training data (80% of flights) with shape (num_train_flights, max_timesteps, num_features - 1).
        test_array (np.ndarray): Testing data (10% of flights) with shape (num_test_flights, max_timesteps, num_features - 1).
        val_array (np.ndarray): Validation data (10% of flights) with shape (num_val_flights, max_timesteps, num_features - 1).
    """
    # Set the random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)

    # Step 1: Randomly shuffle the indices of flight_ids
    indices = np.arange(output_array.shape[0])  # Create an array of indices [0, 1, 2, ..., num_flight_ids - 1]
    np.random.shuffle(indices)  # Shuffle the indices

    # Step 2: Calculate the sizes for train, test, and validation sets
    num_flight_ids = output_array.shape[0]
    train_size = int(0.8 * num_flight_ids)
    test_size = int(0.1 * num_flight_ids)
    val_size = num_flight_ids - train_size - test_size  # Remaining for validation

    # Step 3: Split the shuffled indices into train, test, and validation sets
    train_indices = indices[:train_size]
    test_indices = indices[train_size:train_size + test_size]
    val_indices = indices[train_size + test_size:]

    # Step 4: Use the indices to slice the output_array, excluding the flight_id column (last column)
    train_array = output_array[train_indices, :, :-1]  # Exclude the last column (flight_id)
    test_array = output_array[test_indices, :, :-1]    # Exclude the last column (flight_id)
    val_array = output_array[val_indices, :, :-1]      # Exclude the last column (flight_id)

    # Output the shapes of the resulting arrays
    print(f"Train array shape: {train_array.shape}")
    print(f"Test array shape: {test_array.shape}")
    print(f"Validation array shape: {val_array.shape}")

    return train_array, test_array, val_array

def save_arrays_to_npz(train_array, test_array, val_array, train_file, test_file, val_file):
    """
    Save train, test, and validation arrays into separate .npz files.

    Parameters:
        train_array (np.ndarray): Training data array.
        test_array (np.ndarray): Testing data array.
        val_array (np.ndarray): Validation data array.
        train_file (str): Full path for the training data file (e.g., 'data/train_data.npz').
        test_file (str): Full path for the testing data file (e.g., 'data/test_data.npz').
        val_file (str): Full path for the validation data file (e.g., 'data/val_data.npz').
    """
    # Save each array to a separate .npz file
    np.savez(train_file, data=train_array)
    np.savez(test_file, test=test_array)
    np.savez(val_file, val=val_array)

    print(f"Training data saved to {train_file}")
    print(f"Testing data saved to {test_file}")
    print(f"Validation data saved to {val_file}")


import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_trajectories_comparison(array1, array2, 
                                 titles=('Trajectories 1', 'Trajectories 2'),
                                 figsize=(16, 6), linewidth=1.5, alpha=0.7):
    """
    Plot two sets of flight trajectories side-by-side with altitude-based color gradients.
    Color bar is placed on the right side of the figure, outside the subplots.
    
    Parameters:
        array1 (np.ndarray): First trajectory array (num_traj, timesteps, features)
        array2 (np.ndarray): Second trajectory array (num_traj, timesteps, features)
        titles (tuple): Titles for each subplot
        figsize (tuple): Overall figure size (width, height)
        linewidth (float): Width of trajectory lines
        alpha (float): Transparency of lines
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Create a colormap for altitude (using viridis)
    cmap = plt.get_cmap('viridis')
    
    # Determine global min/max for consistent coloring
    global_alt_min = min(array1[:, :, 2].min(), array2[:, :, 2].min())
    global_alt_max = max(array1[:, :, 2].max(), array2[:, :, 2].max())
    norm = plt.Normalize(global_alt_min, global_alt_max)
    
    def plot_single_array(ax, array, title):
        """Helper function to plot one array"""
        for traj in array:
            lats = traj[:, 0]  # Latitude
            lons = traj[:, 1]  # Longitude
            alts = traj[:, 2]  # Altitude
            
            points = np.array([lons, lats]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            lc = LineCollection(segments, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)
            lc.set_array(alts)
            ax.add_collection(lc)
        
        ax.set_xlim(min(array1[:, :, 1].min(), array2[:, :, 1].min()),
                   max(array1[:, :, 1].max(), array2[:, :, 1].max()))
        ax.set_ylim(min(array1[:, :, 0].min(), array2[:, :, 0].min()),
                   max(array1[:, :, 0].max(), array2[:, :, 0].max()))
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(title)
        ax.grid(True)
    
    # Plot both arrays
    plot_single_array(ax1, array1, titles[0])
    plot_single_array(ax2, array2, titles[1])
    
    # Create an axis for the colorbar on the right side
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.5)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, label='Altitude')
    
    plt.tight_layout()
    plt.show()


def load_single_array(npz_file_path, array_name):
    """
    Load a single NumPy array from an .npz file.
    
    Parameters:
        npz_file_path (str): Path to the .npz file
        array_name (str): Name of the array to load
        
    Returns:
        np.ndarray: The requested NumPy array
        
    Raises:
        FileNotFoundError: If the .npz file doesn't exist
        KeyError: If the specified array doesn't exist in the file
    """
    try:
        with np.load(npz_file_path) as data:
            if array_name not in data:
                available = list(data.keys())
                raise KeyError(
                    f"Array '{array_name}' not found. "
                    f"Available arrays: {available}"
                )
            return data[array_name]
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {npz_file_path}")


sample_rate = 40

### ESSA_LFPG ###
output_ESSA_LFPG = numpy_array_sampled(r"C:\Users\gungo\Downloads\ESSA_LFPG.csv", sample_rate)
ESSA_LFPG_train_array, ESSA_LFPG_test_array , ESSA_LFPG_val_array = array_split_seed(output_ESSA_LFPG)

output_dir = r"C:\Users\gungo\OneDrive\Desktop\A05 Data"  # Use raw string to avoid escaping backslashes

# Define the full paths for the output files
ESSA_LFPG_train_file = f"{output_dir}/ESSA_LFPG_train_data_n={sample_rate}.npz"
ESSA_LFPG_test_file = f"{output_dir}/ESSA_LFPG_test_data_n={sample_rate}.npz"
ESSA_LFPG_val_file = f"{output_dir}/ESSA_LFPG_val_data_n={sample_rate}.npz"

#save_arrays_to_npz(ESSA_LFPG_train_array, ESSA_LFPG_test_array, ESSA_LFPG_val_array, ESSA_LFPG_train_file, ESSA_LFPG_test_file, ESSA_LFPG_val_file)

### LOWW_EGLL ###
output_LOWW_EGLL = numpy_array_sampled(r"C:\Users\gungo\Downloads\LOWW_EGLL.csv", sample_rate)
LOWW_EGLL_train_array, LOWW_EGLL_test_array , LOWW_EGLL_val_array = array_split_seed(output_LOWW_EGLL)

output_dir = r"C:\Users\gungo\OneDrive\Desktop\A05 Data"  # Use raw string to avoid escaping backslashes

# Define the full paths for the output files
LOWW_EGLL_train_file = f"{output_dir}/LOWW_EGLL_train_dataa_n={sample_rate}.npz"
LOWW_EGLL_test_file = f"{output_dir}/LOWW_EGLL_test_dataa_n={sample_rate}.npz"
LOWW_EGLL_val_file = f"{output_dir}/LOWW_EGLL_val_dataa_n={sample_rate}.npz"

#save_arrays_to_npz(LOWW_EGLL_train_array, LOWW_EGLL_test_array, LOWW_EGLL_val_array, LOWW_EGLL_train_file, LOWW_EGLL_test_file, LOWW_EGLL_val_file)

### EHAM_LIMC ###
output_EHAM_LIMC = numpy_array_sampled(r"C:\Users\gungo\Downloads\EHAM_LIMC.csv", sample_rate)
EHAM_LIMC_train_array, EHAM_LIMC_test_array , EHAM_LIMC_val_array = array_split_seed(output_EHAM_LIMC)

output_dir = r"C:\Users\gungo\OneDrive\Desktop\A05 Data"  # Use raw string to avoid escaping backslashes

# Define the full paths for the output files
EHAM_LIMC_train_file = f"{output_dir}/EHAM_LIMC_train_dataa_n={sample_rate}.npz"
EHAM_LIMC_test_file = f"{output_dir}/EHAM_LIMC_test_dataa_n={sample_rate}.npz"
EHAM_LIMC_val_file = f"{output_dir}/EHAM_LIMC_val_dataa_n={sample_rate}.npz"

#save_arrays_to_npz(EHAM_LIMC_train_array, EHAM_LIMC_test_array, EHAM_LIMC_val_array, EHAM_LIMC_train_file, EHAM_LIMC_test_file, EHAM_LIMC_val_file)


result = load_single_array(r"C:\Users\gungo\Downloads\timeVAE_EHAM_LIMC_train_dataa_n=40_prior_samples.npz", "data")
print(result.shape)
array1 = EHAM_LIMC_train_array
array2 = result

# Plot comparison
plot_trajectories_comparison(
    array1, 
    array2,
    titles=('Original Trajectories', 'Generated Trajectories'),
    figsize=(18, 7)
)