import pandas as pd
import numpy as np


def load_data(files):


    data = pd.read_csv(files)
    #data.drop(columns=["callsign", "icao24", "cluster", "timestamp"], inplace=True)
    x = data['latitude'].to_numpy()
    y = data['longitude'].to_numpy()
    z = data['altitude'].to_numpy()
    t = data['timedelta'].to_numpy()
    f = data['flight_id'].to_numpy()

    # Combine the arrays into a single numpy array
    #combined_array = np.column_stack((x, y, z, t))
    #print(load_data(r"C:\Users\gungo\Downloads\ESSA_LFPG.csv"))
    return combined_array


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
    train_size = int(0.015 * num_flight_ids)
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


### ESSA_LFPG ###
output = numpy_array(r"C:\Users\gungo\Downloads\ESSA_LFPG.csv")
ESSA_LFPG_train_array, ESSA_LFPG_test_array , ESSA_LFPG_val_array = array_split(output)

output_dir = r"C:\Users\gungo\OneDrive\Desktop\A05 Data"  # Use raw string to avoid escaping backslashes

# Define the full paths for the output files
ESSA_LFPG_train_file = f"{output_dir}/ESSA_LFPG_train_data.npz"
ESSA_LFPG_test_file = f"{output_dir}/ESSA_LFPG_test_data.npz"
ESSA_LFPG_val_file = f"{output_dir}/ESSA_LFPG_val_data.npz"

save_arrays_to_npz(ESSA_LFPG_train_array, ESSA_LFPG_test_array, ESSA_LFPG_val_array, ESSA_LFPG_train_file, ESSA_LFPG_test_file, ESSA_LFPG_val_file)

### LOWW_EGLL ###
output = numpy_array(r"C:\Users\gungo\Downloads\LOWW_EGLL.csv")
LOWW_EGLL_train_array, LOWW_EGLL_test_array , LOWW_EGLL_val_array = array_split(output)

output_dir = r"C:\Users\gungo\OneDrive\Desktop\A05 Data"  # Use raw string to avoid escaping backslashes

# Define the full paths for the output files
LOWW_EGLL_train_file = f"{output_dir}/LOWW_EGLL_train_data.npz"
LOWW_EGLL_test_file = f"{output_dir}/LOWW_EGLL_test_data.npz"
LOWW_EGLL_val_file = f"{output_dir}/LOWW_EGLL_val_data.npz"

save_arrays_to_npz(LOWW_EGLL_train_array, LOWW_EGLL_test_array, LOWW_EGLL_val_array, LOWW_EGLL_train_file, LOWW_EGLL_test_file, LOWW_EGLL_val_file)

### EHAM_LIMC ###
output = numpy_array(r"C:\Users\gungo\Downloads\EHAM_LIMC.csv")
EHAM_LIMC_train_array, EHAM_LIMC_test_array , EHAM_LIMC_val_array = array_split(output)

output_dir = r"C:\Users\gungo\OneDrive\Desktop\A05 Data"  # Use raw string to avoid escaping backslashes

# Define the full paths for the output files
EHAM_LIMC_train_file = f"{output_dir}/EHAM_LIMC_train_data.npz"
EHAM_LIMC_test_file = f"{output_dir}/EHAM_LIMC_test_data.npz"
EHAM_LIMC_val_file = f"{output_dir}/EHAM_LIMC_val_data.npz"

save_arrays_to_npz(EHAM_LIMC_train_array, EHAM_LIMC_test_array, EHAM_LIMC_val_array, EHAM_LIMC_train_file, EHAM_LIMC_test_file, EHAM_LIMC_val_file)