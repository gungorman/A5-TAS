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

# Determine the number of unique flight IDs
    num_flight_ids = len(grouped_data)

# Determine the maximum number of timesteps (rows) for any flight
    max_timesteps = max(len(group) for _, group in grouped_data)

# Number of features (latitude, longitude, altitude, timedelta, flight_id)
    num_features = 5

# Initialize an empty NumPy array with the desired shape
    output_array = np.zeros((num_flight_ids, max_timesteps, num_features))

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

# Output the resulting NumPy array
    return(output_array)


# (5108,7468,4)

def array_split(output_array):

# Assuming output_array is already created as in your code
# output_array has shape (num_flight_ids, max_timesteps, num_features)

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

# Step 4: Use the indices to slice the output_array
    train_array = output_array[train_indices]
    test_array = output_array[test_indices]
    val_array = output_array[val_indices]

# Output the shapes of the resulting arrays
    print(f"Train array shape: {train_array.shape}")
    print(f"Test array shape: {test_array.shape}")
    print(f"Validation array shape: {val_array.shape}")


output = numpy_array(r"Code(ours)\LOWW_EGLL - Copy.csv")
array_split(output)
print(output.shape)

last_flight_id = output[-1,-1,-1]

print(f"Last flight_id: {int(last_flight_id)}")



print(number_of_flights(r"Code(ours)\LOWW_EGLL - Copy.csv"))
print(output)