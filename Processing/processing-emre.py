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

    print(f"Number of unique flight IDs: {num_unique_flight_ids}")

number_of_flights(r"C:\Users\gungo\Downloads\ESSA_LFPG.csv")